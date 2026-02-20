
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple, Literal
import numpy as np

from unicycle_partition_3d import Box3D, Partition3D
from unicycle_dyn import UnicycleClosedLoop
from helpers.math_utils import prepare_convex_hull_lp, convex_hull_intersects_box

Method = Literal["aabb","poly"]

@dataclass
class TransitionGraph:
    # leaf uid -> set of leaf uids (successors)
    succ: Dict[int, Set[int]]
    pred: Dict[int, Set[int]]

    def __init__(self) -> None:
        self.succ = {}
        self.pred = {}

    def set_succ(self, u: int, vs: Set[int]) -> None:
        # remove old edges
        old = self.succ.get(u, set())
        for v in old:
            self.pred.get(v, set()).discard(u)
        self.succ[u] = set(vs)
        for v in vs:
            self.pred.setdefault(v, set()).add(u)

    def successors(self, u: int) -> Set[int]:
        return set(self.succ.get(u, set()))

    def predecessors(self, u: int) -> Set[int]:
        return set(self.pred.get(u, set()))

class UnicycleAbstraction:
    """
    Adaptive partition abstraction over Partition3D leaves.
    Supports two global transition construction modes for the entire run:
      - AABB: successor if target leaf intersects the image AABB box (with theta wrap handled via split boxes)
      - POLY: candidate targets from AABB, then exact convex hull vs box feasibility test
    """
    OUT_UID = -1

    def __init__(self, part: Partition3D, dyn: UnicycleClosedLoop, *, method: Method = "aabb", allow_self_loops: bool = True, tol: float = 1e-9):
        self.part = part
        self.dyn = dyn
        self.method: Method = method
        self.allow_self_loops = bool(allow_self_loops)
        self.tol = float(tol)
        self.tr = TransitionGraph()
        # Cache per-leaf one-step image info so that after a split we can update
        # predecessor transitions without recomputing their images.
        # uid -> dict with keys: img_boxes, hits_oob, verts, theta_arc_start, lp
        self._img_cache: Dict[int, Dict[str, object]] = {}

    def ap_labels(self, box: Optional[Box3D]) -> List[str]:
        # used by runner for goal/unsafe; defined in run script for this case study
        raise NotImplementedError("set labeler externally")

    def rebuild_all(self, labeler, *, verbose: bool = False) -> None:
        self.ap_labels = labeler
        self.tr = TransitionGraph()
        self._img_cache = {}
        leaves = list(self.part.leaves.keys())
        if verbose:
            print(f"[abs] rebuilding transitions for {len(leaves)} leaves ({self.method})...")
        for idx,u in enumerate(leaves):
            self._rebuild_outgoing(u)
            if verbose and idx>0 and idx % 5000 == 0:
                print(f"  done {idx}/{len(leaves)}")
        # OUT self-loop
        self.tr.set_succ(self.OUT_UID, {self.OUT_UID})

    def rebuild_after_split(self, refined_uid: int) -> None:
        """
        After splitting a leaf uid into children, update transitions incrementally:
          - refined_uid is no longer a leaf; remove its outgoing
          - rebuild outgoing for each new child
          - rebuild outgoing for each predecessor that used to point to refined_uid (since refined_uid disappeared)
        """
        # predecessors that may target refined_uid
        preds = self.tr.predecessors(refined_uid)
        # remove refined_uid mapping if present
        self.tr.set_succ(refined_uid, set())
        # rebuild outgoing for new children (all current leaves under refined_uid)
        # We do not have a direct mapping; simplest: rebuild outgoing for children just created by Partition3D.refine_oct return value at call site.
        for p in preds:
            self._rebuild_outgoing(p)

    def _rebuild_outgoing(self, u: int) -> None:
        if u == self.OUT_UID:
            self.tr.set_succ(u, {self.OUT_UID})
            return
        if u not in self.part.leaves:
            # internal node; no outgoing
            self.tr.set_succ(u, set())
            return

        box_u = self.part.get_box(u)
        next_verts, img_boxes, hits_oob, theta_arc_start_u = self.dyn.image_from_box(box_u)

        # Cache image info for future incremental updates.
        info: Dict[str, object] = {
            "img_boxes": img_boxes,
            "hits_oob": bool(hits_oob),
            "theta_arc_start": float(theta_arc_start_u),
        }

        # candidate successors via AABB queries
        cand: Set[int] = set()
        for ib in img_boxes:
            cand |= self.part.query_intersecting_leaves(ib)

        succs: Set[int] = set()
        if self.method == "aabb":
            succs = cand
        else:
            # exact convex hull intersection test against each candidate leaf box
            # unwrap theta coordinates into a contiguous arc frame starting at theta_arc_start_u
            verts = next_verts.copy()
            # map theta to [0,2pi), then unwrap
            theta_lo = -np.pi
            period = 2*np.pi
            uvals = np.mod(verts[:,2] - theta_lo, period)
            uvals = np.where(uvals < theta_arc_start_u, uvals + period, uvals)
            verts[:,2] = theta_lo + uvals
            lp = prepare_convex_hull_lp(verts)
            info["verts"] = verts
            info["lp"] = lp

            for v in cand:
                b = self.part.get_box(v)
                bmin = np.array([b.p_lo, b.q_lo, b.th_lo], dtype=float)
                bmax = np.array([b.p_hi, b.q_hi, b.th_hi], dtype=float)
                # If theta interval is "low" and our verts are unwrapped above pi, also lift box by +2pi when needed
                # We'll test both placements for safety.
                ok = convex_hull_intersects_box(bmin, bmax, lp, tol=self.tol)
                if not ok:
                    bmin2 = bmin.copy(); bmax2 = bmax.copy()
                    bmin2[2] += period; bmax2[2] += period
                    ok = convex_hull_intersects_box(bmin2, bmax2, lp, tol=self.tol)
                if ok:
                    succs.add(v)

        if not self.allow_self_loops and u in succs:
            succs.discard(u)

        if hits_oob:
            succs.add(self.OUT_UID)

        # ensure total (no dead-ends): add self-loop if empty
        if len(succs) == 0:
            succs.add(u)

        self._img_cache[u] = info
        self.tr.set_succ(u, succs)

    def refine_split(self, uid: int) -> List[int]:
        """
        Split the given leaf into 8 children (oct split). Returns new child uids.
        """
        # Split leaf into children.
        kids = self.part.refine_oct(uid)
        if not kids:
            return []

        # Remove uid from cache and outgoing map (uid becomes internal).
        self._img_cache.pop(uid, None)
        self.tr.set_succ(uid, set())

        # Build outgoing transitions for each new child (this is the only place
        # we recompute images after a split).
        for k in kids:
            self._rebuild_outgoing(k)

        # Incrementally update predecessors that previously pointed to uid.
        # We do NOT recompute predecessor images; we only test intersection
        # against the 8 children using cached image information.
        preds = self.tr.predecessors(uid)
        kid_boxes = {k: self.part.get_box(k) for k in kids}

        for p in preds:
            succs = self.tr.successors(p)
            if uid not in succs:
                continue
            succs.discard(uid)

            info = self._img_cache.get(p)
            if info is None:
                # Fallback if cache missing.
                self._rebuild_outgoing(p)
                continue

            img_boxes = info["img_boxes"]

            if self.method == "aabb":
                for k, kb in kid_boxes.items():
                    for ib in img_boxes:
                        if kb.intersects(ib):
                            succs.add(k)
                            break
            else:
                lp = info.get("lp")
                if lp is None:
                    self._rebuild_outgoing(p)
                    continue
                period = 2*np.pi
                for k, kb in kid_boxes.items():
                    # AABB quick reject
                    ok_aabb = False
                    for ib in img_boxes:
                        if kb.intersects(ib):
                            ok_aabb = True
                            break
                    if not ok_aabb:
                        continue
                    bmin = np.array([kb.p_lo, kb.q_lo, kb.th_lo], dtype=float)
                    bmax = np.array([kb.p_hi, kb.q_hi, kb.th_hi], dtype=float)
                    ok = convex_hull_intersects_box(bmin, bmax, lp, tol=self.tol)
                    if not ok:
                        bmin2 = bmin.copy(); bmax2 = bmax.copy()
                        bmin2[2] += period; bmax2[2] += period
                        ok = convex_hull_intersects_box(bmin2, bmax2, lp, tol=self.tol)
                    if ok:
                        succs.add(k)

            if bool(info.get("hits_oob", False)):
                succs.add(self.OUT_UID)

            if (not self.allow_self_loops) and (p in succs):
                succs.discard(p)
            if len(succs) == 0:
                succs.add(p)
            self.tr.set_succ(p, succs)

        return kids
