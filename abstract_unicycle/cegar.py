
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple, Optional
import numpy as np

from unicycle_partition_3d import Box3D
from unicycle_dyn import UnicycleClosedLoop
from abstraction import UnicycleAbstraction

@dataclass
class CegarResult:
    classification: Dict[int, str]  # uid -> verified/refuted/unknown
    sat_uids: Set[int]
    refined_uids: Set[int]
    iters: int
    last_cex: List[int]

def bounded_A_safe_U_goal(
    uids: List[int],
    succ_list: List[np.ndarray],
    is_goal: np.ndarray,
    is_safe: np.ndarray,
    horizon: int
) -> np.ndarray:
    """
    Compute sat set for A(safe U goal) within bounded horizon.
    sat[t] = goal OR (safe AND AX(sat[t-1]))
    """
    n = len(uids)
    sat = is_goal.copy()
    # precompute for AX: for each state, list of successors indices
    for _ in range(horizon):
        # compute AX(sat): for each state, all successors in sat
        ax = np.ones(n, dtype=bool)
        for i in range(n):
            succ = succ_list[i]
            if succ.size == 0:
                ax[i] = True
            else:
                ax[i] = bool(np.all(sat[succ]))
        sat = is_goal | (is_safe & ax)
    return sat

def extract_counterexample(
    uids: List[int],
    succ_list: List[np.ndarray],
    is_goal: np.ndarray,
    is_safe: np.ndarray,
    sat: np.ndarray,
    horizon: int,
    init_set: Set[int],
) -> List[int]:
    """
    Produce one abstract counterexample path of length horizon+1 from some init uid not satisfying property.
    Strategy: greedily choose successor that stays outside sat_{t-1} when possible.
    """
    uid_to_i = {u:i for i,u in enumerate(uids)}
    # pick initial violating state
    start_uid = None
    for u in init_set:
        i = uid_to_i.get(u, None)
        if i is None:
            continue
        if not sat[i]:
            start_uid = u
            break
    if start_uid is None:
        return []

    path = [start_uid]
    cur = start_uid
    # Recompute sat layers backwards for extraction (cheap for small horizons)
    layers = []
    sat_prev = is_goal.copy()
    layers.append(sat_prev)
    for _ in range(horizon):
        ax = np.ones(len(uids), dtype=bool)
        for i in range(len(uids)):
            succ = succ_list[i]
            if succ.size:
                ax[i] = bool(np.all(sat_prev[succ]))
        sat_prev = is_goal | (is_safe & ax)
        layers.append(sat_prev)

    # layers[t] is sat after t iterations; final is layers[horizon]
    # We want to follow a witness for violation of layers[horizon]
    for t in range(horizon, 0, -1):
        i = uid_to_i[cur]
        succ = succ_list[i]
        if succ.size == 0:
            path.append(cur)
            continue
        # If not safe at cur, we can self-loop
        chosen = None
        # prefer a successor that violates layers[t-1] (keeping violation alive)
        for j in succ:
            if not layers[t-1][j]:
                chosen = uids[j]; break
        if chosen is None:
            chosen = uids[int(succ[0])]
        path.append(chosen)
        cur = chosen
    return path

def propagate_box_one_step(dyn: UnicycleClosedLoop, box: Box3D) -> Box3D:
    corners = box.corners()
    nxt = np.array([dyn.step(c) for c in corners], dtype=float)
    p_lo = float(nxt[:,0].min()); p_hi = float(nxt[:,0].max())
    q_lo = float(nxt[:,1].min()); q_hi = float(nxt[:,1].max())
    th_lo = float(nxt[:,2].min()); th_hi = float(nxt[:,2].max())
    return Box3D(p_lo, p_hi, q_lo, q_hi, th_lo, th_hi)

def validate_counterexample(
    absys: UnicycleAbstraction,
    dyn: UnicycleClosedLoop,
    path: List[int],
    labeler,
) -> Tuple[bool, Optional[int]]:
    """
    Concrete validation via set propagation:
      - Start set = current cell box
      - Propagate AABB (over corners) one step
      - Check intersects next cell box
    If at some step intersection is empty -> spurious, return (False, uid_to_refine)
    If all steps intersect and we encounter unsafe (any corner) before goal -> real counterexample.
    """
    if not path:
        return (False, None)

    cur_uid = path[0]
    cur_set = absys.part.get_box(cur_uid)

    for t in range(len(path)-1):
        # if current is goal, property satisfied along this prefix
        if "goal" in labeler(absys.part.get_box(cur_uid)):
            return (False, None)

        nxt_uid = path[t+1]
        nxt_box = absys.part.get_box(nxt_uid) if nxt_uid != absys.OUT_UID else None

        img = propagate_box_one_step(dyn, cur_set)
        if nxt_box is None:
            # OUT reached -> treat as unsafe reach => real cex
            return (True, None)

        inter = img.intersection(nxt_box)
        if inter is None:
            # spurious at transition from cur_uid -> nxt_uid
            return (False, cur_uid)

        # advance
        cur_uid = nxt_uid
        cur_set = inter

        # If unsafe at this step (any corner unsafe), it's a real counterexample
        if "unsafe" in labeler(absys.part.get_box(cur_uid)):
            return (True, None)

    return (False, None)

def run_cegar(
    absys: UnicycleAbstraction,
    init_uids: Set[int],
    labeler,
    *,
    horizon: int,
    split_budget: int,
    max_iters: int,
    verbose: bool = False,
) -> CegarResult:
    refined: Set[int] = set()
    last_cex: List[int] = []

    for it in range(max_iters):
        leaves = sorted(list(absys.part.leaves.keys()))
        uid_to_i = {u:i for i,u in enumerate(leaves)}
        # build successor lists in index-space
        succ_list = []
        for u in leaves:
            vs = [v for v in absys.tr.successors(u) if v in uid_to_i]  # ignore OUT in dp
            succ_list.append(np.array([uid_to_i[v] for v in vs], dtype=np.int64))

        # labels
        is_goal = np.array([("goal" in labeler(absys.part.get_box(u))) for u in leaves], dtype=bool)
        is_safe = np.array([("unsafe" not in labeler(absys.part.get_box(u))) for u in leaves], dtype=bool)

        sat = bounded_A_safe_U_goal(leaves, succ_list, is_goal, is_safe, horizon)
        sat_uids = {u for u in leaves if sat[uid_to_i[u]]}

        # pick counterexample if any init violates
        violating = [u for u in init_uids if u in uid_to_i and (not sat[uid_to_i[u]])]
        if not violating:
            last_cex = []
            break

        last_cex = extract_counterexample(leaves, succ_list, is_goal, is_safe, sat, horizon, init_uids)
        if not last_cex:
            break

        real, refine_uid = validate_counterexample(absys, absys.dyn, last_cex, labeler)
        if real:
            # mark start as refuted by making it unsafe? classification later uses sat; keep as unknown/refuted
            # To preserve Clarke-style, we record refuted start cells separately by labeling unsafe? We'll return classification mapping later.
            # Here we just stop (one counterexample found).
            break

        if refine_uid is None:
            # spurious but couldn't identify refinement location -> stop
            break

        if len(refined) >= split_budget:
            break

        if refine_uid in refined:
            # already refined; avoid spinning
            break

        # split
        absys.refine_split(refine_uid)
        refined.add(refine_uid)

        if verbose:
            print(f"[cegar] refined uid={refine_uid} ({len(refined)}/{split_budget})")

    # final classification based on sat and trivial labels
    leaves = sorted(list(absys.part.leaves.keys()))
    uid_to_i = {u:i for i,u in enumerate(leaves)}
    succ_list = []
    for u in leaves:
        vs = [v for v in absys.tr.successors(u) if v in uid_to_i]
        succ_list.append(np.array([uid_to_i[v] for v in vs], dtype=np.int64))
    is_goal = np.array([("goal" in labeler(absys.part.get_box(u))) for u in leaves], dtype=bool)
    is_safe = np.array([("unsafe" not in labeler(absys.part.get_box(u))) for u in leaves], dtype=bool)
    sat = bounded_A_safe_U_goal(leaves, succ_list, is_goal, is_safe, horizon)
    sat_uids = {u for u in leaves if sat[uid_to_i[u]]}

    classification: Dict[int,str] = {}
    for u in leaves:
        labs = labeler(absys.part.get_box(u))
        if "unsafe" in labs:
            classification[u] = "refuted"
        elif sat[uid_to_i[u]]:
            classification[u] = "verified"
        else:
            classification[u] = "unknown"

    return CegarResult(
        classification=classification,
        sat_uids=sat_uids,
        refined_uids=refined,
        iters=it+1,
        last_cex=last_cex
    )
