
from __future__ import annotations
import argparse, os, json, time, pickle
import numpy as np

from unicycle_partition_3d import Box3D, Partition3D
from unicycle_dyn import UnicycleClosedLoop, UnicycleParams
from abstraction import UnicycleAbstraction
from cegar import run_cegar

# Paper-spec constants
P_MIN, P_MAX = 0.0, 50.0
Q_MIN, Q_MAX = 0.0, 40.0
TH_MIN, TH_MAX = -np.pi, np.pi

GOAL_CENTER = np.array([40.0, 20.0], dtype=float)
GOAL_RADIUS = 8.0
OBS_CENTER = np.array([25.0, 25.0], dtype=float)
OBS_RADIUS = 5.0

def box_goal_all_corners(box: Box3D) -> bool:
    c = box.corners()
    d = np.linalg.norm(c[:, :2] - GOAL_CENTER[None, :], axis=1)
    return bool(np.all(d <= GOAL_RADIUS))

def box_unsafe_any_corner(box: Box3D) -> bool:
    c = box.corners()
    d = np.linalg.norm(c[:, :2] - OBS_CENTER[None, :], axis=1)
    if np.any(d <= OBS_RADIUS):
        return True
    if (
        np.any(c[:,0] < P_MIN) or np.any(c[:,0] > P_MAX) or
        np.any(c[:,1] < Q_MIN) or np.any(c[:,1] > Q_MAX)
    ):
        return True
    return False

def labeler(box: Box3D | None):
    if box is None:
        return ["unsafe"]
    if box_goal_all_corners(box):
        return ["goal"]
    if box_unsafe_any_corner(box):
        return ["unsafe"]
    return []

def load_gt_map(path: str) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)

def compute_sat_coverage_from_boxes(sat_uids, part: Partition3D) -> float:
    total_vol = (P_MAX-P_MIN)*(Q_MAX-Q_MIN)*(TH_MAX-TH_MIN)
    sat_vol = 0.0
    for u in sat_uids:
        b = part.get_box(u)
        sat_vol += (b.p_hi-b.p_lo)*(b.q_hi-b.q_lo)*(b.th_hi-b.th_lo)
    return sat_vol/total_vol

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["aabb","poly"], default="aabb", help="Global transition method for the entire run.")
    ap.add_argument("--nx", type=int, default=20)
    ap.add_argument("--ny", type=int, default=20)
    ap.add_argument("--nz", type=int, default=20)
    ap.add_argument("--horizon", type=int, default=100)
    ap.add_argument("--split-budget", type=int, default=0, help="Max number of leaf splits (partition refinements).")
    ap.add_argument("--max-iters", type=int, default=200, help="Hard cap on CEGAR iterations (safety).")
    ap.add_argument("--gt-cache", type=str, default=os.path.join("cache","unicycle_cfg_e5336e8c1848.pkl"))
    ap.add_argument("--outdir", type=str, default=os.path.join("out","unicycle_cegar"))
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    t0 = time.process_time()

    root = Box3D(P_MIN, P_MAX, Q_MIN, Q_MAX, TH_MIN, TH_MAX)
    part = Partition3D(root)
    part.make_uniform_grid(args.nx, args.ny, args.nz)

    dyn = UnicycleClosedLoop(UnicycleParams(
        p_bounds=(P_MIN,P_MAX),
        q_bounds=(Q_MIN,Q_MAX),
        goal_center=(float(GOAL_CENTER[0]), float(GOAL_CENTER[1])),
        goal_radius=GOAL_RADIUS,
        obs_center=(float(OBS_CENTER[0]), float(OBS_CENTER[1])),
        obs_radius=OBS_RADIUS,
    ))

    # Spatial hash bins: tie to initial resolution for good locality.
    absys = UnicycleAbstraction(
        part=part,
        dyn=dyn,
        method=args.method,
        allow_self_loops=True,
        bins=(args.nx, args.ny, args.nz),
    )
    absys.rebuild_all(labeler, verbose=args.verbose)

    init_uids = {u for u in absys.part.leaves.keys() if "unsafe" not in labeler(absys.part.get_box(u))}

    res = run_cegar(
        absys,
        init_uids,
        labeler,
        horizon=args.horizon,
        split_budget=args.split_budget,
        max_iters=args.max_iters,
        verbose=args.verbose,
    )

    build_cpu = time.process_time() - t0

    # Save classification
    class_path = os.path.join(args.outdir, "classification.pkl")
    with open(class_path, "wb") as f:
        pickle.dump(res.classification, f)

    # Metrics vs GT cache
    t1 = time.process_time()
    gt = load_gt_map(args.gt_cache)
    # GT coverage (goal regions only)
    gt_goal = sum(1 for v in gt.values() if v == "goal")
    gt_coverage = gt_goal / len(gt) if len(gt) else float("nan")

    # sat coverage (volume)
    sat_coverage = compute_sat_coverage_from_boxes(res.sat_uids, absys.part)
    sr = (sat_coverage / gt_coverage) if (gt_coverage and gt_coverage > 0) else float("nan")

    # For TPR/FNR we need true_goal at leaf level. We'll conservatively mark a leaf as true_goal if ALL GT voxels it covers are goal.
    # We'll reuse the helper in gt_helper_functions if available; else do containment sampling on GT grid by index ranges.
    # To keep repo self-contained, compute using index-range containment against GT grid.
    grid_res = 100
    n = grid_res - 1
    # precompute GT goal boolean array lazily not stored due to size; use dict lookup in ranges.
    def box_true_goal_under_gt(box: Box3D) -> bool:
        # map to GT indices
        def idx_range(lo, hi, lo_dom, hi_dom):
            # convert to [0,n) cells; clamp
            a = int(np.floor((lo - lo_dom) / (hi_dom - lo_dom) * n))
            b = int(np.ceil((hi - lo_dom) / (hi_dom - lo_dom) * n))
            return max(0,a), min(n,b)
        i0,i1 = idx_range(box.p_lo, box.p_hi, P_MIN, P_MAX)
        j0,j1 = idx_range(box.q_lo, box.q_hi, Q_MIN, Q_MAX)
        k0,k1 = idx_range(box.th_lo, box.th_hi, TH_MIN, TH_MAX)
        for i in range(i0,i1):
            for j in range(j0,j1):
                for k in range(k0,k1):
                    if gt.get((i,j,k)) != "goal":
                        return False
        return True

    leaves = list(absys.part.leaves.keys())
    true_goal_uids = {u for u in leaves if box_true_goal_under_gt(absys.part.get_box(u))}
    checked_sat = set(res.sat_uids)
    tp = len(true_goal_uids & checked_sat)
    fn = len(true_goal_uids - checked_sat)
    tpr = tp/len(true_goal_uids) if true_goal_uids else float("nan")
    fnr = fn/len(true_goal_uids) if true_goal_uids else float("nan")

    verify_cpu = time.process_time() - t1

    # mSucc / SLP
    succ_counts = []
    self_loops = 0
    for u in leaves:
        succ = absys.tr.successors(u)
        succ_counts.append(len(succ))
        if u in succ:
            self_loops += 1
    mSucc = float(np.mean(succ_counts)) if succ_counts else 0.0
    slp = float(self_loops/len(leaves)) if leaves else 0.0

    metrics = {
        "method": args.method,
        "nx": args.nx, "ny": args.ny, "nz": args.nz,
        "horizon": args.horizon,
        "split_budget": args.split_budget,
        "max_iters": args.max_iters,
        "build_time_cpu_s": build_cpu,
        "verification_time_cpu_s": verify_cpu,
        "total_states": len(leaves),
        "X_hat": len(leaves),
        "sat_states": len(res.sat_uids),
        "TPR": tpr,
        "FNR": fnr,
        "SR": sr,
        "gt_coverage": gt_coverage,
        "sat_coverage": sat_coverage,
        "mSucc": mSucc,
        "SLP": slp,
        "cegar_iters": res.iters,
        "refined_splits": len(res.refined_uids),
        "last_cex_len": len(res.last_cex),
    }
    with open(os.path.join(args.outdir,"metrics.json"),"w") as f:
        json.dump(metrics, f, indent=2)

    print("\n==================== RESULTS ====================")
    print(f"Method: {args.method.upper()}")
    print(f"Grid: nx={args.nx}, ny={args.ny}, nz={args.nz}  (|X_hat|={len(leaves)})")
    print(f"Horizon: {args.horizon}")
    print(f"Split budget: {args.split_budget}   Max iters: {args.max_iters}")
    print("-------------------------------------------------")
    print(f"Build time (CPU): {build_cpu:.6f} s")
    print(f"Verification time (CPU): {verify_cpu:.6f} s")
    print("-------------------------------------------------")
    print(f"CEGAR iters: {res.iters}")
    print(f"Refined splits: {len(res.refined_uids)}")
    print(f"Last counterexample length: {len(res.last_cex)}")
    print("-------------------------------------------------")
    print(f"TOTAL states (|X_hat|): {len(leaves)}")
    print(f"SAT states (|Sat|): {len(res.sat_uids)}")
    print(f"TPR: {tpr:.6f}")
    print(f"FNR: {fnr:.6f}")
    print(f"SR (coverage proportion): {sr:.6f}")
    print("-------------------------------------------------")
    print(f"mSucc (mean successors): {mSucc:.6f}")
    print(f"SLP (self-loop proportion): {slp:.6f}")
    print("=================================================")
    print(f"\nWrote: {class_path}")
    print(f"Wrote: {os.path.join(args.outdir,'metrics.json')}")

if __name__ == "__main__":
    main()
