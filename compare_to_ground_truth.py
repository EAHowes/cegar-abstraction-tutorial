import time
import argparse
import numpy as np
import sys

from unknown_worklist import classify_state_space_worklist
from krish_abstraction import KrishAbstraction
from abstraction import Rect, RectPartition
from helpers.systems.synthetic import SyntheticSystem

from helpers.model_checking_tools import SyntheticModelChecker, MountainCarModelChecker
from helpers.ground_truth_cache import build_gt_cache_path, load_gt_cache, save_gt_cache

sys.setrecursionlimit(200_000)

def uniform_grid_cells(domain: Rect, resolution: int) -> np.ndarray:
    """
    Build a uniform resolution x resolution grid of rectangles over `domain`.
    Returns np.ndarray shape (resolution^2, 4) with [xmin, xmax, ymin, ymax].
    """
    xs = np.linspace(domain.xmin, domain.xmax, resolution + 1)
    ys = np.linspace(domain.ymin, domain.ymax, resolution + 1)

    cells = np.zeros((resolution * resolution, 4), dtype=float)
    k = 0
    for i in range(resolution):
        for j in range(resolution):
            xmin, xmax = float(xs[i]), float(xs[i + 1])
            ymin, ymax = float(ys[j]), float(ys[j + 1])
            cells[k, :] = (xmin, xmax, ymin, ymax)
            k += 1
    return cells
# --------------------------------------------------
# helper: map point -> refined leaf uid
# --------------------------------------------------
def find_leaf_uid(absys, x, y):
    for uid, node in absys.part.leaves.items():
        r = node.rect
        if (r.xmin <= x <= r.xmax) and (r.ymin <= y <= r.ymax):
            return uid
    return None


# --------------------------------------------------
# build abstraction + run classification
# --------------------------------------------------
import importlib

def run_cegar(system_mod: str, nx: int, ny: int, budget: int, method: str, max_steps: int):
    mod = importlib.import_module(system_mod)
    spec = mod.build(nx=nx, ny=ny, method=method)

    absys = spec["absys"]
    phi = spec["phi"]
    goal_all_fn = spec["goal_all_fn"]

    t0 = time.perf_counter()
    cls, stats = classify_state_space_worklist(
        absys,
        phi,
        goal_all_fn=goal_all_fn,
        budget_steps=budget,
        max_steps_validator=max_steps,
    )
    build_time = time.perf_counter() - t0
    return absys, cls, build_time, stats, spec

def pick_checker(case_study: str):
    cs = (case_study or "").lower()
    if "mountain" in cs:
        return MountainCarModelChecker
    if "unicycle" in cs:
        return UnicycleModelChecker
    return SyntheticModelChecker

# --------------------------------------------------
# compute ground truth using Krish code
# --------------------------------------------------
def compute_ground_truth(absys, case_study: str, resolution: int, max_steps: int, *, cache_dir="gt_cache"):
    Checker = pick_checker(case_study)
    checker = Checker(absys.system)

    d = absys.part.domain
    domain_arr = np.array([d.xmin, d.xmax, d.ymin, d.ymax], dtype=float)

    cfg = {
        "domain": [d.xmin, d.xmax, d.ymin, d.ymax],
        "gt_grid_resolution": int(resolution),
        "gt_max_steps": int(max_steps),
    }
    cache_path = build_gt_cache_path(cache_dir, case_study, cfg)

    if cache_path.exists():
        return load_gt_cache(cache_path)

    gt_regions = checker.get_gt_reach_regions(domain_arr, resolution, max_steps)
    save_gt_cache(cache_path, gt_regions)
    return gt_regions


# --------------------------------------------------
# compare classification vs ground truth
# --------------------------------------------------
def evaluate(absys, cls, spec, gt_resolution, gt_max_steps):
    Checker = pick_checker(spec.get("case_study", "synthetic"))
    checker = Checker(absys.system)

    # checker = SyntheticModelChecker(absys.system)
    d = absys.part.domain
    domain_arr = np.array([d.xmin, d.xmax, d.ymin, d.ymax], dtype=float)

    # --- ground truth ---
    t0 = time.process_time()
    # gt_regions = compute_ground_truth(absys, gt_resolution, gt_max_steps)
    case_study = spec.get("case_study", "synthetic")
    gt_regions = compute_ground_truth(absys, case_study, gt_resolution, gt_max_steps)

    # uniform grid used by Krish
    cells = uniform_grid_cells(d, gt_resolution)

    gt_reference = checker.check_ground_truth_fast(
        cells,
        domain_arr,
        gt_regions,
    )

    # --- predicted SAT states from CEGAR ---
    verified = cls.verified
    sat_states = []

    for i, (xmin, xmax, ymin, ymax) in enumerate(cells):
        cx = 0.5 * (xmin + xmax)
        cy = 0.5 * (ymin + ymax)

        uid = find_leaf_uid(absys, cx, cy)
        if uid in verified:
            sat_states.append(i)

    # --- evaluate ---
    results = checker.evaluate_against_ground_truth(
        sat_states,
        cells,
        gt_reference,
        gt_regions,
    )

    verify_time = time.process_time() - t0
    return results, verify_time

def self_loop_proportion_s(transition_map) -> float:
    # EXACT implementation from log_utils.py
    self_loops = sum(1 for i, succ in enumerate(transition_map) if i in succ)
    n_states = len(transition_map)
    return self_loops / n_states if n_states > 0 else 0.0

# --------------------------------------------------
# main
# --------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--system", type=str, default="helpers.systems.synthetic",
                    help="Python module path with build(nx, ny, method=...)")
    parser.add_argument("--method", type=str, default="POLY", choices=["POLY", "AABB"])
    parser.add_argument("--max_steps", type=int, default=100)

    parser.add_argument("--nx", type=int, default=40)
    parser.add_argument("--ny", type=int, default=40)
    parser.add_argument("--budget", type=int, default=10000)
    parser.add_argument("--gt_grid_resolution", type=int, default=100)
    parser.add_argument("--gt_max_steps", type=int, default=100)
    args = parser.parse_args()

    print("\n[RUNNING CEGAR BUILD]")
    absys, cls, build_time, stats, spec = run_cegar(
        args.system,
        args.nx,
        args.ny,
        args.budget,
        args.method,
        args.max_steps,
    )


    print("\n[GROUND TRUTH COMPARISON]")
    results, verify_time = evaluate(
        absys,
        cls,
        spec,
        args.gt_grid_resolution,
        args.gt_max_steps,
    )

    # Build kripke once and reuse its transition_map (do NOT rebuild transitions)
    _checker, _kripke, _stats, _uid_to_idx, _idx_to_uid, _cells, transition_map = absys.build_kripke()
    s = self_loop_proportion_s(transition_map)

    n_states = len(transition_map) - 1

    # ---- metrics ----
    fnr = results["fnr"]
    tpr = 1.0 - fnr
    sr = results["coverage_proportion"]

    print("\n========== FINAL METRICS ==========")
    print(f"Build time:        {build_time:.3f}s")
    print(f"Verification time: {verify_time:.3f}s")
    print(f"X hat:               {n_states}")
    print(f"TPR:               {tpr:.4f}")
    print(f"FNR:               {fnr:.4f}")
    print(f"SR:                {sr:.4f}")
    print(f"Self-loop proportion (s): {s:.4f}")
    print("===================================\n")


if __name__ == "__main__":
    main()

