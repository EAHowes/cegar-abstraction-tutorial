import time
import argparse
import numpy as np

from unknown_worklist import classify_state_space_worklist
from krish_abstraction import KrishAbstraction
from abstraction import Rect, RectPartition
from helpers.systems.synthetic import SyntheticSystem
from helpers.model_checking_tools import SyntheticModelChecker
from helpers.ground_truth_cache import build_gt_cache_path, load_gt_cache, save_gt_cache


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
def run_cegar(nx, ny, budget):
    domain = Rect(-10.0, 10.0, -10.0, 10.0)
    part = RectPartition.uniform_grid(domain, nx, ny)
    system = SyntheticSystem()
    absys = KrishAbstraction(part=part, system=system, method="POLY")

    def goal_all_fn(points):
        center = np.array([5.0, 5.0])
        r2 = 2.0 ** 2
        d = points - center[None, :]
        return bool(np.all(np.sum(d * d, axis=1) <= r2))

    phi = "A (safe U goal)"

    # ---- BUILD TIMER START ----
    t0 = time.process_time()

    cls, stats = classify_state_space_worklist(
        absys,
        phi,
        goal_all_fn=goal_all_fn,
        budget_steps=budget,
    )

    build_time = time.process_time() - t0
    # ---- BUILD TIMER END ----

    return absys, cls, build_time


# --------------------------------------------------
# compute ground truth using Krish code
# --------------------------------------------------
def compute_ground_truth(absys, resolution, max_steps, cache_dir="gt_cache"):
    checker = SyntheticModelChecker(absys.system)
    d = absys.part.domain
    domain_arr = np.array([d.xmin, d.xmax, d.ymin, d.ymax], dtype=float)

    # Build a Krish-style cache key
    system_name = type(absys.system).__name__
    cfg = {
        "domain": [d.xmin, d.xmax, d.ymin, d.ymax],
        "gt_grid_resolution": int(resolution),
        "gt_max_steps": int(max_steps),
    }

    cache_path = build_gt_cache_path(cache_dir, system_name, cfg)

    if cache_path.exists():
        return load_gt_cache(cache_path)

    gt_regions = checker.get_gt_reach_regions(domain_arr, resolution, max_steps)
    save_gt_cache(cache_path, gt_regions)
    return gt_regions


# --------------------------------------------------
# compare classification vs ground truth
# --------------------------------------------------
def evaluate(absys, cls, gt_resolution, gt_max_steps):
    checker = SyntheticModelChecker(absys.system)
    d = absys.part.domain
    domain_arr = np.array([d.xmin, d.xmax, d.ymin, d.ymax], dtype=float)

    # --- ground truth ---
    t0 = time.process_time()
    gt_regions = compute_ground_truth(absys, gt_resolution, gt_max_steps)

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


# --------------------------------------------------
# main
# --------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nx", type=int, default=40)
    parser.add_argument("--ny", type=int, default=40)
    parser.add_argument("--budget", type=int, default=10000)
    parser.add_argument("--gt_grid_resolution", type=int, default=100)
    parser.add_argument("--gt_max_steps", type=int, default=100)
    args = parser.parse_args()

    print("\n[RUNNING CEGAR BUILD]")
    absys, cls, build_time = run_cegar(args.nx, args.ny, args.budget)

    print("\n[GROUND TRUTH COMPARISON]")
    results, verify_time = evaluate(
        absys,
        cls,
        args.gt_grid_resolution,
        args.gt_max_steps,
    )

    # ---- metrics ----
    fnr = results["fnr"]
    tpr = 1.0 - fnr
    sr = results["coverage_proportion"]

    print("\n========== FINAL METRICS ==========")
    print(f"Build time:        {build_time:.3f}s")
    print(f"Verification time: {verify_time:.3f}s")
    print(f"TPR:               {tpr:.4f}")
    print(f"FNR:               {fnr:.4f}")
    print(f"SR:                {sr:.4f}")
    print("===================================\n")


if __name__ == "__main__":
    main()

