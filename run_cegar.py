import argparse
import subprocess
import sys

# -------------------------
# Experiment defaults
# -------------------------
NX = 100
NY = 100
BUDGET = 2500
MAX_STEPS = 100
GT_RESOLUTION = 100
GT_MAX_STEPS = 100


def run(system, method):
    cmd = [
        sys.executable,
        "compare_to_ground_truth.py",
        "--system", system,
        "--method", method,
        "--nx", str(NX),
        "--ny", str(NY),
        "--budget", str(BUDGET),
        "--max_steps", str(MAX_STEPS),
        "--gt_grid_resolution", str(GT_RESOLUTION),
        "--gt_max_steps", str(GT_MAX_STEPS),
    ]

    print("\nRunning:", " ".join(cmd), "\n")
    subprocess.run(cmd)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=["synthetic", "mountaincar"], required=True, help="Case study to run")
    parser.add_argument("--method", choices=["AABB", "POLY"], required=True, help="Method to run")
    args = parser.parse_args()

    if args.case == "synthetic":
        run("helpers.systems.synthetic", args.method)
    elif args.case == "mountaincar":
        run("helpers.systems.mountain_car", args.method)


if __name__ == "__main__":
    main()
