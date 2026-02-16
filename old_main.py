# main.py
from __future__ import annotations

import importlib
import sys

from cegar_loop import run_cegar


def main() -> None:
    # Usage:
    #   python main.py systems.ges 100 100
    system_mod = sys.argv[1] if len(sys.argv) > 1 else "systems.ges"
    nx = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    ny = int(sys.argv[3]) if len(sys.argv) > 3 else 100

    mod = importlib.import_module(system_mod)
    spec = mod.build(nx=nx, ny=ny)

    print(f"[MAIN] system={system_mod}")
    print(f"[MAIN] leaves={len(spec.init_uids)}")
    print(f"[MAIN] phi={spec.phi}")

    res = run_cegar(
        absys=spec.absys,
        init_uids=spec.init_uids,
        phi=spec.phi,
        max_iters=200,
        max_steps_proxy=100,
        goal_all_fn=spec.goal_all_fn,
        min_cell_width=0.0005,
        min_cell_height=0.0005,
        max_refine_depth=12,
        verbose=True,
    )

    print("\nFINAL:", "VERIFIED" if res.verified else "NOT VERIFIED")
    print("iters:", res.iterations, "ignored:", res.ignored_counterexamples)


if __name__ == "__main__":
    main()

