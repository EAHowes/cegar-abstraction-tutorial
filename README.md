# Ethan CEGAR on Krish Abstraction (Transitions + Kripke + CTL aligned)

This repo keeps Ethan's original **Clarke-style CEGAR loop** and **local refinement**
(split/purge), but plugs in Krish's abstraction stack **end-to-end**:

1. **Transitions:** Krish AABB / Poly(convex hull) / Sampling builders,
   generalized to Ethan's locally-refined `RectPartition`.
2. **Kripke + labeling:** Krish `SyntheticModelChecker.create_kripke(...)`
   (labels: `safe`, `goal`, `fail`, with an optional OOB sink).
3. **CTL model checking objective:** Krish default formula:
   `A (safe U goal)`

Because CTL model checking doesn't automatically hand us a lasso, we build a
violating lasso witness using graph search (`helpers/witness_ctl.py`) on the
Kripke returned by Krish's checker. This lasso is then validated and refined
using Ethan's original reachable-set propagation and local refinement operations.

## Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py
```


## Whole-state-space classification (Verified / Refuted / Unknown)

Run the worklist classifier (continues refining even after real counterexamples):

```bash
python unknown_worklist.py helpers.systems.synthetic 40 40 10000
```

Outputs:
- `classification_worklist.png`
- prints counts of Verified / Refuted / Unknown

`refine_whole_space.py` performs a *single-pass* classification (no refinement), useful for quick snapshots.
