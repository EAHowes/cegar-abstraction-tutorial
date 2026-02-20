# Unicycle CEGAR (Partition-Refinement) on Krish Abstraction

This repo runs a Clarke-style **local partition-refinement CEGAR** loop for the 3D unicycle study.

## Key semantics (matches your requirements)

- **Global transition method** is chosen once and used for the entire run:
  - `--method aabb` uses AABB overlap transitions
  - `--method poly` uses polytope (conv-hull) intersection transitions (candidates pruned by AABB first)
- **Refinement is partition refinement**:
  - when a spurious counterexample is found, we **split a leaf cell** (oct-split in p,q,theta)
  - transitions for new child cells are computed using the **same global method**
- `--split-budget` limits the **number of splits** (refinements).
- `--max-iters` is a safety cap on the number of CEGAR iterations.

Ground truth cache is loaded from `cache/unicycle_cfg_e5336e8c1848.pkl`.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python run_unicycle_cegar.py --method aabb --nx 20 --ny 20 --nz 20 --split-budget 200 --horizon 100
python run_unicycle_cegar.py --method poly --nx 20 --ny 20 --nz 20 --split-budget 200 --horizon 100
```

Outputs:
- `out/unicycle_cegar/classification.pkl` (uid -> verified/refuted/unknown)
- `out/unicycle_cegar/metrics.json`
