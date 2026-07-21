![CEGAR False Negative Map via AABB](/images/false_negative_map_aabb.png)
> CEGAR False Negative Visualization via AABB

# cegar-abstraction-tutorial

A worklist-based CEGAR (Counterexample-Guided Abstraction Refinement) tool for formal verification. It abstracts a continuous state space into a partition (rectangles or polytopes), builds transitions between cells, checks a CTL-style spec, and refines cells found to be "unknown" until the classification converges. Three case studies are included: **synthetic**, **mountain car**, and a 3D **unicycle** model.

## Repo layout

- Root: 2D case studies (synthetic, mountain car): `abstraction.py`, `cegar_loop.py`, `kmeans_abstraction.py`, `compare_to_ground_truth.py`, `helpers/`.
- `abstract_unicycle/`: an independent 3D code path for the unicycle case study, with its own `abstraction.py`, `cegar.py`, and vendored `helpers/`.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

`abstract_unicycle/` has its own `requirements.txt` (numpy, scipy only, no `pyModelChecking`) if you want to run the unicycle case in isolation.

## Running the CEGAR loop

The unified entry point is `run_cegar.py`, which dispatches to the right case study:

```bash
python run_cegar.py --case {synthetic,mountaincar,unicycle} --method {AABB,POLY}
```

For `unicycle`, `--method` only affects the initial partition method (`POLY` → `--init-method poly`); refinement always uses AABB.

## Running case studies directly

For custom parameters, invoke the underlying scripts directly.

Synthetic / mountain car (`compare_to_ground_truth.py`):

```bash
python compare_to_ground_truth.py --system helpers.systems.synthetic --method AABB \
  --nx 8 --ny 8 --budget 50 --max_steps 20 --gt_grid_resolution 20 --gt_max_steps 20
```

Unicycle (run from inside `abstract_unicycle/`):

```bash
cd abstract_unicycle
python run_unicycle_cegar.py --init-method aabb --refine-method aabb \
  --nx 4 --ny 4 --nz 4 --split-budget 5 --max-iters 5 --horizon 5 \
  --gt-cache cache/unicycle_cfg_e5336e8c1848.pkl
```

## Simple demo

`main.py` is a minimal, argument-free test (synthetic system + `KMeansAbstraction`):

```bash
python main.py
```

## Tests

```bash
pytest tests/ -q
```
