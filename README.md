# Electric trucks routing with deterministic and stochastic travel times

This repository contains a solution to an orienteering-style routing problem for a fleet of electric trucks.
It includes:
- a deterministic heuristic (savings construction + 2-opt + reinsertion + swaps)
- an exact MILP benchmark (MTZ) for small instances (PuLP)
- a stochastic simheuristic using Monte Carlo evaluation and a reliability threshold

## Requirements

Python 3.10+ recommended.

Install dependencies:

```bash
pip install -r requirements.txt
```

**Solver note (MILP):** the code uses PuLP. It tries to use the HiGHS solver if available, and falls back to CBC otherwise.

## Instances

Input instances are stored in the `instances/` folder and follow this format:

- `n <int>`
- `m <int>`
- `tmax <float>`
- then one line per node: `x  y  reward`

Node `0` represents the depot.

## How to run

All commands must be executed from the repository root.

### 1) Stochastic comparison (Heuristic + MC vs Simheuristic)

This script:
- builds a deterministic heuristic solution
- evaluates it under uncertainty via Monte Carlo
- runs the simheuristic
- exports CSV and LaTeX tables

```bash
python src/compare_simheuristic_vs_det_eval.py
```

Outputs are automatically saved in the `results/` folder.

### 2) Deterministic comparison (Heuristic vs MILP)

```bash
python src/compare_heuristic_vs_MILP.py
```

### 3) Run the simheuristic on a single instance

```bash
python src/run_simheuristic.py
```

### 4) Run the heuristic on a single instance

```bash
python src/run_heuristic.py
```
