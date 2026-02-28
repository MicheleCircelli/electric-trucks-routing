# Team Orienteering -- Deterministic Heuristic and Simheuristic

This repository implements a deterministic heuristic and a stochastic
simheuristic for a multi-vehicle Team Orienteering Problem (TOP) with
depot-to-depot routes and time budget constraints.

The project is intentionally kept minimal and focused on the algorithmic
components.

------------------------------------------------------------------------

## Project Structure

.
├── src/
│   ├── utilis.py
│   ├── heuristic_deterministic.py
│   ├── simheuristic.py
│   ├── run_heuristic.py
│   └── run_simheuristic.py
├── instances/
│   ├── ...
└── README.md

### Main Components

-   **utilis.py**
    -   Data structures: `Instance`, `Route`
    -   Distance matrix computation
    -   Route metrics (length, reward, visited silos)
    -   Padding to exactly `m` routes
    -   Safety checks (no revisits)
-   **heuristic_deterministic.py**
    -   Savings-based construction (with optional RECOMENDED grid search on
        `alpha`)
    -   Intra-route 2-opt
    -   Greedy reinsertion
    -   Replacement swaps
    -   Formatted solution printing
-   **simheuristic.py**
    -   Lognormal calibration of travel times
    -   Randomized Top-L reinsertion and replacement
    -   Multi-start candidate generation
    -   Monte Carlo evaluation
    -   Reliability-aware selection rule
-   **run_heuristic.py**
    -   Runs the deterministic heuristic on a single instance.
-   **run_simheuristic.py**
    -   Runs the simheuristic on a single instance with configurable stochastic parameters.

------------------------------------------------------------------------

## How to Run

### Deterministic Heuristic

``` bash
python src/run_heuristic.py --instance instances/p4.3.b.txt --grid-search-alpha
```

Options: - `--instance` : path to instance file - `--grid-search-alpha`
: enables alpha grid search in savings construction

------------------------------------------------------------------------

### Simheuristic

``` bash
python src/run_simheuristic.py --instance instances/p4.2.b.txt --grid-search-alpha
```

Key parameters: - `--alpha` : savings weight (ignored if grid search
enabled) - `--c` : variability parameter for lognormal travel times -
`--K` : number of multi-start candidates - `--L-top` : Top-L parameter
for randomized move selection - `--N` : Monte Carlo sample size -
`--beta` : reliability threshold

------------------------------------------------------------------------

## Modeling Conventions

-   The depot is node `0`.
-   `Route.silos` does not include the depot; the depot is implicitly
    added when building explicit sequences.
-   Each silo can be visited at most once across all routes.
-   Solutions are padded to exactly `m` routes for reporting
    consistency.
-   In the stochastic setting, travel times follow a lognormal
    distribution calibrated from deterministic distances.
-   Reliability is computed as the average success rate across all
    trucks.
