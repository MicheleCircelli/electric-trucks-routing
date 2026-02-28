from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

from utilis import(
    compute_distance_matrix,
    load_instance_from_txt,
)

from simheuristic import(
    build_lognormal_mu_sigma,
    pre_sample_scenarios,
    solve_stochastic_simheuristic,
    print_simheuristic_solution
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the stochastic simheuristic on a single instance.")
    p.add_argument(
        "--instance",
        type=str,
        default="instances/p4.2.b.txt",
        help="Path to the instance .txt file (default: instances/p4.2.b.txt).",
    )

    # Simheuristic parameters (keep defaults equal to the previous script)
    p.add_argument("--alpha", type=float, default=0.5, help="Alpha weight in the savings score (ignored if --grid-search-alpha).")
    p.add_argument("--grid-search-alpha", action="store_true", help="Enable grid-search for alpha in {0.1,...,0.9}.")
    p.add_argument("--c", type=float, default=0.05, help="Variability parameter for lognormal travel times.")
    p.add_argument("--K", type=int, default=120, help="Number of randomized multi-start candidates.")
    p.add_argument("--L-top", dest="L_top", type=int, default=15, help="Top-L parameter for randomized move selection.")
    p.add_argument("--N", type=int, default=1000, help="Monte Carlo sample size.")
    p.add_argument("--beta", type=float, default=0.90, help="Reliability threshold in [0,1].")
    p.add_argument("--seed", type=int, default=0, help="Random seed.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    filepath = Path(args.instance)
    inst = load_instance_from_txt(filepath)
    t = compute_distance_matrix(inst.coords)

    # alpha grid (used only when grid_search_alpha=True)
    alphas = [i / 10 for i in range(1, 10)]  # 0.1..0.9

    # CRN scenarios (less noise when comparing alphas/candidates)
    rng = random.Random(args.seed)
    mu, sigma = build_lognormal_mu_sigma(t, args.c)
    TT = pre_sample_scenarios(inst, mu, sigma, args.N, rng)

    t0 = time.perf_counter()
    res = solve_stochastic_simheuristic(
        inst,
        alpha=args.alpha,
        c=args.c,
        K=args.K,
        L_top=args.L_top,
        N=args.N,
        beta=args.beta,
        seed=args.seed,
        TT=TT,
        grid_search_alpha=args.grid_search_alpha,
        alphas=alphas,
    )
    elapsed = time.perf_counter() - t0

    # print solution
    print_simheuristic_solution(
        res.routes,
        inst,
        F_hat=res.F_hat,
        R_hat=res.R_hat,
        p_hat=res.p_hat,
        t=t
    )
    print(f"Time (s): {elapsed:.3f}")


if __name__ == "__main__":
    main()