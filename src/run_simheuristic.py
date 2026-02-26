from __future__ import annotations

import random
import time
from pathlib import Path

from core import compute_distance_matrix, load_instance_from_txt, pad_routes_to_m, print_simheuristic_solution
from simheuristic import (
    build_lognormal_mu_sigma,
    pre_sample_scenarios,
    solve_stochastic_simheuristic,
)


def main() -> None:
    filepath = Path("instances/p4.3.b.txt")
    inst = load_instance_from_txt(filepath)
    t = compute_distance_matrix(inst.coords)

    # parameters
    alpha = 0.5  # ignored when grid_search_alpha=True
    c = 0.05
    K = 20
    L_top = 5
    N = 200
    beta = 0.90
    seed = 0

    # alpha grid
    alphas = [i / 10 for i in range(1, 10)]  # 0.1..0.9

    # CRN scenarios (less noise when comparing alphas/candidates)
    rng = random.Random(seed)
    mu, sigma = build_lognormal_mu_sigma(t, c)
    TT = pre_sample_scenarios(inst, mu, sigma, N, rng)

    t0 = time.perf_counter()
    res = solve_stochastic_simheuristic(
        inst,
        alpha=alpha,
        c=c,
        K=K,
        L_top=L_top,
        N=N,
        beta=beta,
        seed=seed,
        TT=TT,
        grid_search_alpha=True,
        alphas=alphas,
    )
    elapsed = time.perf_counter() - t0

    routes = pad_routes_to_m(res.routes, inst.m)

    print_simheuristic_solution(
        routes,
        inst,
        F_hat=res.F_hat,
        R_hat=res.R_hat,
        p_hat=res.p_hat,
        t=t,
        title="Algorithm: Simheuristic",
    )


if __name__ == "__main__":
    main()