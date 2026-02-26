from __future__ import annotations

import time
from pathlib import Path

from core import compute_distance_matrix, load_instance_from_txt, pad_routes_to_m, print_heuristic_solution
from heuristic_deterministic import solve_deterministic


def main() -> None:
    filepath = Path("instances/p4.3.b.txt")
    inst = load_instance_from_txt(filepath)
    t = compute_distance_matrix(inst.coords)

    t0 = time.perf_counter()
    routes = solve_deterministic(inst, grid_search_alpha=True)
    elapsed = time.perf_counter() - t0

    # show exactly m routes
    routes = pad_routes_to_m(routes, inst.m)

    print_heuristic_solution(routes, inst, t=t, title="Algorithm: Heuristic")

if __name__ == "__main__":
    main()
