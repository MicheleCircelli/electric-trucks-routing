from __future__ import annotations

import argparse
import time
from pathlib import Path

from utilis import compute_distance_matrix, load_instance_from_txt
from heuristic_deterministic import solve_deterministic, print_heuristic_solution


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the deterministic heuristic on a single instance.")
    p.add_argument(
        "--instance",
        type=str,
        default="instances/p4.3.b.txt",
        help="Path to the instance .txt file (default: instances/p4.3.b.txt).",
    )
    p.add_argument(
        "--grid-search-alpha",
        action="store_true",
        help="Enable grid-search for alpha in the savings-based construction.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    filepath = Path(args.instance)
    inst = load_instance_from_txt(filepath)
    t = compute_distance_matrix(inst.coords)

    t0 = time.perf_counter()
    routes = solve_deterministic(inst, grid_search_alpha=args.grid_search_alpha)
    elapsed = time.perf_counter() - t0

    print_heuristic_solution(routes, inst, t=t)
    print(f"Time (s): {elapsed:.3f} s")


if __name__ == "__main__":
    main()