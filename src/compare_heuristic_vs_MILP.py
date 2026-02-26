from __future__ import annotations

import time
from pathlib import Path
from typing import List, Tuple

import pandas as pd

from core import (
    Instance,
    Route,
    compute_distance_matrix,
    load_instance_from_txt,
    pad_routes_to_m,
    routes_to_explicit,
)
from heuristic_deterministic import solve_deterministic
from MILP_model import solve_exact_ilp_mtz


def build_tables(
    inst: Instance,
    t: List[List[float]],
    heur_routes: List[Route],
    ilp_routes_explicit: List[List[int]],
    heur_time: float,
    ilp_time: float,
    ilp_status: str,
    ilp_obj: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    
    # Heuristic
    heur_routes_padded = pad_routes_to_m(heur_routes, inst.m)
    heur_explicit = routes_to_explicit(heur_routes_padded)

    heur_visited = {i for r in heur_explicit for i in r if i != 0}
    heur_reward = sum(inst.reward[i] for i in heur_visited)
    heur_trucks_used = sum(1 for r in heur_routes if r.silos)

    # MILP
    ilp_routes_padded = ilp_routes_explicit + [[0, 0]] * max(0, inst.m - len(ilp_routes_explicit))

    ilp_visited = {i for r in ilp_routes_explicit for i in r if i != 0}
    ilp_reward = sum(inst.reward[i] for i in ilp_visited)
    ilp_trucks_used = sum(1 for r in ilp_routes_explicit if len(r) > 2)

    # Summary table
    summary = pd.DataFrame(
        [
            {
                "Algorithm": "Heuristic",
                "Status": "—",
                "Time (s)": round(heur_time, 3),
                "Total reward": round(heur_reward, 2),
                "Silos": len(heur_visited),
                "Trucks": heur_trucks_used,
                "Routes": inst.m,
            },
            {
                "Algorithm": "MILP",
                "Status": ilp_status,
                "Time (s)": round(ilp_time, 3),
                "Total reward": round(ilp_reward, 2),
                "Silos": len(ilp_visited),
                "Trucks": ilp_trucks_used,
                "Routes": inst.m,
            },
        ]
    )

    # Relative gap vs MILP (using realized reward from extracted routes)
    if ilp_reward > 1e-9:
        gap = (ilp_reward - heur_reward) / ilp_reward
        summary.loc[summary["Algorithm"] == "Heuristic", "Gap (%)"] = round(100 * gap, 2)
        summary.loc[summary["Algorithm"] == "MILP", "Gap (%)"] = 0.0
    else:
        summary["Gap (%)"] = None

    # Routes table
    rows = []
    for alg_name, routes in [("Heuristic", heur_explicit), ("MILP", ilp_routes_padded)]:
        for k, r in enumerate(routes, start=1):
            length = sum(t[r[i]][r[i + 1]] for i in range(len(r) - 1))
            reward = sum(inst.reward[i] for i in r if i != 0)

            rows.append(
                {
                    "Algorithm": alg_name,
                    "Truck": k,
                    # LaTeX-friendly arrow (requires escape=False in to_latex)
                    "Route": r" $\rightarrow$ ".join(map(str, r)),
                    "Silos": sum(1 for i in r if i != 0),
                    "Reward": round(reward, 2),
                    "Length": round(length, 2),
                    "Slack": round(inst.Tmax - length, 2),
                }
            )

    routes_df = pd.DataFrame(rows).sort_values(["Algorithm", "Truck"])
    return summary, routes_df


def main() -> None:
    filepath = Path("instances/p4.4.b_modified.txt")
    outdir = Path("results")
    outdir.mkdir(parents=True, exist_ok=True)

    inst = load_instance_from_txt(filepath)
    t = compute_distance_matrix(inst.coords)

    # Heuristic
    t0 = time.perf_counter()
    heur_routes = solve_deterministic(inst, grid_search_alpha=True)
    heur_time = time.perf_counter() - t0

    # Warm start for MILP from heuristic
    mip_start_routes = routes_to_explicit(pad_routes_to_m(heur_routes, inst.m))

    # MILP
    t1 = time.perf_counter()
    ilp_status, ilp_obj, _max_frac, ilp_routes = solve_exact_ilp_mtz(
        inst,
        time_limit_sec=1800,
        mip_gap=0.0,
        msg=False,
        mip_start_routes=mip_start_routes,
    )
    ilp_time = time.perf_counter() - t1

    # Build tables
    df_summary, df_routes = build_tables(
        inst=inst,
        t=t,
        heur_routes=heur_routes,
        ilp_routes_explicit=ilp_routes,
        heur_time=heur_time,
        ilp_time=ilp_time,
        ilp_status=ilp_status,
        ilp_obj=ilp_obj,
    )

    # Save CSV
    df_summary.to_csv(outdir / "Heuristic_vs_MILP_summary.csv", index=False)
    df_routes.to_csv(outdir / "Heuristic_vs_MILP_routes.csv", index=False)

    # Save LaTeX
    latex_dir = outdir / "latex"
    latex_dir.mkdir(parents=True, exist_ok=True)

    df_summary.to_latex(
        latex_dir / "Heuristic_vs_MILP_summary.tex",
        index=False,
        float_format="%.2f",
        longtable=False,
        escape=True,
        caption="Deterministic comparison between heuristic and MILP.",
        label="tab:det_compare_summary",
    )

    df_routes.to_latex(
        latex_dir / "Heuristic_vs_MILP_routes.tex",
        index=False,
        float_format="%.2f",
        longtable=True,
        escape=False,
        caption="Routes details for heuristic and MILP.",
        label="tab:det_compare_routes",
    )

    # Print + paths
    print(df_summary.to_string(index=False))
    print("\nSaved:")
    print(f" - {outdir / 'Heuristic_vs_MILP_summary.csv'}")
    print(f" - {outdir / 'Heuristic_vs_MILP_routes.csv'}")
    print(f" - {latex_dir / 'Heuristic_vs_MILP_summary.tex'}")
    print(f" - {latex_dir / 'Heuristic_vs_MILP_routes.tex'}")


if __name__ == "__main__":
    main()