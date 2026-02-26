from __future__ import annotations

import random
import time
from pathlib import Path
from typing import List, Tuple

import pandas as pd

from core import (
    compute_distance_matrix,
    load_instance_from_txt,
    pad_routes_to_m,
    routes_to_explicit,
    visited_silos,
)
from heuristic_deterministic import solve_deterministic
from simheuristic import (
    build_lognormal_mu_sigma,
    pre_sample_scenarios,
    evaluate_solution_mc_presampled,
    solve_stochastic_simheuristic,
)

# Helpers
def route_to_latex(route: List[int]) -> str:
    """Convert explicit route [0, i1, ..., 0] to LaTeX math string."""
    return r" \to ".join(map(str, route))


def latex_escape_text(s: str) -> str:
    """Escape for LaTeX text (captions, table cells)."""
    return s.replace("_", r"\_")


def latex_safe_label(s: str) -> str:
    """Make a LaTeX-safe label string."""
    out = str(s)
    for ch in ["\\", "{", "}", " ", "#", "%", "&", "$", "^", "~"]:
        out = out.replace(ch, "")
    out = out.replace(".", "").replace("-", "").replace("_", "")
    return out


def percent_improvement(sim: float, det: float) -> float:
    """
    Percent improvement of Simheuristic vs Heuristic+MC based on E[Reward]:
      100*(sim-det)/det
    If det is 0, returns 0.0.
    """
    if abs(det) <= 1e-12:
        return 0.0
    return 100.0 * (sim - det) / det

# LaTeX writers
def write_summary_table_tex(df_summary: pd.DataFrame, outpath: Path, caption: str, label: str) -> None:
    """
    Write a LaTeX table (booktabs) for the multi-instance summary.
    """
    lines: List[str] = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(rf"\caption{{{caption}}}")
    lines.append(rf"\label{{{latex_safe_label(label)}}}")

    # Instance | n | m | Tmax | Algorithm | #Visited | Reward | E[Reward] | Reliability | %Improve | Time
    lines.append(r"\begin{tabular}{lrrr l r r r r r r}")
    lines.append(r"\toprule")
    lines.append(
        r"Instance & $n$ & $m$ & $T_{\max}$ & Algorithm & \#Visited & Reward & "
        r"$\mathbb{E}[\mathrm{Reward}]$ & Reliability & \%Impr. & Time (s) \\"
    )
    lines.append(r"\midrule")

    # Custom order so baseline appears first
    order = {"Heuristic + MC": 0, "Simheuristic": 1}
    df = df_summary.copy()
    df["_ord"] = df["Algorithm"].map(order).fillna(99)
    df = df.sort_values(["Instance", "_ord"]).drop(columns=["_ord"])

    last_inst = None
    for _, row in df.iterrows():
        inst = latex_escape_text(str(row["Instance"]))
        n = int(row["n"])
        m = int(row["m"])
        tmax = f'{float(row["T_max"]):.2f}'
        alg = latex_escape_text(str(row["Algorithm"]))
        nvis = int(row["Visited"])
        reward = f'{float(row["Reward"]):.2f}'
        e_reward = f'{float(row["E[Reward]"]):.2f}'
        rel = f'{float(row["Reliability"]):.4f}'
        pimpr = row["Percent improvement"]
        pimpr_s = "" if pd.isna(pimpr) else f'{float(pimpr):.1f}'
        t_s = f'{float(row["Time (s)"]):.3f}'

        if last_inst is not None and inst != last_inst:
            lines.append(r"\midrule")
        last_inst = inst

        lines.append(
            f"{inst} & {n:d} & {m:d} & {tmax} & {alg} & {nvis:d} & {reward} & "
            f"{e_reward} & {rel} & {pimpr_s} & {t_s} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    outpath.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_route_paths_tables_tex(
    df_routes: pd.DataFrame,
    outpath: Path,
    *,
    caption_prefix: str = "Route paths for instance",
    label_prefix: str = "tab:routes_",
) -> None:
    """
    One LaTeX table per instance:
      Algorithm | Truck | #Visited | Route
    """
    pieces: List[str] = []
    df = df_routes.copy()

    for inst_name, g_inst in df.groupby("Instance", sort=True):
        inst_text = latex_escape_text(str(inst_name))
        label = latex_safe_label(f"{label_prefix}{inst_name}")

        pieces.append(r"\begin{table}[t]")
        pieces.append(r"\centering")
        pieces.append(r"\small")
        pieces.append(rf"\caption{{{caption_prefix} \texttt{{{inst_text}}}.}}")
        pieces.append(rf"\label{{{label}}}")

        pieces.append(r"\begin{tabularx}{\textwidth}{llrX}")
        pieces.append(r"\toprule")
        pieces.append(r"Algorithm & Truck & \#Visited & Route \\")
        pieces.append(r"\midrule")

        # Custom order so heuristic routes appear first
        order = {"Heuristic": 0, "Simheuristic": 1}
        gg = g_inst.copy()
        gg["_ord"] = gg["Algorithm"].map(order).fillna(99)
        gg = gg.sort_values(["_ord", "Truck"]).drop(columns=["_ord"])

        for _, row in gg.iterrows():
            alg = latex_escape_text(str(row["Algorithm"]))
            truck = int(row["Truck"])
            silos = int(row["Silos"])
            route = str(row["Route"]).strip()
            if not route:
                route = r"0 \to 0"
            pieces.append(f"{alg} & {truck} & {silos} & $ {route} $ \\\\")
        pieces.append(r"\bottomrule")
        pieces.append(r"\end{tabularx}")
        pieces.append(r"\end{table}")
        pieces.append("")

    outpath.write_text("\n".join(pieces), encoding="utf-8")

# Experiment runner
def run_one_instance(
    filepath: Path,
    *,
    c: float,
    N: int,
    beta: float,
    seed: int,
    alpha: float,
    K: int,
    L_top: int,
) -> Tuple[List[dict], List[dict]]:
    inst_name = filepath.stem

    inst = load_instance_from_txt(filepath)
    t = compute_distance_matrix(inst.coords)

    # CRN scenarios
    mu, sigma = build_lognormal_mu_sigma(t, c)
    rng_scen = random.Random(seed)
    TT = pre_sample_scenarios(inst, mu, sigma, N, rng_scen)

    # deterministic heuristic (baseline)
    t0 = time.perf_counter()
    det_routes = solve_deterministic(inst, grid_search_alpha=True)
    det_time = time.perf_counter() - t0
    det_routes = pad_routes_to_m(det_routes, inst.m)

    det_reward = float(sum(inst.reward[i] for i in visited_silos(det_routes)))
    det_visited = len(visited_silos(det_routes))
    det_eval = evaluate_solution_mc_presampled(inst, det_routes, TT)

    # simheuristic
    t1 = time.perf_counter()
    sim_res = solve_stochastic_simheuristic(
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
    )
    sim_time = time.perf_counter() - t1
    sim_routes = pad_routes_to_m(sim_res.routes, inst.m)

    sim_reward = float(sum(inst.reward[i] for i in visited_silos(sim_routes)))
    sim_visited = len(visited_silos(sim_routes))
    pimpr = percent_improvement(sim_res.F_hat, det_eval.F_hat)

    # Summary rows
    summary_rows = [
        {
            "Instance": inst_name,
            "n": inst.n,
            "m": inst.m,
            "T_max": inst.Tmax,
            "Algorithm": "Heuristic + MC",
            "Visited": det_visited,
            "Reward": round(det_reward, 2),
            "E[Reward]": round(det_eval.F_hat, 2),
            "Reliability": round(det_eval.R_hat, 4),
            "Percent improvement": float("nan"),
            "Time (s)": round(det_time, 3),
        },
        {
            "Instance": inst_name,
            "n": inst.n,
            "m": inst.m,
            "T_max": inst.Tmax,
            "Algorithm": "Simheuristic",
            "Visited": sim_visited,
            "Reward": round(sim_reward, 2),
            "E[Reward]": round(sim_res.F_hat, 2),
            "Reliability": round(sim_res.R_hat, 4),
            "Percent improvement": round(pimpr, 1),
            "Time (s)": round(sim_time, 3),
        },
    ]

    # Route paths rows
    det_exp = routes_to_explicit(det_routes)
    sim_exp = routes_to_explicit(sim_routes)

    route_rows: List[dict] = []
    for alg, routes in [("Heuristic", det_exp), ("Simheuristic", sim_exp)]:
        for k, r in enumerate(routes, start=1):
            silos_count = sum(1 for x in r if x != 0)
            route_rows.append(
                {
                    "Instance": inst_name,
                    "Algorithm": alg,
                    "Truck": k,
                    "Silos": silos_count,
                    "Route": route_to_latex(r),
                }
            )

    return summary_rows, route_rows


def main() -> None:
    # Instances
    instances: List[Path] = [
        Path("instances/p4.2.b.txt"),
        Path("instances/p4.2.h.txt"),
        Path("instances/p4.2.t.txt"),
        Path("instances/p4.3.b.txt"),
        Path("instances/p4.3.k.txt"),
        Path("instances/p4.3.t.txt"),
        Path("instances/p4.4.d.txt"),
        Path("instances/p4.4.l.txt"),
        Path("instances/p4.4.t.txt"),
    ]

    # Safety check: skip missing files instead of crashing
    instances = [p for p in instances if p.exists()]
    if not instances:
        raise FileNotFoundError("No instance files found. Check paths / working directory.")

    outdir = Path("results")
    outdir.mkdir(parents=True, exist_ok=True)
    latex_dir = outdir / "latex"
    latex_dir.mkdir(parents=True, exist_ok=True)

    # Output filenames
    SUMMARY_CSV = "Heuristic_vs_Simheuristic_summary.csv"
    ROUTES_CSV = "Heuristic_vs_Simheuristic_routes.csv"
    SUMMARY_TEX = "Heuristic_vs_Simheuristic_summary.tex"
    ROUTES_TEX = "Heuristic_vs_Simheuristic_routes.tex"

    # Stochastic parameters
    c = 0.05
    N = 1000
    beta = 0.90
    seed = 0

    # Simheuristic parameters
    alpha = 0.5
    K = 120
    L_top = 15

    all_summary: List[dict] = []
    all_routes: List[dict] = []

    for fp in instances:
        summary_rows, route_rows = run_one_instance(
            fp, c=c, N=N, beta=beta, seed=seed, alpha=alpha, K=K, L_top=L_top
        )
        all_summary.extend(summary_rows)
        all_routes.extend(route_rows)

    df_summary = pd.DataFrame(all_summary)
    df_routes = pd.DataFrame(all_routes)

    # Sort for nicer CSV outputs
    order_sum = {"Heuristic + MC": 0, "Simheuristic": 1}
    df_summary["_ord"] = df_summary["Algorithm"].map(order_sum).fillna(99)
    df_summary = df_summary.sort_values(["Instance", "_ord"]).drop(columns=["_ord"])

    order_routes = {"Heuristic": 0, "Simheuristic": 1}
    df_routes["_ord"] = df_routes["Algorithm"].map(order_routes).fillna(99)
    df_routes = df_routes.sort_values(["Instance", "_ord", "Truck"]).drop(columns=["_ord"])

    # Save CSVs
    df_summary.to_csv(outdir / SUMMARY_CSV, index=False)
    df_routes.to_csv(outdir / ROUTES_CSV, index=False)

    # Save LaTeX tables
    write_summary_table_tex(
        df_summary,
        outpath=latex_dir / SUMMARY_TEX,
        caption=rf"Stochastic results summary (Monte Carlo with $N={N}$, $c={c}$, $\beta={beta}$). "
        r"Percent improvement is computed on $\mathbb{E}[\mathrm{Reward}]$ w.r.t.\ the heuristic baseline.",
        label="tab:stoc_summary",
    )
    write_route_paths_tables_tex(
        df_routes,
        outpath=latex_dir / ROUTES_TEX,
        caption_prefix="Route paths for instance",
        label_prefix="tab:routes_",
    )

    print(df_summary.to_string(index=False))
    print("\nSaved:")
    print(f" - {outdir / SUMMARY_CSV}")
    print(f" - {outdir / ROUTES_CSV}")
    print(f" - {latex_dir / SUMMARY_TEX}")
    print(f" - {latex_dir / ROUTES_TEX}")


if __name__ == "__main__":
    main()