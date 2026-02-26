from __future__ import annotations

from dataclasses import dataclass
from math import inf, log, sqrt
from typing import Dict, List, Optional, Set, Tuple
import random

from core import (
    Instance, Route, 
    Node,
    MCResult, SimheuristicResult, 
    compute_distance_matrix, 
    route_length, route_reward, 
    pad_routes_to_m
)
from heuristic_deterministic import (
    savings_construction,
    two_opt_first_improvement,
    best_insertion_position,
)

# Lognormal calibration (mean = t_ij, var = c * t_ij)
def _lognormal_params_from_mean_var(mean: float, var: float) -> Tuple[float, float]:
    """
    If T ~ LogNormal(mu, sigma^2) with underlying Normal(mu, sigma^2),
    return (mu, sigma) given mean=E[T], var=Var[T].

    sigma^2 = ln(1 + var/mean^2)
    mu      = ln(mean) - 0.5*sigma^2
    """
    if mean <= 0.0:
        return 0.0, 0.0
    if var < 0.0:
        raise ValueError("Variance must be nonnegative.")
    sigma2 = log(1.0 + var / (mean * mean))
    sigma = sqrt(sigma2)
    mu = log(mean) - 0.5 * sigma2
    return mu, sigma

def build_lognormal_mu_sigma(t: List[List[float]], c: float) -> Tuple[List[List[float]], List[List[float]]]:
    if c < 0.0:
        raise ValueError("c must be >= 0")
    n = len(t) - 1
    mu = [[0.0] * (n + 1) for _ in range(n + 1)]
    sig = [[0.0] * (n + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        for j in range(n + 1):
            if i == j:
                mu[i][j] = 0.0
                sig[i][j] = 0.0
                continue
            mean = t[i][j]
            var = c * mean
            mu[i][j], sig[i][j] = _lognormal_params_from_mean_var(mean, var)
    return mu, sig

# Beginning of the simheuristic: randomized Top-L moves (reinsertion + replacement)

# Randomized greedy reinsertion
def randomized_reinsertion_topL(
    routes: List[Route],
    unvisited: Set[Node],
    t: List[List[float]],
    u: Dict[Node, float],
    Tmax: float,
    L_top: int,
    rng: random.Random,
    eps: float = 1e-9,
) -> Tuple[List[Route], Set[Node]]:
    """
    Randomized variant of greedy reinsertion:
      - Build all admissible insertion moves (j into route r at best position)
      - Rank by score = u_j / max(Delta, eps), with Delta<=0 treated as dominating
      - Pick uniformly at random among Top-L_top moves
      - Apply, repeat until no admissible move exists
    """
    if L_top <= 0:
        raise ValueError("L_top must be >= 1")

    while True:
        candidates: List[Tuple[float, float, Node, int, int]] = []
        # (score, delta, j, route_idx, insert_pos)

        for j in unvisited:
            uj = u[j]
            for r_idx, r in enumerate(routes):
                ins = best_insertion_position(r, j, t, Tmax)
                if ins is None:
                    continue
                pos, delta = ins
                score = inf if delta <= eps else (uj / max(delta, eps))
                candidates.append((score, delta, j, r_idx, pos))

        if not candidates:
            break

        candidates.sort(key=lambda z: (-z[0], z[1], -u[z[2]]))
        top = candidates[: min(L_top, len(candidates))]
        _, _, j_star, r_star, pos_star = rng.choice(top)

        routes[r_star].silos.insert(pos_star, j_star)
        unvisited.remove(j_star)

    return routes, unvisited

#Randomized replacement swaps
def randomized_replacement_topL(
    routes: List[Route],
    unvisited: Set[Node],
    t: List[List[float]],
    u: Dict[Node, float],
    Tmax: float,
    L_top: int,
    rng: random.Random,
) -> Tuple[List[Route], Set[Node]]:
    """
    Randomized variant of replacement swaps (visited <-> unvisited):
      - Consider improving feasible replacements (u_j > u_h)
      - Rank by gain desc, tie by shorter resulting route length
      - Pick uniformly at random among Top-L_top
      - Apply and repeat until no improving move exists
    """
    if L_top <= 0:
        raise ValueError("L_top must be >= 1")

    def rebuild_index() -> Tuple[Set[Node], Dict[Node, int]]:
        visited: Set[Node] = set()
        where: Dict[Node, int] = {}
        for ridx, r in enumerate(routes):
            for h in r.silos:
                visited.add(h)
                where[h] = ridx
        return visited, where

    visited, where = rebuild_index()

    while True:
        candidates: List[Tuple[float, float, Node, Node, int, int]] = []
        # (gain, new_len, h, j, route_idx, insert_pos)

        for j in unvisited:
            uj = u[j]
            for h in visited:
                uh = u[h]
                if uj <= uh:
                    continue
                r_idx = where[h]
                r = routes[r_idx]
                r_minus = Route([x for x in r.silos if x != h])

                ins = best_insertion_position(r_minus, j, t, Tmax)
                if ins is None:
                    continue
                pos, _delta = ins

                cand = Route(r_minus.silos[:])
                cand.silos.insert(pos, j)
                new_len = route_length(cand, t)

                gain = uj - uh
                candidates.append((gain, new_len, h, j, r_idx, pos))

        if not candidates:
            break

        candidates.sort(key=lambda z: (-z[0], z[1]))
        top = candidates[: min(L_top, len(candidates))]
        _, _, h_star, j_star, r_star, pos_star = rng.choice(top)

        # apply
        routes[r_star].silos = [x for x in routes[r_star].silos if x != h_star]
        routes[r_star].silos.insert(pos_star, j_star)
        unvisited.remove(j_star)
        unvisited.add(h_star)

        visited, where = rebuild_index()

    return routes, unvisited

# Candidate generation
def build_candidate_solution(
    inst: Instance,
    alpha: float,
    L_top: int,
    rng: random.Random,
    randomize_replacement: bool = True,
) -> List[Route]:
    """
    Randomized variant of the deterministic pipeline, used in the simheuristic:
      - savings construction
      - 2-opt
      - Top-L randomized reinsertion
      - Top-L randomized replacement swaps
    Returns up to m routes (as in deterministic), then we pad to exactly m if needed.
    """
    t = compute_distance_matrix(inst.coords)

    # Construction
    routes = savings_construction(inst, t, alpha)

    # 2-opt
    routes = [two_opt_first_improvement(r, t, inst.Tmax) for r in routes]

    # unvisited
    visited = set().union(*(r.visited_set() for r in routes)) if routes else set()
    unvisited = set(range(1, inst.n + 1)) - visited
    unvisited = {i for i in unvisited if t[0][i] + t[i][0] <= inst.Tmax}

    # randomized improvements
    routes, unvisited = randomized_reinsertion_topL(routes, unvisited, t, inst.reward, inst.Tmax, L_top, rng)
    if randomize_replacement:
        routes, unvisited = randomized_replacement_topL(routes, unvisited, t, inst.reward, inst.Tmax, L_top, rng)

    # Ensure exactly m routes (paper-style: a solution is a set of m routes)
    if len(routes) < inst.m:
        routes = routes + [Route([]) for _ in range(inst.m - len(routes))]
    elif len(routes) > inst.m:
        routes = routes[: inst.m]

    return pad_routes_to_m(routes, inst.m)

# Monte Carlo evaluation of the candidate
def evaluate_solution_mc(
    inst: Instance,
    routes: List[Route],
    mu: List[List[float]],
    sigma: List[List[float]],
    N: int,
    rng: random.Random,
) -> MCResult:
    """
    All-or-nothing reward evaluation.
    Reliability: average success rate across ALL m routes.
    Empty routes (unused trucks) have S=0, so success is always 1 and reward 0.
    """
    if N <= 0:
        raise ValueError("N must be >= 1")

    # Ensure exactly m routes
    if len(routes) < inst.m:
        routes = routes + [Route([]) for _ in range(inst.m - len(routes))]
    elif len(routes) > inst.m:
        routes = routes[: inst.m]

    # Precompute route arcs and deterministic rewards
    arc_lists: List[List[Tuple[int, int]]] = []
    det_rewards: List[float] = []
    for r in routes:
        seq = r.as_sequence()
        arc_lists.append([(seq[i], seq[i + 1]) for i in range(len(seq) - 1)])
        det_rewards.append(route_reward(r, inst.reward))

    successes = [0] * len(routes)
    total_reward_acc = 0.0

    for _s in range(N):
        scen_reward = 0.0
        for k, arcs in enumerate(arc_lists):
            S = 0.0
            for (i, j) in arcs:
                if i == j:
                    continue
                Tij = rng.lognormvariate(mu[i][j], sigma[i][j])
                S += Tij
            if S <= inst.Tmax:
                scen_reward += det_rewards[k]
                successes[k] += 1
        total_reward_acc += scen_reward

    F_hat = total_reward_acc / N
    p_hat = [s / N for s in successes]
    R_hat = sum(p_hat) / len(p_hat) if p_hat else 1.0  # average over m routes

    return MCResult(F_hat=F_hat, R_hat=R_hat, p_hat=p_hat)

def pre_sample_scenarios(
    inst: Instance,
    mu: List[List[float]],
    sigma: List[List[float]],
    N: int,
    rng: random.Random,
) -> List[List[List[float]]]:
    """
    Pre-sample travel times for CRN evaluation.
    Returns TT[s][i][j] = sampled travel time for scenario s and arc (i,j).
    Shape: N x (n+1) x (n+1)
    """
    if N <= 0:
        raise ValueError("N must be >= 1")

    n = inst.n
    TT = [[[0.0] * (n + 1) for _ in range(n + 1)] for _ in range(N)]
    for s in range(N):
        for i in range(n + 1):
            for j in range(n + 1):
                if i == j:
                    TT[s][i][j] = 0.0
                else:
                    TT[s][i][j] = rng.lognormvariate(mu[i][j], sigma[i][j])
    return TT

def evaluate_solution_mc_presampled(
    inst: Instance,
    routes: List[Route],
    TT: List[List[List[float]]],
) -> MCResult:
    """
    Monte Carlo evaluation using pre-sampled travel times (CRN).
    Uses the same all-or-nothing reward logic as evaluate_solution_mc().
    """
    N = len(TT)
    if N <= 0:
        raise ValueError("TT must contain at least 1 scenario")

    # Ensure exactly m routes
    routes = pad_routes_to_m(routes, inst.m)

    # Precompute route arcs and deterministic rewards
    arc_lists: List[List[Tuple[int, int]]] = []
    det_rewards: List[float] = []
    for r in routes:
        seq = r.as_sequence()
        arc_lists.append([(seq[i], seq[i + 1]) for i in range(len(seq) - 1)])
        det_rewards.append(route_reward(r, inst.reward))

    successes = [0] * len(routes)
    total_reward_acc = 0.0

    for s in range(N):
        scen_reward = 0.0
        for k, arcs in enumerate(arc_lists):
            S = 0.0
            for (i, j) in arcs:
                S += TT[s][i][j]
            if S <= inst.Tmax:
                scen_reward += det_rewards[k]
                successes[k] += 1
        total_reward_acc += scen_reward

    F_hat = total_reward_acc / N
    p_hat = [x / N for x in successes]
    R_hat = sum(p_hat) / len(p_hat) if p_hat else 1.0  # average over m trucks

    return MCResult(F_hat=F_hat, R_hat=R_hat, p_hat=p_hat)

# Full simheuristic
def _solve_stochastic_simheuristic_fixed_alpha(
    inst: Instance,
    alpha: float,
    c: float,
    K: int,
    L_top: int,
    N: int,
    beta: float,
    seed: int = 0,
    randomize_replacement: bool = True,
    TT: Optional[List[List[List[float]]]] = None,
) -> SimheuristicResult:
    """
    Solve simheuristic, but with a fixed alpha.
    """
    if K <= 0:
        raise ValueError("K must be >= 1")
    if L_top <= 0:
        raise ValueError("L_top must be >= 1")
    if N <= 0:
        raise ValueError("N must be >= 1")
    if not (0.0 < beta < 1.0):
        raise ValueError("beta must be in (0,1)")
    if c < 0.0:
        raise ValueError("c must be >= 0")

    rng = random.Random(seed)

    t = compute_distance_matrix(inst.coords)
    mu, sigma = build_lognormal_mu_sigma(t, c)

    best_q: Optional[int] = None
    best_F = float("-inf")

    best_rel_q: Optional[int] = None

    candidates: List[List[Route]] = []
    evals: List[MCResult] = []

    for q in range(1, K + 1):
        cand_routes = build_candidate_solution(
            inst, alpha, L_top, rng, randomize_replacement=randomize_replacement
        )
        candidates.append(cand_routes)

        if TT is None:
            res = evaluate_solution_mc(inst, cand_routes, mu, sigma, N, rng)
        else:
            res = evaluate_solution_mc_presampled(inst, cand_routes, TT)
        evals.append(res)

        # most reliable (fallback)
        if best_rel_q is None:
            best_rel_q = q
        else:
            prev_rel = evals[best_rel_q - 1]
            better_R = res.R_hat > prev_rel.R_hat + 1e-12
            tie_R = abs(res.R_hat - prev_rel.R_hat) <= 1e-12
            better_F_in_tie = res.F_hat > prev_rel.F_hat + 1e-12
            if better_R or (tie_R and better_F_in_tie):
                best_rel_q = q

        # best expected reward among those meeting reliability threshold
        if res.R_hat >= beta:
            if best_q is None:
                best_F = res.F_hat
                best_q = q
            else:
                prev = evals[best_q - 1]
                better_F = res.F_hat > prev.F_hat + 1e-12
                tie_F = abs(res.F_hat - prev.F_hat) <= 1e-12
                better_R_in_tie = res.R_hat > prev.R_hat + 1e-12
                if better_F or (tie_F and better_R_in_tie):
                    best_F = res.F_hat
                    best_q = q

    if best_q is None:
        assert best_rel_q is not None
        best_q = best_rel_q

    idx = best_q - 1
    return SimheuristicResult(
        routes=candidates[idx],
        F_hat=evals[idx].F_hat,
        R_hat=evals[idx].R_hat,
        p_hat=evals[idx].p_hat,
        q_best=best_q,
    )

def solve_stochastic_simheuristic(
    inst: Instance,
    alpha: float,
    c: float,
    K: int,
    L_top: int,
    N: int,
    beta: float,
    seed: int = 0,
    randomize_replacement: bool = True,
    TT: Optional[List[List[List[float]]]] = None,  # CRN scenarios
    grid_search_alpha: bool = False,
    alphas: Optional[List[float]] = None,
) -> SimheuristicResult:
    """
    If grid_search_alpha=False:
      run simheuristic with the given alpha (as before).

    If grid_search_alpha=True:
      try multiple alpha values and return the best solution using the same rule:
        - maximize F_hat among solutions with R_hat >= beta
        - if none satisfies beta, maximize R_hat (tie-break by F_hat)
    """
    if not grid_search_alpha:
        return _solve_stochastic_simheuristic_fixed_alpha(
            inst=inst,
            alpha=alpha,
            c=c,
            K=K,
            L_top=L_top,
            N=N,
            beta=beta,
            seed=seed,
            randomize_replacement=randomize_replacement,
            TT=TT,
        )

    if alphas is None:
        alphas = [i / 10 for i in range(1, 10)]  # 0.1..0.9 default grid
    if not alphas:
        raise ValueError("alphas must be non-empty when grid_search_alpha=True")

    # run once per alpha
    results: List[Tuple[float, SimheuristicResult]] = []
    for i, a in enumerate(alphas):
        res = _solve_stochastic_simheuristic_fixed_alpha(
            inst=inst,
            alpha=float(a),
            c=c,
            K=K,
            L_top=L_top,
            N=N,
            beta=beta,
            seed=seed + 10007 * i,  # diversify construction randomness across alphas
            randomize_replacement=randomize_replacement,
            TT=TT,  # if you pass CRN, it stays common across alphas too
        )
        results.append((float(a), res))

    # pick best with the same policy rule
    feasible = [r for (_a, r) in results if r.R_hat >= beta]
    if feasible:
        return max(feasible, key=lambda r: (r.F_hat, r.R_hat))
    return max((r for (_a, r) in results), key=lambda r: (r.R_hat, r.F_hat))