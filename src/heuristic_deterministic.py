from __future__ import annotations

from math import inf
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Set

from utilis import (
    Node,
    Instance,
    Route,
    compute_distance_matrix,
    route_length,
    route_reward,
    total_reward,
    visited_silos,
    pad_routes_to_m,
    count_nonempty_routes,
    assert_no_revisits
)

# Dummy solutions
def preprocessing_feasible_silos(inst: Instance, t: List[List[float]]) -> List[Node]:
    """Keep silos i such that (0,i,0) is feasible."""
    feasible: List[Node] = []
    for i in range(1, inst.n + 1):
        if t[0][i] + t[i][0] <= inst.Tmax:
            feasible.append(i)
    return feasible

def build_dummy_routes(feasible_silos: Iterable[Node]) -> List[Route]:
    return [Route([i]) for i in feasible_silos]

# Savings heuristics (Juan's paper-style)
def savings_score(i: Node, j: Node, t: List[List[float]], u: Dict[Node, float], alpha: float) -> float:
    """
    Depot-to-depot saving:
      s_ij = t_{i0} + t_{0j} - t_{ij}
      s'_ij = alpha*s_ij + (1-alpha)*(u_i + u_j)
    """
    s_ij = t[i][0] + t[0][j] - t[i][j]
    return alpha * s_ij + (1.0 - alpha) * (u[i] + u[j])

def merge_routes_if_possible(
    routes: Dict[int, Route],
    rid_end: Dict[Node, int],
    rid_start: Dict[Node, int],
    t: List[List[float]],
    Tmax: float,
    i: Node,
    j: Node,
) -> bool:
    """
    Attempt to merge route ending at i with route starting at j by connecting i->j:
      R_i = (0,...,i,0), R_j=(0,j,...,0)
    Merge admissible if:
      - i is last of its route and j is first of its route
      - routes are distinct
      - merged length <= Tmax

    Returns True if merged, False otherwise.
    """
    if i not in rid_end or j not in rid_start:
        return False
    r1_id = rid_end[i]
    r2_id = rid_start[j]
    if r1_id == r2_id:
        return False

    r1 = routes[r1_id]
    r2 = routes[r2_id]

    # L(R1 ⊕ R2) = L(R1) + L(R2) - t_{i0} - t_{0j} + t_{ij}
    L1 = route_length(r1, t)
    L2 = route_length(r2, t)
    merged_L = L1 + L2 - t[i][0] - t[0][j] + t[i][j]
    if merged_L > Tmax:
        return False

    # Merge: concatenate silos (depot implicit)
    new_route = Route(r1.silos + r2.silos)

    # Remove old routes
    del routes[r1_id]
    del routes[r2_id]

    f1, l1 = r1.first(), r1.last()
    f2, l2 = r2.first(), r2.last()

    if f1 is not None:
        rid_start.pop(f1, None)
    if l1 is not None:
        rid_end.pop(l1, None)
    if f2 is not None:
        rid_start.pop(f2, None)
    if l2 is not None:
        rid_end.pop(l2, None)

    new_id = max(routes.keys(), default=-1) + 1
    routes[new_id] = new_route
    rid_start[new_route.first()] = new_id  # type: ignore[arg-type]
    rid_end[new_route.last()] = new_id     # type: ignore[arg-type]
    return True


def savings_construction(inst: Instance, t: List[List[float]], alpha: float) -> List[Route]:
    feasible_silos = preprocessing_feasible_silos(inst, t)
    dummy = build_dummy_routes(feasible_silos)

    routes: Dict[int, Route] = {k: r for k, r in enumerate(dummy)}
    rid_start: Dict[Node, int] = {r.first(): rid for rid, r in routes.items() if r.first() is not None}  # type: ignore[misc]
    rid_end: Dict[Node, int] = {r.last(): rid for rid, r in routes.items() if r.last() is not None}      # type: ignore[misc]

    pairs: List[Tuple[float, Node, Node]] = []
    for i in feasible_silos:
        for j in feasible_silos:
            if i == j:
                continue
            pairs.append((savings_score(i, j, t, inst.reward, alpha), i, j))
    pairs.sort(reverse=True, key=lambda x: x[0])

    for _, i, j in pairs:
        merge_routes_if_possible(routes, rid_end, rid_start, t, inst.Tmax, i, j)

    final_routes = list(routes.values())
    final_routes.sort(key=lambda r: (-route_reward(r, inst.reward), route_length(r, t)))
    return final_routes[: inst.m]

# 2-opt intra route (first improvement)
def two_opt_first_improvement(route: Route, t: List[List[float]], Tmax: float) -> Route:
    """
    Intra-route 2-opt, first-improvement, preserving feasibility.
    Depot is implicit.
    """
    if len(route.silos) < 4:
        return route

    best = Route(route.silos[:])
    best_len = route_length(best, t)

    improved = True
    while improved:
        improved = False
        p = len(best.silos)
        for a in range(0, p - 1):
            for b in range(a + 1, p):
                cand_silos = best.silos[:]
                cand_silos[a : b + 1] = reversed(cand_silos[a : b + 1])
                cand = Route(cand_silos)
                cand_len = route_length(cand, t)
                if cand_len <= Tmax and cand_len + 1e-12 < best_len:
                    best = cand
                    best_len = cand_len
                    improved = True
                    break
            if improved:
                break
    return best

# Best insertion (second improvement)
def best_insertion_position(route: Route, j: Node, t: List[List[float]], Tmax: float) -> Optional[Tuple[int, float]]:
    """
    Return (pos, delta) where pos is the index in route.silos to insert j,
    and delta is the incremental length.
    """
    seq = route.as_sequence()  # [0] + silos + [0]
    base_len = route_length(route, t)

    best_delta = inf
    best_pos: Optional[int] = None

    # Insert between seq[k] and seq[k+1], k = 0..len(seq)-2.
    # Corresponds to insert into silos at pos=k (since seq[0]=0).
    for k in range(len(seq) - 1):
        a = seq[k]
        b = seq[k + 1]
        delta = t[a][j] + t[j][b] - t[a][b]
        if base_len + delta <= Tmax and delta < best_delta:
            best_delta = delta
            best_pos = k    # insert into silos at index k

    if best_pos is None:
        return None
    return best_pos, best_delta


def greedy_reinsertion(
    routes: List[Route],
    unvisited: Set[Node],
    t: List[List[float]],
    u: Dict[Node, float],
    Tmax: float,
    eps: float = 1e-9,
) -> Tuple[List[Route], Set[Node]]:
    """
    Greedy reinsertion: repeatedly insert the best admissible move (j, route, position)
    maximizing score = u_j / max(delta, eps), with priority for delta <= 0.
    """
    while True:
        best_move: Optional[Tuple[Node, int, int, float]] = None  # (j, route_idx, pos, delta)
        best_score = -inf

        for j in unvisited:
            uj = u[j]
            for r_idx, r in enumerate(routes):
                ins = best_insertion_position(r, j, t, Tmax)
                if ins is None:
                    continue
                pos, delta = ins
                den = max(delta, eps)
                score = inf if delta <= eps else (uj / den)
                # tie-break: higher score, then smaller delta, then higher uj
                if (score > best_score) or (
                    abs(score - best_score) <= 1e-12 and best_move is not None and delta < best_move[3]
                ):
                    best_score = score
                    best_move = (j, r_idx, pos, delta)

        if best_move is None:
            break

        j, r_idx, pos, _ = best_move
        # Apply insertion
        routes[r_idx].silos.insert(pos, j)
        unvisited.remove(j)

    return routes, unvisited

# Replacement swaps visited <-> unvisited (third improvement)
def replacement_swaps_visited_unvisited(
    routes: List[Route],
    unvisited: Set[Node],
    t: List[List[float]],
    u: Dict[Node, float],
    Tmax: float,
) -> Tuple[List[Route], Set[Node]]:
    """
    Iterate visited<->unvisited replacement moves until no improving move exists.
    Choose the best move maximizing gain = u_j - u_h, ties by smaller resulting route length.
    """
    # Build a fast visited list and also locate which route contains each visited silo
    def rebuild_index() -> Tuple[Set[Node], Dict[Node, int]]:
        vis: Set[Node] = set()
        where: Dict[Node, int] = {}
        for idx, r in enumerate(routes):
            for h in r.silos:
                vis.add(h)
                where[h] = idx
        return vis, where

    visited, where = rebuild_index()

    improved = True
    while improved:
        improved = False
        best_gain = 0.0
        best_choice: Optional[Tuple[Node, Node, int, int]] = None  # (h, j, route_idx, insert_pos)

        # Consider all (h in visited) and (j in unvisited) with u_j > u_h.
        for j in unvisited:
            uj = u[j]
            for h in visited:
                uh = u[h]
                if uj <= uh:
                    continue

                r_idx = where[h]
                r = routes[r_idx]
                # Build route without h
                r_minus = Route([x for x in r.silos if x != h])
                ins = best_insertion_position(r_minus, j, t, Tmax)
                if ins is None:
                    continue
                pos, _ = ins
                # If insertion exists, feasibility holds by construction in best_insertion_position

                gain = uj - uh
                if gain > best_gain + 1e-12:
                    best_gain = gain
                    best_choice = (h, j, r_idx, pos)
                elif abs(gain - best_gain) <= 1e-12 and best_choice is not None:
                    # tie-break by shorter resulting route length
                    h0, j0, r0, pos0 = best_choice
                    r0_minus = Route([x for x in routes[r0].silos if x != h0])
                    r0_minus.silos.insert(pos0, j0)
                    L0 = route_length(r0_minus, t)

                    r1_minus = Route([x for x in routes[r_idx].silos if x != h])
                    r1_minus.silos.insert(pos, j)
                    L1 = route_length(r1_minus, t)

                    if L1 < L0:
                        best_choice = (h, j, r_idx, pos)

        if best_choice is None:
            break

        # Apply best replacement
        h, j, r_idx, pos = best_choice
        # remove h
        routes[r_idx].silos = [x for x in routes[r_idx].silos if x != h]
        # insert j
        routes[r_idx].silos.insert(pos, j)

        # update sets
        unvisited.remove(j)
        unvisited.add(h)

        improved = True

        # rebuild index
        visited, where = rebuild_index()

    return routes, unvisited

# Heuristic solution
def solve_deterministic(
    inst: Instance,
    alpha: float = 0.5,
    grid_search_alpha: bool = True,
    alphas: Sequence[float] = tuple([i / 10 for i in range(1, 10)]),
) -> List[Route]:
    """
    Full deterministic pipeline:
      savings construction (+ optional alpha grid search),
      2-opt on each kept route,
      greedy reinsertion,
      replacement swaps visited<->unvisited.
    """
    t = compute_distance_matrix(inst.coords)

    # Choose alpha
    if grid_search_alpha:
        best_routes: List[Route] = []
        best_val = -inf
        best_len_sum = inf
        for a in alphas:
            cand = savings_construction(inst, t, a)
            val = total_reward(cand, inst.reward)
            len_sum = sum(route_length(r, t) for r in cand)
            if (val > best_val) or (abs(val - best_val) <= 1e-12 and len_sum < best_len_sum):
                best_val = val
                best_len_sum = len_sum
                best_routes = cand
        routes = best_routes
    else:
        routes = savings_construction(inst, t, alpha)

    assert_no_revisits(routes, inst)

    # 2-opt each route (once)
    routes = [two_opt_first_improvement(r, t, inst.Tmax) for r in routes]
    assert_no_revisits(routes, inst)

    # compute unvisited
    visited = visited_silos(routes)
    unvisited = set(range(1, inst.n + 1)) - visited
    # remove infeasible singletons
    unvisited = {i for i in unvisited if t[0][i] + t[i][0] <= inst.Tmax}

    # greedy reinsertion
    routes, unvisited = greedy_reinsertion(routes, unvisited, t, inst.reward, inst.Tmax)
    assert_no_revisits(routes, inst)

    # replacement swaps visited <-> unvisited
    routes, unvisited = replacement_swaps_visited_unvisited(routes, unvisited, t, inst.reward, inst.Tmax)
    assert_no_revisits(routes, inst)

    return routes

# Printing the solution
def print_heuristic_solution(
        routes: List[Route], 
        inst: Instance, 
        t: Optional[List[List[float]]] = None
    ) -> None:
    if t is None:
        t = compute_distance_matrix(inst.coords)

    assert_no_revisits(routes, inst)

    routes_m = pad_routes_to_m(routes, inst.m)

    vis = visited_silos(routes_m)
    rew = total_reward(routes_m, inst.reward)

    print("=== Deterministic Heuristic Solution ===")
    print(f"Silos visited: {len(vis)} / {inst.n}")
    print(f"Total reward: {rew:.3f}")
    print(f"Nonempty routes: {count_nonempty_routes(routes_m)} / {inst.m}")
    print("Routes:")

    for k, r in enumerate(routes_m, start=1):
        seq = r.as_sequence()
        seq_str = " -> ".join(map(str, seq))
        L = route_length(r, t)
        slack = inst.Tmax - L
        rr = route_reward(r, inst.reward)
        print(f"  Truck {k}: {seq_str} | length={L:.3f} | slack={slack:.3f} | reward={rr:.3f}")