from __future__ import annotations
from typing import Dict, List, Tuple, Optional, Set, cast
import pulp
from core import Node, Instance, load_instance_from_txt, compute_distance_matrix, routes_to_explicit, pad_routes_to_m
from heuristic_deterministic import solve_deterministic

# Extract routes from the binary decision variables
def extract_routes_from_x(
    x: Dict[Tuple[int, int, int], pulp.LpVariable],
    n: int,
    m: int,
    tol: float = 1e-6,
) -> List[List[int]]:
    """
    Reconstruct routes for each truck k from binary arc vars x[i,j,k].
    Assumes subtours are eliminated and solution is integral (or near-integral).
    Returns list of routes including depot, e.g. [0, i1, i2, 0].
    """
    routes: List[List[int]] = []
    nodes = list(range(0, n + 1))

    for k in range(1, m + 1):
        start = None
        for j in range(1, n + 1):
            v = x.get((0, j, k))
            if v is not None:
                val = v.value() 
                if val is not None and val > 1 - tol:
                    start = j
                    break
        if start is None:
            continue

        route = [0, start]
        cur = start
        visited: Set[int] = {start}

        while True:
            nxt = None
            for j in nodes:
                if j == cur:
                    continue
                v = x.get((cur, j, k))
                if v is not None:
                    val = v.value()
                    if val is not None and val > 1 - tol:
                        nxt = j
                        break
            if nxt is None:
                break
            route.append(nxt)
            if nxt == 0:
                break
            if nxt in visited:
                break
            visited.add(nxt)
            cur = nxt

        routes.append(route)

    return routes

# Calculate the total reward
def _routes_reward(routes: List[List[int]], reward: Dict[int, float]) -> float:
    return sum(reward[i] for r in routes for i in r if i != 0)

# Give heuristic solution as warm start for the MILP
def apply_mip_start_from_routes(
    routes: List[List[int]],
    x, z, y, yk, ordv,
    n: int,
    m: int,
) -> None:
    """
    routes: list of routes including depot 0, e.g. [[0,i1,...,0], [0,j1,...,0], ...]
    Assign routes to trucks k=1..m in given order. Remaining trucks unused.
    """
    # set all initial values to 0
    for var in x.values(): var.setInitialValue(0)
    for var in z.values(): var.setInitialValue(0)
    for var in y.values(): var.setInitialValue(0)
    for var in yk.values(): var.setInitialValue(0)
    for var in ordv.values(): var.setInitialValue(0)

    for k_idx, route in enumerate(routes[:m], start=1):
        if len(route) <= 2:
            continue

        yk[k_idx].setInitialValue(1)

        nodes_in_route = [i for i in route if i != 0]
        for pos, i in enumerate(nodes_in_route, start=1):
            if 1 <= i <= n:
                z[(i, k_idx)].setInitialValue(1)
                y[i].setInitialValue(1)
                ordv[(i, k_idx)].setInitialValue(pos)

        for a, b in zip(route[:-1], route[1:]):
            if a != b:
                # x is defined only for a!=b, so this is safe
                x[(a, b, k_idx)].setInitialValue(1)

# Set HiGHS as preferred solver
def _make_solver(msg: bool, time_limit_sec: int, mip_gap: float):
    """
    Prefer HiGHS if available, fallback to CBC.
    """
    # HiGHS
    if hasattr(pulp, "HiGHS"):
        try:
            return pulp.HiGHS(msg=msg, timeLimit=time_limit_sec, gapRel=mip_gap)
        except TypeError:
            try:
                return pulp.HiGHS(msg=msg)
            except TypeError:
                return pulp.HiGHS()

    # fallback CBC
    try:
        return pulp.PULP_CBC_CMD(msg=msg, timeLimit=time_limit_sec, gapRel=mip_gap, mip=True, warmStart=True)
    except TypeError:
        return pulp.PULP_CBC_CMD(msg=msg, timeLimit=time_limit_sec, gapRel=mip_gap, mip=True)

# Solve the MILP model
def solve_exact_ilp_mtz(
    inst: Instance,
    time_limit_sec: int = 300,
    mip_gap: float = 0.0,
    msg: bool = True,
    mip_start_routes: Optional[List[List[int]]] = None,
) -> Tuple[str, float, float, List[List[int]]]:
    """
    ILP with MTZ subtour elimination, multi-vehicle, depot-to-depot, prize collecting.

    Returns: (status_string, objective_value, max_frac_x, routes)
    where max_frac_x is the maximum fractional part among x variables.
    """
    n = inst.n
    t = compute_distance_matrix(inst.coords)
    u = inst.reward
    m = inst.m

    depot = 0
    customers = list(range(1, n + 1))
    nodes = list(range(0, n + 1))
    K = list(range(1, m + 1))

    prob = pulp.LpProblem("Deterministic_Orienteering_MTZ", pulp.LpMaximize)

    # x[i,j,k] binary arc usage
    x = pulp.LpVariable.dicts(
        "x",
        ((i, j, k) for i in nodes for j in nodes for k in K if i != j),
        lowBound=0,
        upBound=1,
        cat=pulp.LpBinary,
    )

    # z[i,k] = 1 if truck k visits customer i
    z = pulp.LpVariable.dicts(
        "z",
        ((i, k) for i in customers for k in K),
        lowBound=0,
        upBound=1,
        cat=pulp.LpBinary,
    )

    # y[i] = 1 if customer i visited by some truck
    y = pulp.LpVariable.dicts(
        "y",
        (i for i in customers),
        lowBound=0,
        upBound=1,
        cat=pulp.LpBinary,
    )

    # yk[k] = 1 if truck k is used
    yk = pulp.LpVariable.dicts(
        "yk",
        (k for k in K),
        lowBound=0,
        upBound=1,
        cat=pulp.LpBinary,
    )

    # MTZ order vars
    ordv = pulp.LpVariable.dicts(
        "ord",
        ((i, k) for i in customers for k in K),
        lowBound=0,
        upBound=n,
        cat=pulp.LpContinuous,
    )

    # Objective
    prob += pulp.lpSum(u[i] * y[i] for i in customers)

    # Link y and z: each customer visited at most once in fleet
    for i in customers:
        prob += pulp.lpSum(z[(i, k)] for k in K) == y[i], f"link_y_z_{i}"

    # Degree constraints for each truck
    for k in K:
        for i in customers:
            prob += pulp.lpSum(x[(j, i, k)] for j in nodes if j != i) == z[(i, k)], f"in_deg_{i}_{k}"
            prob += pulp.lpSum(x[(i, j, k)] for j in nodes if j != i) == z[(i, k)], f"out_deg_{i}_{k}"

    # Depot departure/return linked to yk
    for k in K:
        prob += pulp.lpSum(x[(depot, j, k)] for j in customers) == yk[k], f"dep_out_{k}"
        prob += pulp.lpSum(x[(i, depot, k)] for i in customers) == yk[k], f"dep_in_{k}"

    # Time budget linked to yk
    for k in K:
        prob += pulp.lpSum(t[i][j] * x[(i, j, k)] for i in nodes for j in nodes if i != j) <= inst.Tmax * yk[k], f"time_{k}"

    # MTZ bounds
    for k in K:
        for i in customers:
            prob += ordv[(i, k)] >= z[(i, k)], f"ord_lb_{i}_{k}"
            prob += ordv[(i, k)] <= n * z[(i, k)], f"ord_ub_{i}_{k}"

    # MTZ constraints among customers
    for k in K:
        for i in customers:
            for j in customers:
                if i == j:
                    continue
                prob += ordv[(i, k)] - ordv[(j, k)] + n * x[(i, j, k)] <= n - 1, f"mtz_{i}_{j}_{k}"

    # Warm start (heuristic)
    if mip_start_routes:
        apply_mip_start_from_routes(mip_start_routes, x, z, y, yk, ordv, n=n, m=m)

    solver = _make_solver(msg=msg, time_limit_sec=time_limit_sec, mip_gap=mip_gap)
    prob.solve(solver)

    status = pulp.LpStatus[prob.status]
    obj_val = cast(Optional[float], pulp.value(prob.objective))
    obj = 0.0 if obj_val is None else float(obj_val)

    # Integrality diagnostic on x
    x_vals = [v.value() for v in x.values() if v.value() is not None]
    if x_vals:
        max_frac = max(abs(val - round(val)) for val in x_vals)
    else:
        max_frac = float("inf")

    # Extract only if we really have an integer incumbent loaded
    routes: List[List[int]] = []
    if max_frac < 1e-6:
        routes = extract_routes_from_x(x, n=n, m=m, tol=1e-6)
        return status, obj, max_frac, routes

    # Fallback: if solver didn't load any incumbent but we provided a warm start,
    # return the warm start so comparison is still meaningful.
    if mip_start_routes:
        fallback_routes = mip_start_routes[:m]
        fallback_obj = _routes_reward(fallback_routes, u)
        return f"{status} (no incumbent loaded; using warm start)", fallback_obj, 0.0, fallback_routes

    return status, obj, max_frac, []