from __future__ import annotations

from dataclasses import dataclass
from math import hypot
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

Node = int  # 0 is the depot, 1..n are silos

# Data structures
@dataclass(frozen=True)
class Instance:
    """
    Deterministic instance.

    coords[i]  = (x_i, y_i)  for i in {0..n}
    reward[i]  = u_i         (reward[0] should be 0.0)
    m          = number of trucks (max number of routes)
    Tmax       = time budget per route
    """
    coords: Dict[Node, Tuple[float, float]]
    reward: Dict[Node, float]
    m: int
    Tmax: float

    @property
    def n(self) -> int:
        return max(self.coords.keys())

@dataclass
class Route:
    """
    Route represented by an ordered list of visited silos [i1, i2, ..., ip].
    Depot 0 is implicit at start/end.
    """
    silos: List[Node]

    def first(self) -> Optional[Node]:
        return self.silos[0] if self.silos else None

    def last(self) -> Optional[Node]:
        return self.silos[-1] if self.silos else None

    def visited_set(self) -> Set[Node]:
        return set(self.silos)

    def as_sequence(self) -> List[Node]:
        """Explicit node sequence including depot 0 at both ends."""
        return [0] + self.silos + [0]

@dataclass
class MCResult:
    """Monte Carlo evaluation."""
    F_hat: float
    R_hat: float
    p_hat: List[float]


@dataclass
class SimheuristicResult:
    """Simheuristic result."""
    routes: List[Route]
    F_hat: float
    R_hat: float
    p_hat: List[float]
    q_best: int


# Loading data
def load_instance_from_txt(filepath: str | Path) -> Instance:
    """
    Format:
      n <int>
      m <int>
      tmax <float>
      x y reward
      x y reward
      ...

    Node 0 is depot (reward forced to 0).
    If header n mismatches actual number of node lines, we trust the lines.
    """
    fp = Path(filepath)
    lines = [ln.strip() for ln in fp.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if len(lines) < 4:
        raise ValueError("File too short / wrong format.")

    n_header = int(lines[0].split()[1])
    m = int(lines[1].split()[1])
    Tmax = float(lines[2].split()[1])

    node_lines = lines[3:]
    n_nodes = len(node_lines)
    if n_header != n_nodes:
        print(
            f"[load_instance_from_txt] Warning: header n={n_header}, found {n_nodes} node lines. "
            f"Using {n_nodes} nodes."
        )

    coords: Dict[Node, Tuple[float, float]] = {}
    reward: Dict[Node, float] = {}

    for idx, line in enumerate(node_lines):
        parts = line.replace(",", ".").split()
        if len(parts) != 3:
            raise ValueError(f"Bad node line #{idx}: {line!r}")
        x_s, y_s, u_s = parts
        coords[idx] = (float(x_s), float(y_s))
        reward[idx] = float(u_s)

    reward[0] = 0.0
    return Instance(coords=coords, reward=reward, m=m, Tmax=Tmax)


# Distances and metrics
def compute_distance_matrix(coords: Dict[Node, Tuple[float, float]]) -> List[List[float]]:
    """Euclidean distance matrix t[i][j]."""
    n = max(coords.keys())
    t = [[0.0] * (n + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        xi, yi = coords[i]
        for j in range(n + 1):
            xj, yj = coords[j]
            t[i][j] = hypot(xi - xj, yi - yj)
    return t


def route_length(route: Route, t: List[List[float]]) -> float:
    seq = route.as_sequence()
    return sum(t[seq[k]][seq[k + 1]] for k in range(len(seq) - 1))


def route_reward(route: Route, u: Dict[Node, float]) -> float:
    return sum(u[i] for i in route.silos)

def total_reward(routes: List[Route], u: Dict[Node, float]) -> float:
    return sum(route_reward(r, u) for r in routes)

def visited_silos(routes: List[Route]) -> Set[Node]:
    out: Set[Node] = set()
    for r in routes:
        out |= r.visited_set()
    return out

# Route formatting helpers
def pad_routes_to_m(routes: List[Route], m: int) -> List[Route]:
    """Pad with empty routes so that len(routes) == m (and truncate if longer)."""
    if len(routes) >= m:
        return routes[:m]
    return routes + [Route([]) for _ in range(m - len(routes))]


def routes_to_explicit(routes: List[Route], m: Optional[int] = None) -> List[List[int]]:
    """
    Convert Route objects to explicit lists [0, ..., 0].
    If m is provided, routes are padded/truncated to length m.
    """
    if m is not None:
        routes = pad_routes_to_m(routes, m)
    return [r.as_sequence() for r in routes]


def count_nonempty_routes(routes: List[Route]) -> int:
    return sum(1 for r in routes if r.silos)

# Printing
def print_heuristic_solution(
    routes: List[Route],
    inst: Instance,
    t: Optional[List[List[float]]] = None,
    title: str = "",
) -> None:
    """
    Deterministic/heuristic print:
      - title
      - then per-route deterministic details + deterministic totals
    """
    if t is None:
        t = compute_distance_matrix(inst.coords)

    if title:
        print(title)
    print(f"Number of routes: {len(routes)} (m={inst.m})   T_max={inst.Tmax:.2f}")

    visited: Set[Node] = set()
    sum_route_rewards = 0.0

    for idx, r in enumerate(routes, start=1):
        L = route_length(r, t)
        R = route_reward(r, inst.reward)
        sum_route_rewards += R

        vset = r.visited_set()
        visited |= vset

        seq = " -> ".join(map(str, r.as_sequence()))
        print(
            f"Route {idx}: {seq}   | "
            f"Reward={R:.2f}  Length={L:.2f}  Slack={inst.Tmax - L:.2f}  Visited={len(vset)}"
        )

    total_unique_reward = sum(inst.reward[i] for i in visited)
    print(f"Visited silos: {len(visited)}")
    print(f"Total reward: {total_unique_reward:.2f}")


def print_simheuristic_solution(
    routes: List[Route],
    inst: Instance,
    *,
    F_hat: float,
    R_hat: float,
    p_hat: Optional[List[float]] = None,
    t: Optional[List[List[float]]] = None,
    title: str = "",
) -> None:
    """
    Stochastic/simheuristic print:
      - title
      - then stochastic KPIs (E[Reward], Reliability)
      - then deterministic per-route details (+ optional per-route p_success)
      - then deterministic totals
    """
    if t is None:
        t = compute_distance_matrix(inst.coords)

    if title:
        print(title)
    print(f"E[Reward]: {F_hat:.2f}")
    print(f"Reliability: {R_hat:.4f}")
    print(f"Number of routes: {len(routes)} (m={inst.m})   T_max={inst.Tmax:.2f}")

    visited: Set[Node] = set()
    sum_route_rewards = 0.0

    for idx, r in enumerate(routes, start=1):
        L = route_length(r, t)
        R = route_reward(r, inst.reward)
        sum_route_rewards += R

        vset = r.visited_set()
        visited |= vset

        seq = " -> ".join(map(str, r.as_sequence()))
        extra = ""
        if p_hat is not None and (idx - 1) < len(p_hat):
            extra = f"  p_success={p_hat[idx - 1]:.4f}"

        print(
            f"Route {idx}: {seq}   | "
            f"Reward={R:.2f}  Length={L:.2f}  Slack={inst.Tmax - L:.2f}  Visited={len(vset)}{extra}"
        )

    total_unique_reward = sum(inst.reward[i] for i in visited)
    print(f"Visited silos: {len(visited)}")
    print(f"Total reward: {total_unique_reward:.2f}")