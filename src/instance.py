"""Instance data structure, JSON I/O and structural statistics.

An instance of the crew duty covering problem is stored as a JSON file with:
    name, source, params        -- provenance (generator parameters or source file)
    W, T                        -- full-time max working time / max total time (minutes)
    cost_ft, cost_ht            -- 1.0 and 0.6
    start, end                  -- duty start / end times (minutes), lists of length n
    arcs                        -- compatibility graph G, list of [i, j] (0-based):
                                   j can follow i for the same employee
    start_terminal, end_terminal, terminal_travel
                                -- optional spatial data (generated instances only)

Half-time limits are always W/2 and T/2 (assignment). The models only use
start, end, arcs, W, T: the spatial data are kept for documentation.
"""
from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

COST_FT = 1.0
COST_HT = 0.6


@dataclass
class Instance:
    name: str
    W: int
    T: int
    start: list[int]
    end: list[int]
    arcs: list[tuple[int, int]]
    source: str = "generator"
    params: dict = field(default_factory=dict)
    start_terminal: list[int] | None = None
    end_terminal: list[int] | None = None
    terminal_travel: list[list[int]] | None = None

    # ------------------------------------------------------------------ basics
    @property
    def n(self) -> int:
        return len(self.start)

    @property
    def duration(self) -> list[int]:
        return [e - s for s, e in zip(self.start, self.end)]

    def successors(self) -> list[list[int]]:
        succ: list[list[int]] = [[] for _ in range(self.n)]
        for i, j in self.arcs:
            succ[i].append(j)
        return succ

    # --------------------------------------------------------------------- I/O
    def to_json(self, path: str | Path) -> None:
        data = {
            "name": self.name,
            "source": self.source,
            "params": self.params,
            "W": self.W,
            "T": self.T,
            "cost_ft": COST_FT,
            "cost_ht": COST_HT,
            "n": self.n,
            "start": self.start,
            "end": self.end,
            "arcs": [list(a) for a in self.arcs],
            "start_terminal": self.start_terminal,
            "end_terminal": self.end_terminal,
            "terminal_travel": self.terminal_travel,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, separators=(",", ":"))

    @classmethod
    def from_json(cls, path: str | Path) -> "Instance":
        with open(path) as f:
            d = json.load(f)
        return cls(
            name=d["name"],
            W=d["W"],
            T=d["T"],
            start=d["start"],
            end=d["end"],
            arcs=[tuple(a) for a in d["arcs"]],
            source=d.get("source", "generator"),
            params=d.get("params", {}),
            start_terminal=d.get("start_terminal"),
            end_terminal=d.get("end_terminal"),
            terminal_travel=d.get("terminal_travel"),
        )

    # ------------------------------------------------------------- validation
    def check(self) -> None:
        """Raise ValueError if the instance violates the assumptions of Sec. 1.2."""
        n = self.n
        if len(self.end) != n:
            raise ValueError("start/end length mismatch")
        if self.W >= self.T:
            raise ValueError("assumption A3 violated: need W < T")
        if self.W % 2 or self.T % 2:
            raise ValueError("W and T must be even so that W/2, T/2 are integers")
        for i, d in enumerate(self.duration):
            if d <= 0:
                raise ValueError(f"duty {i} has non-positive duration")
            if d > self.W:
                raise ValueError(f"assumption A4 violated: duty {i} longer than W")
        for i, j in self.arcs:
            if not (0 <= i < n and 0 <= j < n) or i == j:
                raise ValueError(f"bad arc {(i, j)}")
            if self.end[i] > self.start[j]:
                raise ValueError(f"arc {(i, j)} violates time order")


# =============================================================================
# Structural statistics
# =============================================================================
def transitivity_violations(inst: Instance) -> int:
    """Number of pairs of arcs (i,j),(j,k) with (i,k) not in A."""
    succ = [set(s) for s in inst.successors()]
    return sum(1 for i in range(inst.n) for j in succ[i] for k in succ[j] if k not in succ[i])


def max_concurrency(inst: Instance) -> int:
    """Max number of duties running at the same instant (pairwise incompatible)."""
    events = [(s, 1) for s in inst.start] + [(e, -1) for e in inst.end]
    events.sort(key=lambda x: (x[0], x[1]))  # ends (-1) before starts at equal times
    cur = best = 0
    for _, delta in events:
        cur += delta
        best = max(best, cur)
    return best


def min_path_cover(inst: Instance) -> int:
    """rho(G): minimum number of vertex-disjoint paths covering the DAG.

    rho(G) = n - (maximum matching in the bipartite graph out-copy -> in-copy),
    computed with Hopcroft-Karp.
    """
    n = inst.n
    adj = inst.successors()
    INF = float("inf")
    match_l = [-1] * n  # left i -> right j
    match_r = [-1] * n
    dist = [0.0] * n

    def bfs() -> bool:
        q = deque()
        for u in range(n):
            if match_l[u] == -1:
                dist[u] = 0
                q.append(u)
            else:
                dist[u] = INF
        found = False
        while q:
            u = q.popleft()
            for v in adj[u]:
                w = match_r[v]
                if w == -1:
                    found = True
                elif dist[w] == INF:
                    dist[w] = dist[u] + 1
                    q.append(w)
        return found

    def dfs(root: int) -> bool:
        # iterative DFS along the BFS layering
        stack = [(root, iter(adj[root]))]
        path = []
        while stack:
            u, it = stack[-1]
            advanced = False
            for v in it:
                w = match_r[v]
                if w == -1:
                    path.append((u, v))
                    for a, b in path:
                        match_l[a] = b
                        match_r[b] = a
                    return True
                if dist[w] == dist[u] + 1:
                    path.append((u, v))
                    stack.append((w, iter(adj[w])))
                    advanced = True
                    break
            if not advanced:
                dist[u] = INF
                stack.pop()
                if path:
                    path.pop()
        return False

    matching = 0
    while bfs():
        for u in range(n):
            if match_l[u] == -1 and dfs(u):
                matching += 1
    return n - matching


def summary(inst: Instance) -> dict:
    """Statistics reported in Section 2 of the report."""
    n = inst.n
    d = inst.duration
    rho = min_path_cover(inst)
    work_lb = sum(d) / inst.W
    pairs = n * (n - 1) / 2 if n > 1 else 1.0
    # arcs that a full-time employee can actually use (pair fits in the spread limit)
    usable = sum(1 for i, j in inst.arcs if inst.end[j] - inst.start[i] <= inst.T)
    return {
        "name": inst.name,
        "source": inst.source,
        "n": n,
        "arcs": len(inst.arcs),
        "density": len(inst.arcs) / pairs,
        "eff_density": usable / pairs,
        "transitive": transitivity_violations(inst) == 0,
        "W": inst.W,
        "T": inst.T,
        "d_avg": sum(d) / n,
        "d_max": max(d),
        "ht_eligible": sum(1 for x in d if x <= inst.W // 2) / n,
        "horizon": max(inst.end) - min(inst.start),
        "max_conc": max_concurrency(inst),
        "rho": rho,
        # Lower bounds on the optimal cost (Sec. 1.5 / 2.4)
        "lb_path": COST_HT * rho,
        "lb_work": work_lb,
        "lb": max(COST_HT * rho, work_lb),
    }
