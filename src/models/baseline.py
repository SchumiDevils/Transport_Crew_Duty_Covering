"""Baseline compact formulations of Section 1.3.

    build_af(inst, UF, UH)                 -> (AF) arc-flow, valid for any DAG
    build_cf(inst, UF, UH, spread="pair")  -> (CF)   conflict formulation
    build_cf(inst, UF, UH, spread="bigm")  -> (CF-M) big-M spread variant

Each builder returns (milp, decode), where decode(x) turns a solution vector
into the shift list used by the validator. Employees 0..UF-1 are full-time,
UF..UF+UH-1 half-time. No preprocessing, symmetry breaking or valid
inequalities are added here: those are the improvements of Section 5.

(CF)/(CF-M) are exact when G is transitive. On a general DAG every set they
accept is still a feasible shift (an acyclic tournament has a Hamiltonian
path), but some feasible shifts are excluded: they are then a *restriction*
and their optimum is an upper bound.
"""
from __future__ import annotations

from src.instance import COST_FT, COST_HT, Instance
from src.milp import INF, MILP


def _types(inst: Instance, UF: int, UH: int):
    """(type, W_k, T_k, c_k) for each employee of the pool."""
    return ([("FT", inst.W, inst.T, COST_FT)] * UF +
            [("HT", inst.W // 2, inst.T // 2, COST_HT)] * UH)


# =============================================================================
# (AF) arc-flow formulation
# =============================================================================
def build_af(inst: Instance, UF: int, UH: int):
    n, s, e, d = inst.n, inst.start, inst.end, inst.duration
    SRC, SNK = -1, -2
    # augmented arc set: sigma->i, i->omega, then the arcs of G
    arcs = [(SRC, i) for i in range(n)] + [(i, SNK) for i in range(n)] + list(inst.arcs)
    na = len(arcs)
    in_arcs = [[] for _ in range(n)]
    out_arcs = [[] for _ in range(n)]
    for a, (u, v) in enumerate(arcs):
        if v >= 0:
            in_arcs[v].append(a)
        if u >= 0:
            out_arcs[u].append(a)

    emp = _types(inst, UF, UH)
    m = MILP()
    base = []
    for typ, Wk, Tk, ck in emp:
        base.append(m.num_vars)
        for a, (u, v) in enumerate(arcs):
            m.add_var(cost=ck if u == SRC else 0.0)

    # (3) coverage
    for i in range(n):
        idx = [b + a for b in base for a in in_arcs[i]]
        m.add_row(idx, [1.0] * len(idx), 1.0, 1.0)
    for k, (typ, Wk, Tk, ck) in enumerate(emp):
        b = base[k]
        # (4) flow conservation
        for i in range(n):
            idx = [b + a for a in in_arcs[i]] + [b + a for a in out_arcs[i]]
            val = [1.0] * len(in_arcs[i]) + [-1.0] * len(out_arcs[i])
            m.add_row(idx, val, 0.0, 0.0)
        # (5) at most one path
        m.add_row([b + i for i in range(n)], [1.0] * n, -INF, 1.0)
        # (6) working time: sum_i d_i * inflow(i) - W_k * y_k <= 0
        idx, val = [], []
        for a, (u, v) in enumerate(arcs):
            if v >= 0:
                idx.append(b + a)
                val.append(d[v] - (Wk if u == SRC else 0.0))
        m.add_row(idx, val, -INF, 0.0)
        # (7) spread: sum e_i z_{i,omega} - sum (s_i + T_k) z_{sigma,i} <= 0
        idx = [b + i for i in range(n)] + [b + n + i for i in range(n)]
        val = [-(s[i] + Tk) for i in range(n)] + [float(e[i]) for i in range(n)]
        m.add_row(idx, val, -INF, 0.0)

    def decode(x):
        shifts = []
        for k, (typ, *_rest) in enumerate(emp):
            b = base[k]
            first = [i for i in range(n) if x[b + i] > 0.5]
            if not first:
                continue
            path, cur = [first[0]], first[0]
            while True:
                nxt = [arcs[a][1] for a in out_arcs[cur] if x[b + a] > 0.5]
                if not nxt or nxt[0] == SNK:
                    break
                cur = nxt[0]
                path.append(cur)
            shifts.append({"type": typ, "duties": path})
        return shifts

    return m, decode


# =============================================================================
# (CF) conflict formulation and (CF-M) big-M variant
# =============================================================================
def time_conflicts(inst: Instance) -> list[tuple[int, int]]:
    """Unordered pairs {i,j} with neither (i,j) nor (j,i) in A."""
    arcs = set(inst.arcs)
    return [(i, j) for i in range(inst.n) for j in range(i + 1, inst.n)
            if (i, j) not in arcs and (j, i) not in arcs]


def spread_conflicts(inst: Instance, T: int, exclude=frozenset()) -> list[tuple[int, int]]:
    s, e = inst.start, inst.end
    return [(i, j) for i in range(inst.n) for j in range(i + 1, inst.n)
            if (i, j) not in exclude and max(e[i], e[j]) - min(s[i], s[j]) > T]


def build_cf(inst: Instance, UF: int, UH: int, spread: str = "pair"):
    n, s, e, d = inst.n, inst.start, inst.end, inst.duration
    H = max(e)
    emp = _types(inst, UF, UH)
    tc = time_conflicts(inst)
    conf = {}
    if spread == "pair":
        tcs = frozenset(tc)
        for T in {inst.T, inst.T // 2}:
            conf[T] = tc + spread_conflicts(inst, T, tcs)
    elif spread == "bigm":
        conf = {inst.T: tc, inst.T // 2: tc}
    else:
        raise ValueError(spread)

    m = MILP()
    y = [m.add_var(cost=ck) for (_, _, _, ck) in emp]
    x = [[m.add_var() for _ in range(n)] for _ in emp]       # x[k][i]
    SE = []
    if spread == "bigm":
        SE = [(m.add_var(0.0, 0.0, H, False), m.add_var(0.0, 0.0, H, False)) for _ in emp]

    for i in range(n):                                       # (9) coverage
        m.add_row([x[k][i] for k in range(len(emp))], [1.0] * len(emp), 1.0, 1.0)
    for k, (typ, Wk, Tk, ck) in enumerate(emp):
        for i in range(n):                                   # (10) linking
            m.add_row([x[k][i], y[k]], [1.0, -1.0], -INF, 0.0)
        for i, j in conf[Tk]:                                # (11) conflicts
            m.add_row([x[k][i], x[k][j], y[k]], [1.0, 1.0, -1.0], -INF, 0.0)
        m.add_row([x[k][i] for i in range(n)] + [y[k]],      # (12) working time
                  [float(d[i]) for i in range(n)] + [-float(Wk)], -INF, 0.0)
        if spread == "bigm":                                 # (13)
            Sk, Ek = SE[k]
            for i in range(n):
                m.add_row([Sk, x[k][i]], [1.0, float(H)], -INF, float(s[i] + H))
                m.add_row([Ek, x[k][i]], [1.0, -float(e[i])], 0.0, INF)
            m.add_row([Ek, Sk, y[k]], [1.0, -1.0, -float(Tk)], -INF, 0.0)

    def decode(xv):
        shifts = []
        for k, (typ, *_rest) in enumerate(emp):
            duties = sorted((i for i in range(n) if xv[x[k][i]] > 0.5), key=lambda i: s[i])
            if duties:
                shifts.append({"type": typ, "duties": duties})
        return shifts

    return m, decode


MODELS = {
    "AF": build_af,
    "CF": lambda inst, UF, UH: build_cf(inst, UF, UH, "pair"),
    "CFM": lambda inst, UF, UH: build_cf(inst, UF, UH, "bigm"),
}
