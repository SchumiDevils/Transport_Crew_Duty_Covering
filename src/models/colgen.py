"""Column generation for the set partitioning model (SP) of Section 1.4.

    solve_sp(inst, time_limit) -> dict of statistics + best solution

1. Root LP by column generation. The restricted master problem (RMP) keeps
   the equality rows sum_p a_ip lambda_p = 1, so (SP) is exact on any DAG
   (covering rows would need transitivity). It starts from the singleton
   shifts and the greedy solution. The pricing problem, one per type, is the
   RCSPP of Section 1.4, solved by labelling in topological order. A heuristic
   pass keeps few labels per node; the exact pass (no cap) is used only when
   the heuristic finds no negative column, so the last iteration proves
   LP optimality: z_LP is a valid lower bound, and so is ceil(5 z_LP)/5
   because every cost is a multiple of 0.2.
2. Price-and-branch: the RMP with its columns is solved as a MILP (upper bound).
3. Reduced-cost enumeration (exactness without branch-and-price). With the
   optimal duals pi, every partition S costs z(S) = z_LP + sum_{p in S} rc_p
   with rc_p >= 0, so each column of a solution of cost <= UB - 0.2 has
   rc_p <= UB - 0.2 - z_LP. All such columns are enumerated by a DFS with a
   completion bound; the MILP over them either finds a better solution, which
   is then optimal, or proves that UB is optimal.
"""
from __future__ import annotations

import bisect
import heapq
import math
import time

import numpy as np

from src.heuristics.greedy import greedy
from src.instance import COST_FT, COST_HT, Instance

EPS = 1e-7
STEP = 0.2                      # all costs are multiples of 0.2


def ceil_step(z: float) -> float:
    return math.ceil(z / STEP - 1e-6) * STEP


class Shifts:
    """Feasibility and cost of shifts (paths of G)."""

    def __init__(self, inst: Instance):
        self.inst = inst
        self.n = inst.n
        self.s, self.e, self.d = inst.start, inst.end, inst.duration
        self.order = sorted(range(inst.n), key=lambda i: (self.s[i], self.e[i]))
        self.succ = [[] for _ in range(inst.n)]
        for i, j in inst.arcs:
            self.succ[i].append(j)
        self.types = [("FT", inst.W, inst.T, COST_FT), ("HT", inst.W // 2, inst.T // 2, COST_HT)]

    def cost(self, path) -> float:
        work = sum(self.d[i] for i in path)
        spread = self.e[path[-1]] - self.s[path[0]]
        return COST_HT if work <= self.inst.W // 2 and spread <= self.inst.T // 2 else COST_FT

    def typ(self, path) -> str:
        return "HT" if self.cost(path) == COST_HT else "FT"


# =============================================================================
# pricing: labelling for the RCSPP
# =============================================================================
def price(S: Shifts, pi: np.ndarray, W: int, T: int, c: float, cap: int | None, max_cols: int):
    """Shifts of type (W, T, c) with negative reduced cost, most negative first.

    Label = (cost so far = -sum pi, work, first start s_f, path). The labels
    of a node are grouped by s_f (the spread of every extension depends only
    on s_f and the end node). Exact mode keeps, in each group, the Pareto front
    in (work, cost); a label is discarded only if another label with the same
    s_f has no larger work and cost, so no negative column is lost. Heuristic
    mode (cap) keeps the cheapest label per group and extends only the `cap`
    cheapest groups. In both modes a label is dropped when even the cheapest
    resource-free completion g cannot make its reduced cost negative.
    Returns (columns, min reduced cost found).
    """
    s, e, d = S.s, S.e, S.d
    g = completion(S, pi)
    labels = [{} for _ in range(S.n)]       # node -> {s_f: front}
    best = []
    lim = -c - EPS        # a label needs cost + g(node) < -c to lead to a negative column
    ins = _ins_exact if cap is None else _ins_heur
    for j in S.order:
        groups = labels[j]
        if d[j] <= W and -pi[j] + g[j] < lim:
            ins(groups, s[j], d[j], -pi[j], (j,))
        if cap is None:
            L = [(cost, w, sf, path) for sf, (ws, cs, ps) in groups.items()
                 for w, cost, path in zip(ws, cs, ps)]
        else:
            L = heapq.nsmallest(cap, ((cost, w, sf, path) for sf, (w, cost, path) in groups.items()))
        for lab in L:
            rc = c + lab[0]
            if rc < -EPS:
                best.append((rc, lab[3]))
        for k in S.succ[j]:
            ek, dk, pk, gk = e[k], d[k], pi[k], g[k]
            Gk = labels[k]
            for cost, w, sf, path in L:
                if w + dk <= W and ek - sf <= T and cost - pk + gk < lim:
                    ins(Gk, sf, w + dk, cost - pk, path + (k,))
        labels[j] = None                      # free memory
    best.sort()
    min_rc = best[0][0] if best else 0.0
    out, seen = [], set()
    for rc, p in best:
        if p not in seen:
            seen.add(p)
            out.append(p)
            if len(out) >= max_cols:
                break
    return out, min_rc


def completion(S: Shifts, pi: np.ndarray) -> list[float]:
    """g[j] = min over paths leaving j (j excluded) of -sum pi; resources
    ignored, so it is a lower bound on the cost of any extension of a label."""
    g = [0.0] * S.n
    for j in reversed(S.order):
        g[j] = min([0.0] + [g[k] - pi[k] for k in S.succ[j]])
    return g


def _ins_exact(groups: dict, sf, w, c, path) -> None:
    """Insert into the Pareto front of group sf (works ascending, costs descending)."""
    f = groups.get(sf)
    if f is None:
        groups[sf] = ([w], [c], [path])
        return
    ws, cs, ps = f
    i = bisect.bisect_right(ws, w)
    if i > 0 and cs[i - 1] <= c + EPS:          # dominated by a label with work <= w
        return
    if i > 0 and ws[i - 1] == w:                # same work, higher cost: replaced
        i -= 1
        del ws[i], cs[i], ps[i]
    k = i
    while k < len(ws) and cs[k] >= c - EPS:     # more work and no cheaper: dominated
        k += 1
    del ws[i:k], cs[i:k], ps[i:k]
    ws.insert(i, w)
    cs.insert(i, c)
    ps.insert(i, path)


def _ins_heur(groups: dict, sf, w, c, path) -> None:
    o = groups.get(sf)
    if o is None or (c, w) < (o[1], o[0]):
        groups[sf] = (w, c, path)


# =============================================================================
# reduced-cost enumeration
# =============================================================================
def enumerate_columns(S: Shifts, pi: np.ndarray, delta: float, max_cols: int, deadline: float):
    """All shifts p with c_p - sum_{i in p} pi_i <= delta (None if a limit is hit)."""
    s, e, d, W, T = S.s, S.e, S.d, S.inst.W, S.inst.T
    Wh, Th = W // 2, T // 2
    g = completion(S, pi)
    cols, pops = [], 0
    stack = [(j,) for j in S.order if d[j] <= W]
    meta = {p: (-pi[p[0]], d[p[0]]) for p in stack}
    while stack:
        p = stack.pop()
        cost, w = meta.pop(p)
        j = p[-1]
        ht = w <= Wh and e[j] - s[p[0]] <= Th
        c = COST_HT if ht else COST_FT
        if c + cost <= delta + EPS:
            cols.append(p)
            if len(cols) > max_cols:
                return None
        # completion: cheapest possible type of an extension is HT only if still HT-feasible
        if (COST_HT if ht else COST_FT) + cost + g[j] > delta + EPS:
            continue
        pops += 1
        if pops % 2000 == 0 and time.perf_counter() > deadline:
            return None
        for k in S.succ[j]:
            if w + d[k] <= W and e[k] - s[p[0]] <= T:
                q = p + (k,)
                meta[q] = (cost - pi[k], w + d[k])
                stack.append(q)
    return cols


# =============================================================================
# MILP over a set of columns
# =============================================================================
def solve_columns(S: Shifts, cols: list, deadline: float, start: list | None = None,
                  cutoff: float | None = None):
    """(SP) restricted to `cols` as a MILP, stopping at `deadline`.

    start  = a known feasible 0/1 vector (used through the improving cut);
    cutoff = only look for solutions of cost <= cutoff - 0.2.
    Returns None if no time is left after building the model.
    """
    from src.milp import INF, MILP, solve, solve_from
    m = MILP()
    rows = [[] for _ in range(S.n)]
    for c, p in enumerate(cols):
        m.add_var(cost=S.cost(p))
        for i in p:
            rows[i].append(c)
    for i in range(S.n):
        m.add_row(rows[i], [1.0] * len(rows[i]), 1.0, 1.0)
    tl = deadline - time.perf_counter()        # measured after building the model
    if tl < 1.0:
        return None
    if start is not None:
        return solve_from(m, start, STEP, "highs", time_limit=tl)
    if cutoff is None:
        return solve(m, "highs", time_limit=tl)
    # improving cut c.x <= cutoff - 0.2 (no feasible start inside these columns)
    m.add_row(list(range(m.num_vars)), list(m.cost), -INF, cutoff - STEP + 1e-6)
    return solve(m, "highs", time_limit=tl)


# =============================================================================
# main procedure
# =============================================================================
def solve_sp(inst: Instance, time_limit: float = 300, enum_max: int = 250_000,
             heur_cap: int = 30, cols_per_iter: int = 300, pb_share: float = 0.4) -> dict:
    import highspy

    t0 = time.perf_counter()
    deadline = t0 + time_limit
    S = Shifts(inst)
    n = S.n
    g = greedy(inst)                               # general greedy: exact on any DAG
    ub, best = g["objective"], [tuple(sh["duties"]) for sh in g["shifts"]]

    # ---------------------------------------------------------------- RMP
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.setOptionValue("threads", 1)
    h.addRows(n, np.ones(n), np.ones(n), 0, np.array([0], dtype=np.int32),
              np.array([], dtype=np.int32), np.array([], dtype=float))
    cols, colset = [], set()

    def add(paths):
        new = [p for p in paths if p not in colset]
        if not new:
            return 0
        starts, idx = [], []
        for p in new:
            starts.append(len(idx))
            idx.extend(p)
            colset.add(p)
            cols.append(p)
        k = len(new)
        h.addCols(k, np.array([S.cost(p) for p in new]), np.zeros(k), np.full(k, np.inf), len(idx),
                  np.array(starts, dtype=np.int32), np.array(idx, dtype=np.int32), np.ones(len(idx)))
        return k

    add([(i,) for i in range(n)] + best)
    iters, lagr, z_rmp, exact_done = 0, -math.inf, math.nan, False
    pi = np.zeros(n)
    status = "optimal"
    while True:
        h.run()
        z_rmp = h.getInfo().objective_function_value
        pi = np.asarray(h.getSolution().row_dual)
        iters += 1
        if time.perf_counter() > deadline:
            status = "time_limit"
            break
        new = []
        for exact in (False, True):
            min_rc = {}
            for typ, W, T, c in S.types:
                p, min_rc[typ] = price(S, pi, W, T, c, None if exact else heur_cap, cols_per_iter)
                new += p
            if new:
                break
        if exact:   # Lagrangian bound (valid since the exact minimum reduced costs are known)
            lagr = max(lagr, z_rmp + sum(ub / c * min(0.0, min_rc[typ]) for typ, _, _, c in S.types))
        if not new:
            exact_done = True
            break
        add(new)
    cg_sec = time.perf_counter() - t0
    z_lp = z_rmp if exact_done else math.nan
    lb = ceil_step(z_lp) if exact_done else (ceil_step(lagr) if math.isfinite(lagr) else 0.0)
    ncols_cg = len(cols)

    # ------------------------------------------------------ price-and-branch
    pb_obj = math.nan

    def price_and_branch(share: float) -> None:
        nonlocal ub, best, pb_obj
        x0 = np.zeros(len(cols))
        x0[[pos[p] for p in best]] = 1.0
        now = time.perf_counter()
        r = solve_columns(S, cols, now + share * (deadline - now), x0)
        if r is None:
            return
        pb_obj = r.obj
        if r.x is not None and r.obj < ub - EPS:
            ub = r.obj
            best = [cols[k] for k in np.flatnonzero(r.x > 0.5)]

    pos = {p: k for k, p in enumerate(cols)}
    if time.perf_counter() < deadline and ub - lb > EPS:
        price_and_branch(pb_share)     # part of the time; the rest for the enumeration

    # ------------------------------------------- reduced-cost enumeration
    enum_cols, nodes, enum_done = math.nan, 0, False
    if exact_done and ub - lb > EPS and time.perf_counter() < deadline:
        delta = ub - STEP - z_lp
        ec = enumerate_columns(S, pi, delta, enum_max, deadline)
        if ec is not None:
            enum_done = True
            enum_cols = len(ec)
            r = solve_columns(S, ec, deadline, cutoff=ub) if ec else None
            if not ec or (r is not None and r.status == "infeasible"):
                lb = ub          # no solution better than ub exists: ub is optimal
            elif r is not None:
                nodes = r.nodes
                if r.x is not None and r.obj < ub - EPS:
                    ub = r.obj
                    best = [ec[k] for k in np.flatnonzero(r.x > 0.5)]
                if r.status == "optimal":
                    lb = ub
                elif math.isfinite(r.bound):
                    lb = max(lb, min(ub, ceil_step(r.bound)))
    if not enum_done and ub - lb > EPS and time.perf_counter() < deadline - 1.0:
        price_and_branch(1.0)          # enumeration not possible: all the rest to the MILP
    status = "optimal" if ub - lb <= EPS else "time_limit"
    if status == "optimal":
        lb = ub

    shifts = [{"type": S.typ(p), "duties": list(p)} for p in best]
    return {"status": status, "obj": ub, "bound": lb, "gap": (ub - lb) / ub, "lp_bound": z_lp,
            "root_bound": ceil_step(z_lp) if exact_done else math.nan, "nodes": nodes,
            "runtime_sec": time.perf_counter() - t0, "cg_iters": iters, "cg_sec": cg_sec,
            "n_vars": len(cols), "cg_cols": ncols_cg, "enum_cols": enum_cols, "pb_obj": pb_obj,
            "greedy_ub": g["objective"], "shifts": shifts}
