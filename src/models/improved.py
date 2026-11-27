"""Strengthened conflict formulation (CF+) of Section 5.1.

Same variables y_k, x_ik as (CF) (Section 1.3), but the O(|K| n^2) pairwise
conflict rows are replaced by O(|K| n) rows that are at least as strong:

  * time conflicts  -> clique rows  sum_{i in Q} x_ik <= y_k  over a clique
    cover of the time-conflict graph (interval cliques of duties running at
    the same moment, then a greedy cover of the remaining conflict pairs);
  * spread conflicts: pairwise rows as in (CF) (default), or optionally
    (chain=True) a monotone "started-before" chain per employee:
        sigma_qk >= sigma_{q-1,k},  sigma_qk >= x_{(q)k},
        x_jk + sigma_{p(j),k} <= y_k,
    where duties are sorted by start, (q) is the q-th duty and p(j) the last
    position with s < e_j - T_k. Its LP projection on (x, y) is exactly the
    set of pairwise spread rows of (CF). It is O(n) rows instead of O(n^2),
    but in our tests HiGHS's root cuts were weaker with it (the pairwise
    conflicts feed its clique table), so it is off by default;
  * symmetry breaking: within each type, employees are ordered by their first
    duty, so duty at position q can only go to employees k <= q, and
    y_k >= y_{k+1};
  * the (pairwise) greedy solution as starting point, passed through the
    improving cut of src/milp.py:solve_from (HiGHS 1.16 MIP starts are
    unreliable, see the note there).

Each component can be switched off for the ablation study.
Like (CF), the model is exact on transitive graphs and a restriction otherwise.
"""
from __future__ import annotations

from src.instance import COST_FT, COST_HT, Instance
from src.milp import INF, MILP
from src.models.baseline import time_conflicts


def interval_cliques(inst: Instance) -> list[list[int]]:
    """Maximal sets of duties running at a common moment (one per start time)."""
    s, e = inst.start, inst.end
    order = sorted(range(inst.n), key=lambda i: s[i])
    cl = [frozenset(i for i in range(inst.n) if s[i] <= s[j] < e[i]) for j in order]
    # If the set at one start time is contained in the set at a later start
    # time, it is contained in the set at the next start time (its intervals
    # stay active in between), so comparing neighbours is enough.
    return [sorted(q) for a, q in enumerate(cl) if a + 1 == len(cl) or not q <= cl[a + 1]]


def clique_cover(inst: Instance) -> list[list[int]]:
    """Cliques of the time-conflict graph covering all its edges."""
    n = inst.n
    adj = [set() for _ in range(n)]
    for i, j in time_conflicts(inst):
        adj[i].add(j)
        adj[j].add(i)
    cliques = interval_cliques(inst)
    covered = set()
    for q in cliques:
        for a in range(len(q)):
            for b in range(a + 1, len(q)):
                covered.add((q[a], q[b]))
    order = sorted(range(n), key=lambda i: inst.start[i])
    for i in order:
        for j in sorted(adj[i]):
            if j < i or (min(i, j), max(i, j)) in covered:
                continue
            # grow {i, j} greedily, preferring vertices with uncovered edges
            q = [i, j]
            cand = adj[i] & adj[j]
            while cand:
                v = max(cand, key=lambda v: (sum((min(v, u), max(v, u)) not in covered for u in q), -v))
                q.append(v)
                cand &= adj[v]
            q.sort()
            for a in range(len(q)):
                for b in range(a + 1, len(q)):
                    covered.add((q[a], q[b]))
            cliques.append(q)
    return cliques


class TooLarge(Exception):
    pass


def build_cfp(inst: Instance, UF: int, UH: int, cliques: bool = True, chain: bool = False,
              sym: bool = True, max_nnz: int | None = None):
    n, s, e, d = inst.n, inst.start, inst.end, inst.duration
    emp = ([("FT", inst.W, inst.T, COST_FT, k) for k in range(UF)] +
           [("HT", inst.W // 2, inst.T // 2, COST_HT, k) for k in range(UH)])
    order = sorted(range(n), key=lambda i: (s[i], e[i]))
    pos = {i: q for q, i in enumerate(order)}
    tc = clique_cover(inst) if cliques else [list(p) for p in time_conflicts(inst)]

    m = MILP()
    y = [m.add_var(cost=ck) for (_, _, _, ck, _) in emp]
    x = []                                   # x[k][i] = column index or None
    for typ, Wk, Tk, ck, r in emp:
        x.append([m.add_var() if d[i] <= Wk and (not sym or pos[i] >= r) else None
                  for i in range(n)])

    for i in range(n):                                        # coverage
        idx = [x[k][i] for k in range(len(emp)) if x[k][i] is not None]
        m.add_row(idx, [1.0] * len(idx), 1.0, 1.0)

    sig = []
    for k, (typ, Wk, Tk, ck, r) in enumerate(emp):
        xk = x[k]
        if not cliques:                                       # linking (as in CF)
            for i in range(n):
                if xk[i] is not None:
                    m.add_row([xk[i], y[k]], [1.0, -1.0], -INF, 0.0)
        for q in tc:                                          # time cliques / pairs
            idx = [xk[i] for i in q if xk[i] is not None]
            if len(idx) >= (1 if cliques else 2):
                m.add_row(idx + [y[k]], [1.0] * len(idx) + [-1.0], -INF, 0.0)
        idx = [xk[i] for i in range(n) if xk[i] is not None]  # working time
        m.add_row(idx + [y[k]], [float(d[i]) for i in range(n) if xk[i] is not None] + [-float(Wk)],
                  -INF, 0.0)
        # spread: p(j) = last position q with s_(q) < e_j - T_k
        pj = {}
        q = 0
        for j in sorted(range(n), key=lambda j: e[j]):
            while q < n and s[order[q]] < e[j] - Tk:
                q += 1
            if q > 0 and xk[j] is not None:
                pj[j] = q - 1
        if chain and pj:
            top = max(pj.values())
            sg = [m.add_var(0.0, 0.0, 1.0, False) for _ in range(top + 1)]
            sig.append((k, sg))
            for qq in range(top + 1):
                i = order[qq]
                if xk[i] is not None:
                    m.add_row([sg[qq], xk[i]], [1.0, -1.0], 0.0, INF)
                if qq > 0:
                    m.add_row([sg[qq], sg[qq - 1]], [1.0, -1.0], 0.0, INF)
            for j, p in pj.items():
                m.add_row([xk[j], sg[p], y[k]], [1.0, 1.0, -1.0], -INF, 0.0)
        elif pj:                                              # pairwise (as in CF)
            tcs = {tuple(sorted(p)) for p in time_conflicts(inst)}
            for j, p in pj.items():
                for qq in range(p + 1):
                    i = order[qq]
                    if xk[i] is not None and tuple(sorted((i, j))) not in tcs:
                        m.add_row([xk[i], xk[j], y[k]], [1.0, 1.0, -1.0], -INF, 0.0)
        if sym and r > 0:                                     # y_{k-1} >= y_k
            m.add_row([y[k - 1], y[k]], [1.0, -1.0], 0.0, INF)
        if max_nnz is not None and m.nnz > max_nnz:
            raise TooLarge(m.nnz)

    def decode(xv):
        shifts = []
        for k, (typ, *_r) in enumerate(emp):
            duties = sorted((i for i in range(n) if x[k][i] is not None and xv[x[k][i]] > 0.5),
                            key=lambda i: s[i])
            if duties:
                shifts.append({"type": typ, "duties": duties})
        return shifts

    def start(sol: dict):
        """Solution vector for a solution whose shifts are pairwise compatible."""
        v = [0.0] * m.num_vars
        for typ, off in (("FT", 0), ("HT", UF)):
            sh = sorted((sorted(t["duties"], key=lambda i: pos[i]) for t in sol["shifts"]
                         if t["type"] == typ), key=lambda dl: pos[dl[0]])
            for r, dl in enumerate(sh):
                k = off + r
                v[y[k]] = 1.0
                for i in dl:
                    if x[k][i] is None:
                        return None           # incompatible with the symmetry rule
                    v[x[k][i]] = 1.0
        for k, sg in sig:
            run = 0.0
            for qq in range(len(sg)):
                i = order[qq]
                if x[k][i] is not None:
                    run = max(run, v[x[k][i]])
                v[sg[qq]] = run
        return v

    return m, decode, start


# model name -> keyword arguments (ablation: each component switched off once)
VARIANTS = {
    "CF+": {},
    "CF+noclq": {"cliques": False},
    "CF+chain": {"chain": True},
    "CF+nosym": {"sym": False},
    "CF+nostart": {},          # same model, without the improving cut (see solve.py)
}
