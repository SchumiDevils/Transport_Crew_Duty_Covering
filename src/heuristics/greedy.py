"""Greedy constructive heuristic.

Used in the baseline only to obtain a feasible cost z_bar, which bounds the
size of the employee pool (Section 1.3): an optimal solution uses at most
floor(z_bar) full-time and floor(z_bar / 0.6) half-time employees.

Duties are scanned by start time; each duty is appended to the open shift
whose last duty has an arc to it in G, which still respects the limits, and
which ended most recently ("best fit": least idle time); otherwise a new shift
is opened. The scan is done twice, once with the FT limits and once with the
HT limits (duties longer than W/2 then get a FT shift of their own), and the
cheaper result is kept. Each shift finally gets the cheapest feasible type.
"""
from __future__ import annotations

from src.instance import COST_FT, COST_HT, Instance


def shift_type(inst: Instance, duties: list[int]) -> str:
    work = sum(inst.end[i] - inst.start[i] for i in duties)
    spread = inst.end[duties[-1]] - inst.start[duties[0]]
    return "HT" if work <= inst.W // 2 and spread <= inst.T // 2 else "FT"


def _scan(inst: Instance, W: int, T: int, pairwise: bool) -> list[list[int]]:
    arcs = set(inst.arcs)
    d = inst.duration
    shifts: list[dict] = []
    for i in sorted(range(inst.n), key=lambda k: (inst.start[k], inst.end[k])):
        best = None
        for sh in shifts:
            last = sh["duties"][-1]
            linked = all((j, i) in arcs for j in sh["duties"]) if pairwise else (last, i) in arcs
            if linked and sh["work"] + d[i] <= W and inst.end[i] - sh["first"] <= T:
                if best is None or inst.end[last] > inst.end[best["duties"][-1]]:
                    best = sh
        if best is None:
            shifts.append({"duties": [i], "work": d[i], "first": inst.start[i]})
        else:
            best["duties"].append(i)
            best["work"] += d[i]
    return [s["duties"] for s in shifts]


def greedy(inst: Instance, pairwise: bool = False) -> dict:
    """Return a feasible solution {"objective", "shifts": [{"type", "duties"}]}.

    pairwise=True only builds shifts whose duties are pairwise joined by arcs,
    i.e. solutions that are also feasible for (CF). On transitive graphs both
    modes give the same result; on general DAGs this is needed to size the
    pool of (CF), which is then a restriction of the problem.
    """
    best = None
    for W, T in [(inst.W, inst.T), (inst.W // 2, inst.T // 2)]:
        # with HT limits, a duty longer than W/2 still opens its own (FT) shift
        shifts = _scan(inst, W, T, pairwise)
        sol = [{"type": shift_type(inst, s), "duties": s} for s in shifts]
        cost = sum(COST_FT if s["type"] == "FT" else COST_HT for s in sol)
        if best is None or cost < best["objective"] - 1e-9:
            best = {"objective": round(cost, 6), "shifts": sol}
    return best
