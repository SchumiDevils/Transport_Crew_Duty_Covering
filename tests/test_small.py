"""Hand-crafted instances with known optimal cost (Section 3.3 of the report).

Run:  python tests/test_small.py            (HiGHS; also Gurobi if available)

Every model is solved to optimality, its solution is checked by the
validator, and the cost is compared with the value derived by hand.
W = 480, T = 720 unless stated (half-time: 240 / 360).
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.heuristics.greedy import greedy  # noqa: E402
from src.instance import Instance  # noqa: E402
from src.milp import solve  # noqa: E402
from src.models.baseline import MODELS  # noqa: E402
from src.solve import pool_sizes  # noqa: E402
from src.validator import validate  # noqa: E402


def make(name, iv, arcs=None, W=480, T=720):
    s, e = [a for a, _ in iv], [b for _, b in iv]
    if arcs is None:  # all time-feasible pairs (zero transfer times -> transitive)
        arcs = [(i, j) for i in range(len(iv)) for j in range(len(iv)) if i != j and e[i] <= s[j]]
    inst = Instance(name, W, T, s, e, arcs, source="test")
    inst.check()
    return inst


# 3-PARTITION yes-instance (m = 2, B = 100), duties back to back: optimum = m
_a = [26, 34, 40, 30, 33, 37]
_s = [sum(_a[:k]) for k in range(6)]

# name -> (instance, expected optimum for AF / CF / CFM, explanation)
CASES = [
    (make("all_ht", [(0, 100), (110, 200)]), 0.6, "190 min work, 200 min spread: one HT"),
    (make("overlap", [(0, 100), (50, 150)]), 1.2, "overlapping: two HT"),
    (make("work_binds", [(0, 200), (210, 400)]), 1.0, "390 > W/2: one FT beats two HT"),
    (make("ht_spread_binds", [(0, 60), (600, 660)]), 1.0, "spread 660 > T/2: one FT"),
    (make("ft_spread_binds", [(0, 60), (700, 760)]), 1.2, "spread 760 > T: two HT"),
    (make("mixed", [(0, 200), (210, 400), (50, 150)], arcs=[(0, 1)]), 1.6, "FT{0,1} + HT{2}"),
    (make("long_duty", [(0, 300)]), 1.0, "d > W/2: FT only"),
    (make("three_partition", [(s, s + a) for s, a in zip(_s, _a)], W=100, T=800), 2.0,
     "Prop. 6 reduction, yes-instance: m FT"),
]
# Non-transitive graph: 0->1->2 but no arc 0->2. AF finds the path (one HT);
# CF/CFM see {0,2} as a conflict -> restriction, optimum 1.2.
NONTRANS = make("non_transitive", [(0, 50), (60, 110), (120, 170)], arcs=[(0, 1), (1, 2)])


def check(inst, expected: dict, solvers) -> int:
    fails = 0
    for solver in solvers:
        for model, exp in expected.items():
            g = greedy(inst, pairwise=model != "AF")
            if validate(inst, g):
                print(f"  greedy INVALID on {inst.name}")
                fails += 1
            UF, UH = pool_sizes(inst, g["objective"])
            milp, decode = MODELS[model](inst, UF, UH)
            r = solve(milp, solver, time_limit=60)
            sol = {"objective": r.obj, "shifts": decode(r.x)}
            errs = validate(inst, sol)
            ok = r.status == "optimal" and math.isclose(r.obj, exp, abs_tol=1e-6) and not errs
            fails += not ok
            print(f"  {'ok  ' if ok else 'FAIL'} {inst.name:16s} {model:3s} {solver:6s} "
                  f"obj={r.obj:.2f} expected={exp:.2f} greedy={g['objective']:.2f} {errs or ''}")
    return fails


def main() -> None:
    solvers = ["highs"]
    try:
        import gurobipy  # noqa: F401
        solvers.append("gurobi")   # tiny models fit the size-limited licence
    except ImportError:
        pass
    fails = 0
    for inst, opt, why in CASES:
        print(f"{inst.name}: {why}")
        fails += check(inst, {"AF": opt, "CF": opt, "CFM": opt}, solvers)
    print("non_transitive: AF exact (0.6), CF/CFM restriction (1.2)")
    fails += check(NONTRANS, {"AF": 0.6, "CF": 1.2, "CFM": 1.2}, solvers)
    # The validator must reject broken solutions (negative tests).
    inst = CASES[2][0]  # work_binds: duties 0 (200 min) -> 1 (190 min)
    bad = {
        "HT over working time": {"objective": 0.6, "shifts": [{"type": "HT", "duties": [0, 1]}]},
        "duty missing": {"objective": 1.0, "shifts": [{"type": "FT", "duties": [0]}]},
        "duty twice": {"objective": 2.0, "shifts": [{"type": "FT", "duties": [0, 1]},
                                                     {"type": "FT", "duties": [1]}]},
        "wrong cost": {"objective": 0.8, "shifts": [{"type": "FT", "duties": [0, 1]}]},
        "not a path": {"objective": 0.6, "shifts": [{"type": "HT", "duties": [0, 2]},
                                                     {"type": "HT", "duties": [1]}]},
    }
    bad["not a path"]["_inst"] = NONTRANS
    for what, sol in bad.items():
        errs = validate(sol.pop("_inst", inst), sol)
        fails += not errs
        print(f"  {'ok  ' if errs else 'FAIL'} validator rejects: {what} -> {errs[:1]}")
    print("ALL PASSED" if fails == 0 else f"{fails} FAILURES")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
