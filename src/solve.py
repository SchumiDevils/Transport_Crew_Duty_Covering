"""Solve one instance with one model and return a row of statistics.

Usage: python src/solve.py <instance.json> --model AF|CF|CFM [--time-limit 300]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.heuristics.greedy import greedy  # noqa: E402
from src.instance import COST_HT, Instance, summary  # noqa: E402
from src.milp import solve, solve_from  # noqa: E402
from src.models.baseline import MODELS  # noqa: E402
from src.models.colgen import STEP, solve_sp  # noqa: E402
from src.models.improved import VARIANTS, TooLarge, build_cfp  # noqa: E402

ALL_MODELS = list(MODELS) + list(VARIANTS) + ["SP"]
from src.validator import validate  # noqa: E402

MAX_NNZ = 4_000_000   # larger models are not built (recorded as "too_large")


def pool_sizes(inst: Instance, z_bar: float) -> tuple[int, int]:
    """Employee pool bounds of Section 1.3, from a feasible cost z_bar."""
    ht_elig = sum(1 for d in inst.duration if d <= inst.W // 2)
    UF = min(inst.n, math.floor(z_bar + 1e-9))
    UH = min(ht_elig, math.floor(z_bar / COST_HT + 1e-9))
    return UF, UH


def estimate_nnz(model: str, inst: Instance, K: int) -> int:
    n, m = inst.n, len(inst.arcs)
    if model == "AF":
        return K * (4 * m + 8 * n)
    pairs = n * (n - 1) // 2
    return K * (3 * pairs + 4 * n) if model == "CF" else K * (3 * pairs + 8 * n)


def run(inst_path: str | Path, model: str, time_limit: float = 300, threads: int = 1,
        solver: str = "highs", sol_dir: Path | None = None, log_dir: Path | None = None,
        lp_time_limit: float = 60) -> dict:
    inst = Instance.from_json(inst_path)
    st = summary(inst)
    row = {k: st[k] for k in ("name", "n", "arcs", "density", "transitive", "lb")}
    row.update(instance=row.pop("name"), group=Path(inst_path).parent.name, model=model, solver=solver,
               time_limit=time_limit, threads=threads)
    if model == "SP":
        return _run_sp(inst, row, time_limit, sol_dir)

    # (CF)-type models are restrictions on non-transitive graphs: their pool
    # must be sized from a solution they can represent (pairwise-compatible).
    g = greedy(inst, pairwise=model != "AF")
    assert not validate(inst, g), "greedy produced an invalid solution"
    UF, UH = pool_sizes(inst, g["objective"])
    row.update(greedy_ub=g["objective"], UF=UF, UH=UH)

    plus = model in VARIANTS
    if not plus and estimate_nnz(model, inst, UF + UH) > 1.5 * MAX_NNZ:
        row.update(status="too_large")
        return row
    t0 = time.perf_counter()
    if plus:
        try:
            milp, decode, start = build_cfp(inst, UF, UH, max_nnz=MAX_NNZ, **VARIANTS[model])
        except TooLarge as exc:
            row.update(status="too_large", nnz=exc.args[0], build_sec=time.perf_counter() - t0)
            return row
    else:
        milp, decode = MODELS[model](inst, UF, UH)
    row.update(build_sec=time.perf_counter() - t0, n_vars=milp.num_vars, n_int=milp.num_int,
               n_rows=milp.num_rows, nnz=milp.nnz)
    if milp.nnz > MAX_NNZ:
        row.update(status="too_large")
        return row

    log = str(log_dir / f"{model}_{row['instance']}.log") if log_dir else None
    lp = solve(milp, solver, time_limit=lp_time_limit, threads=threads, relax=True)
    row.update(lp_bound=lp.obj if lp.status == "optimal" else math.nan, lp_sec=lp.runtime)

    if plus and model != "CF+nostart":
        x0 = start(g)
        assert x0 is not None, "greedy solution not representable in CF+"
        r = solve_from(milp, x0, STEP, solver, time_limit=time_limit, threads=threads, log_file=log)
    else:
        r = solve(milp, solver, time_limit=time_limit, threads=threads, log_file=log)
    row.update(status=r.status, obj=r.obj, bound=r.bound, gap=r.gap, root_bound=r.root_bound,
               nodes=r.nodes, runtime_sec=r.runtime)
    if r.x is not None:
        shifts = decode(r.x)
        # An incumbent may pay for employees with y_k = 1 but no duty (the
        # compact models allow it; never optimal). Validate the decoded
        # shifts at their own cost and record the idle cost separately.
        cost = sum(1.0 if s["type"] == "FT" else COST_HT for s in shifts)
        idle = r.obj - cost
        sol = {"instance": row["instance"], "model": model, "objective": round(cost, 6), "shifts": shifts}
        errs = validate(inst, sol)
        if idle < -1e-6:
            errs.append(f"decoded cost {cost:.4f} above reported {r.obj:.4f}")
        row.update(valid=not errs, num_ft=sum(s["type"] == "FT" for s in shifts),
                   num_ht=sum(s["type"] == "HT" for s in shifts),
                   errors="; ".join(errs[:3]) or (f"idle employees cost {idle:.1f}" if idle > 1e-6 else ""))
        if sol_dir:
            sol_dir.mkdir(parents=True, exist_ok=True)
            (sol_dir / f"{model}_{row['instance']}.json").write_text(json.dumps(sol))
    return row


def _run_sp(inst: Instance, row: dict, time_limit: float, sol_dir: Path | None) -> dict:
    """Column generation + price-and-branch + reduced-cost enumeration (Section 5)."""
    r = solve_sp(inst, time_limit)
    shifts = r.pop("shifts")
    row.update(r)
    sol = {"instance": row["instance"], "model": "SP", "objective": round(r["obj"], 6), "shifts": shifts}
    errs = validate(inst, sol)
    row.update(valid=not errs, errors="; ".join(errs[:3]), num_ft=sum(s["type"] == "FT" for s in shifts),
               num_ht=sum(s["type"] == "HT" for s in shifts))
    if sol_dir:
        sol_dir.mkdir(parents=True, exist_ok=True)
        (sol_dir / f"SP_{row['instance']}.json").write_text(json.dumps(sol))
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("instance")
    ap.add_argument("--model", default="AF", choices=ALL_MODELS)
    ap.add_argument("--solver", default="highs", choices=["highs", "gurobi"])
    ap.add_argument("--time-limit", type=float, default=300)
    ap.add_argument("--threads", type=int, default=1)
    a = ap.parse_args()
    row = run(a.instance, a.model, a.time_limit, a.threads, a.solver)
    for k, v in row.items():
        print(f"{k:>12}: {v}")


if __name__ == "__main__":
    main()
