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
from src.milp import solve  # noqa: E402
from src.models.baseline import MODELS  # noqa: E402
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

    # (CF)/(CF-M) are restrictions on non-transitive graphs: their pool must be
    # sized from a solution they can represent (pairwise-compatible shifts).
    g = greedy(inst, pairwise=model != "AF")
    assert not validate(inst, g), "greedy produced an invalid solution"
    UF, UH = pool_sizes(inst, g["objective"])
    row.update(greedy_ub=g["objective"], UF=UF, UH=UH)

    if estimate_nnz(model, inst, UF + UH) > 1.5 * MAX_NNZ:
        row.update(status="too_large")
        return row
    t0 = time.perf_counter()
    milp, decode = MODELS[model](inst, UF, UH)
    row.update(build_sec=time.perf_counter() - t0, n_vars=milp.num_vars, n_int=milp.num_int,
               n_rows=milp.num_rows, nnz=milp.nnz)
    if milp.nnz > MAX_NNZ:
        row.update(status="too_large")
        return row

    log = str(log_dir / f"{model}_{row['instance']}.log") if log_dir else None
    lp = solve(milp, solver, time_limit=lp_time_limit, threads=threads, relax=True)
    row.update(lp_bound=lp.obj if lp.status == "optimal" else math.nan, lp_sec=lp.runtime)

    r = solve(milp, solver, time_limit=time_limit, threads=threads, log_file=log)
    row.update(status=r.status, obj=r.obj, bound=r.bound, gap=r.gap, root_bound=r.root_bound,
               nodes=r.nodes, runtime_sec=r.runtime)
    if r.x is not None:
        shifts = decode(r.x)
        sol = {"instance": row["instance"], "model": model, "objective": round(r.obj, 6), "shifts": shifts}
        errs = validate(inst, sol)
        row.update(valid=not errs, num_ft=sum(s["type"] == "FT" for s in shifts),
                   num_ht=sum(s["type"] == "HT" for s in shifts),
                   errors="; ".join(errs[:3]))
        if sol_dir:
            sol_dir.mkdir(parents=True, exist_ok=True)
            (sol_dir / f"{model}_{row['instance']}.json").write_text(json.dumps(sol))
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("instance")
    ap.add_argument("--model", default="AF", choices=list(MODELS))
    ap.add_argument("--solver", default="highs", choices=["highs", "gurobi"])
    ap.add_argument("--time-limit", type=float, default=300)
    ap.add_argument("--threads", type=int, default=1)
    a = ap.parse_args()
    row = run(a.instance, a.model, a.time_limit, a.threads, a.solver)
    for k, v in row.items():
        print(f"{k:>12}: {v}")


if __name__ == "__main__":
    main()
