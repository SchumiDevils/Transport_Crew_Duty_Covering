"""Run a set of models on the benchmark suite and append one CSV row per run.

Runs are executed in parallel worker processes (each solver run uses
`--threads` threads). Already completed (instance, model) pairs found in the
output CSV are skipped, so an interrupted experiment can be resumed.

Examples (Sections 4 and 6):
    python src/run_experiments.py --models AF CF CFM --groups scale factor orlib \
        --time-limit 300 --workers 5 --out results/raw/baseline.csv
    python src/run_experiments.py --models CF+ SP --groups scale factor orlib \
        --time-limit 300 --workers 5 --out results/raw/improved.csv
    python src/run_experiments.py --models CF+noclq CF+chain CF+nosym CF+nostart \
        --groups scale --min-n 40 --max-n 100 --time-limit 300 --workers 5 \
        --out results/raw/ablation.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

FIELDS = ["instance", "group", "model", "solver", "n", "arcs", "density", "transitive", "lb",
          "greedy_ub", "UF", "UH", "n_vars", "n_int", "n_rows", "nnz", "build_sec", "status",
          "obj", "bound", "gap", "lp_bound", "lp_sec", "root_bound", "nodes", "runtime_sec",
          "num_ft", "num_ht", "valid", "errors", "time_limit", "threads",
          # column generation only (Section 5)
          "cg_iters", "cg_sec", "cg_cols", "enum_cols", "pb_obj"]


def _job(args):
    path, model, tl, threads, solver, sol_dir, log_dir = args
    from src.solve import run
    try:
        return run(path, model, tl, threads, solver, Path(sol_dir), Path(log_dir))
    except Exception as exc:  # keep the experiment going, record the failure
        return {"instance": Path(path).stem, "group": Path(path).parent.name, "model": model,
                "solver": solver, "status": "error", "errors": repr(exc)[:300],
                "time_limit": tl, "threads": threads, "_tb": traceback.format_exc()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["AF", "CF", "CFM"])
    ap.add_argument("--groups", nargs="+", default=["scale", "factor", "orlib"])
    ap.add_argument("--min-n", type=int, default=0)
    ap.add_argument("--max-n", type=int, default=10**9)
    ap.add_argument("--time-limit", type=float, default=300)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--solver", default="highs")
    ap.add_argument("--out", default=str(ROOT / "results" / "raw" / "baseline.csv"))
    a = ap.parse_args()

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        with open(out) as f:
            done = {(r["instance"], r["model"]) for r in csv.DictReader(f) if r["status"] != "error"}
    sol_dir = out.parent / (out.stem + "_solutions")
    log_dir = out.parent / (out.stem + "_logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    import json
    jobs = []
    for g in a.groups:
        for p in sorted((ROOT / "data" / "instances" / g).glob("*.json"),
                        key=lambda p: (json.loads(p.read_text())["n"], p.name)):
            if not a.min_n <= json.loads(p.read_text())["n"] <= a.max_n:
                continue
            for mdl in a.models:
                if (p.stem, mdl) not in done:
                    jobs.append((str(p), mdl, a.time_limit, a.threads, a.solver, str(sol_dir), str(log_dir)))
    print(f"{len(jobs)} runs to do ({len(done)} already in {out})", flush=True)

    new_file = not out.exists()
    t0 = time.time()
    with open(out, "a", newline="") as f, ProcessPoolExecutor(a.workers) as pool:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        if new_file:
            w.writeheader()
        futs = {pool.submit(_job, j): j for j in jobs}
        for k, fut in enumerate(as_completed(futs), 1):
            row = fut.result()
            if row.get("_tb"):
                print(row["_tb"], file=sys.stderr)
            w.writerow(row)
            f.flush()
            print(f"[{k}/{len(jobs)} {time.time() - t0:7.0f}s] {row['model']:10s} {row['instance']:24s} "
                  f"{row.get('status')} obj={row.get('obj', '')} valid={row.get('valid', '')}", flush=True)


if __name__ == "__main__":
    main()
