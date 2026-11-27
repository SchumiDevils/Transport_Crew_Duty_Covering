# Transport Crew Duty Covering

Project n. 2 of *Model-driven Decision Methods*, Università di Pisa,
A.Y. 2026/2027. Group 1: Iachim Cristian.

The full report is [`report/main.pdf`](report/main.pdf). This README
summarises the project and explains how to run the code and reproduce every
number, table and figure in the report.

## Problem

A transport company must cover a set of driving duties, each with a start
and an end time. A compatibility graph G states which duty can follow which
for the same employee: duty j can follow duty i if the transfer time from the
end of i to the start of j fits in between. Every employee has a maximum
working time (sum of the driving times) and a maximum total time (from the
start of the first duty to the end of the last one). Full-time employees have
limits (W, T) and cost 1. Half-time employees have limits (W/2, T/2) and cost
0.6. The goal is to assign every duty to exactly one employee at minimum total
cost.

Equivalently: partition the duties into paths of the DAG G, each path
feasible for the type of employee it is given to. The problem is strongly
NP-hard (report, Proposition 6). Working time counts only the driving time of
the duties, not the transfers between them (report, §1.2).

## Methods

| Model | Idea | Exact on | Report |
|---|---|---|---|
| (AF) | arc-flow, one flow per candidate employee | any DAG | §1.3 |
| (CF) | assignment with pairwise conflict constraints | transitive G | §1.3 |
| (CF-M) | (CF) with big-M spread constraints | transitive G | §1.3 |
| (CF+) | (CF) with clique constraints, symmetry breaking and an improving cut | transitive G | §5.1 |
| (SP-CG) | set partitioning by column generation, with price-and-branch and reduced-cost enumeration for optimality | any DAG | §5.2 |

On non-transitive graphs, such as the OR-Library instances, the (CF)-type
models are restrictions of the problem: their solutions are feasible, but they
can miss the optimum.

## Main results

76 instances (66 generated, 10 from the OR-Library), 300 s per run, HiGHS
1.16, one thread per run.

| | (AF) | (CF) | (CF-M) | (CF+) | (SP-CG) |
|---|---|---|---|---|---|
| instances solved to optimality | 15 | 47 | 41 | 49 | **70** |
| largest n solved (all seeds) | 25 | 50 | 50 | 50 | **200** |
| mean LP gap | 19.6% | 29.6% | 29.5% | 24.2% | **0.1%** |

(SP-CG) finds the best known solution on all 76 instances. It proves
optimality for the OR-Library instances up to n = 200 and ends within 1.2–3.1%
of optimality on the larger ones. Section 6 of the report has the full
comparison, including an ablation study of (CF+).

## Setup

Python 3.12. From this folder:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate      Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

All experiments use **HiGHS 1.16.0**, which needs no licence. Other HiGHS
versions may give different running times. The report (§5.1, §6.1) describes
two HiGHS 1.16 behaviours that the code works around. Gurobi is optional:
without a licence it is only used to cross-check the small tests.

## Quick check (under a minute)

```bash
python tests/test_small.py
```

This runs 9 hand-made instances with known optimum and 8 generated instances
through every model, and checks each solution with the independent validator.
It also checks that the validator rejects 5 broken solutions. It ends with
`ALL PASSED`.

Solve one instance with one model:

```bash
python src/solve.py data/instances/orlib/csp100.json --model SP --time-limit 60
```

Models: `AF`, `CF`, `CFM` (baseline, §1.3); `CF+` and its ablation variants
`CF+noclq`, `CF+chain`, `CF+nosym`, `CF+nostart` (§5.1); and `SP` (column
generation, §5.2). To check any solution file independently:

```bash
python src/validator.py <instance.json> <solution.json>
```

## Full reproduction

The instances, and the tables and figures computed from the experiment CSV
files, regenerate byte-identical, up to line endings, which follow the
operating system. The experiments themselves stop at a time limit of 300 s, so
on a different machine the running times will differ. For runs that hit the
limit, the final solution and bound can differ too, and in borderline cases so
can whether a run finishes in time. Proven optimal values cannot.

The results in the report were obtained on an Intel Core i5-12500H with 16 GB
of RAM (Windows 11), with 1 thread per run and 5 runs in parallel. To check
the analysis without rerunning the experiments, run the two `analyze_*.py`
scripts on the CSV files provided.

| Step | Command | Output | Time |
|---|---|---|---|
| Instances (§2) | `python data/generator.py --suite` | `data/instances/{scale,factor}` (66) | seconds |
| | `python data/convert_orlib.py` | `data/instances/orlib` (10) | seconds |
| | `python data/summarize.py` | `results/instances_summary.csv`, `report/tables/inst_*.tex` | seconds |
| | `python data/plot_instances.py` | `report/figures/instances.pdf` | seconds |
| Baseline (§4) | `python src/run_experiments.py --models AF CF CFM --groups scale factor orlib --time-limit 300 --threads 1 --workers 5 --out results/raw/baseline.csv` | `results/raw/baseline*` | ~1.7 h |
| | `python src/analyze_baseline.py` | `report/tables/base_*.tex`, `report/figures/base_scale.pdf` | seconds |
| Improvements (§6) | `python src/run_experiments.py --models CF+ SP --groups scale factor orlib --time-limit 300 --threads 1 --workers 5 --out results/raw/improved.csv` | `results/raw/improved*` | ~35 min |
| Ablation (§6.6) | `python src/run_experiments.py --models CF+noclq CF+chain CF+nosym CF+nostart --groups scale --min-n 40 --max-n 100 --time-limit 300 --threads 1 --workers 5 --out results/raw/ablation.csv` | `results/raw/ablation*` | ~1 h |
| | `python src/analyze_improved.py` | `report/tables/cmp_*.tex`, `report/figures/cmp_*.pdf` | seconds |
| Report | `cd report && latexmk -pdf main.tex` | `report/main.pdf` | seconds |

The experiment runner skips runs already present in its output CSV, so an
interrupted experiment resumes when the same command is run again. To rerun
from scratch, delete or rename the CSV first.

`python make_archive.py` packs the report and all files of this repository
into `dist/MDDM_Project2_Group1.zip`.

## Contents

```text
report/            main.pdf (the report), LaTeX sources, generated tables and figures
src/
  instance.py      instance format (JSON), structural statistics, lower bounds
  milp.py          solver-independent MILP container; HiGHS and Gurobi back-ends
  models/
    baseline.py    (AF) arc-flow, (CF) conflict, (CF-M) big-M formulations   (§1.3)
    improved.py    (CF+) strengthened conflict formulation + ablation variants (§5.1)
    colgen.py      (SP-CG) column generation, price-and-branch, enumeration   (§5.2)
  heuristics/
    greedy.py      constructive heuristic (pool sizes, starting solution)
  validator.py     independent feasibility and cost checker
  solve.py         one instance x one model -> statistics row
  run_experiments.py   parallel, resumable benchmark runner
  analyze_baseline.py, analyze_improved.py   tables and figures of §4 and §6
data/
  generator.py     scalable instance generator (seeded)                         (§2)
  convert_orlib.py OR-Library crew scheduling files -> JSON
  orlib/           original OR-Library files (Beasley & Cao, 1996)
  instances/       the 76 benchmark instances (scale, factor, orlib)
tests/test_small.py    verification tests (§3.3, §5.3)
results/
  instances_summary.csv
  raw/{baseline,improved,ablation}.csv   one row per run (all metrics)
  raw/*_solutions/  every solution found (JSON, validator format)
  raw/*_logs/       HiGHS logs of the compact-model runs (SP-CG runs write none)
make_archive.py    builds the submission archive
```

## Instance and solution format

An instance is a JSON file with `W` and `T` (minutes), the duty `start` and
`end` times, and the arcs of the compatibility graph (`arcs`, pairs of duty
indices). A solution is
`{"instance", "objective", "shifts": [{"type": "FT"|"HT", "duties": [...]}]}`,
with each shift's duties in order. See `src/instance.py` and
`src/validator.py`.
