"""Scalable instance generator for the crew duty covering problem (Project n. 2).

Model of a day of urban/regional bus service:
  * L terminals placed uniformly in a square; travel time between terminals is
    the Euclidean distance (scaled so that the largest one is about
    `travel_max` minutes), rounded up, plus a fixed changeover time
    `turnaround`. Both operations preserve the triangle inequality, so the
    compatibility graph is transitive (Proposition 1) unless arcs are
    removed with `arc_drop`.
  * Each duty starts at a random terminal and ends at a random terminal.
    Start times follow either a uniform profile over the service day or a
    "peak" profile (morning and evening rush hours on top of a uniform base).
  * Durations are drawn uniformly from the range of a duration class.
  * Arc (i, j) exists iff e_i + t(end_term_i, start_term_j) <= s_j.
  * `arc_drop` > 0 removes that fraction of arcs at random, which models a
    "given" compatibility graph with extra restrictions (e.g. relief points
    where a driver change is not allowed) and makes G non-transitive.

Usage:
    python data/generator.py --n 40 --seed 1 --out data/instances/test.json
    python data/generator.py --suite            # writes the benchmark suite
"""
from __future__ import annotations

import argparse
import itertools
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.instance import Instance  # noqa: E402

# --------------------------------------------------------------------------- #
# Parameter presets (all times in minutes)
# --------------------------------------------------------------------------- #
DURATION_CLASSES = {
    "short": (20, 90),     # almost everything fits a half-time shift
    "mixed": (20, 240),    # default: up to W/2
    "long": (120, 300),    # many duties only coverable by full-time employees
}
TRAVEL_LEVELS = {"low": 10, "med": 40, "high": 150}  # urban / suburban / regional network

DEFAULTS = dict(
    n=40,
    profile="peak",        # "uniform" | "peak"
    durations="mixed",     # key of DURATION_CLASSES
    travel="med",          # key of TRAVEL_LEVELS
    terminals=8,
    turnaround=5,
    W=480,                 # FT max working time  (8 h)
    T=720,                 # FT max total time    (12 h)  -> W/T = 0.67
    arc_drop=0.0,
    day_start=300,         # 05:00
    day_end=1440,          # 24:00
)

# Peak profile: mixture weights and (mean, std) of the two rush hours
PEAKS = [(0.35, 450, 45), (0.35, 1050, 60)]  # 07:30 and 17:30


def _sample_starts(rng: np.random.Generator, n: int, durations: np.ndarray, p: dict) -> np.ndarray:
    lo, hi = p["day_start"], p["day_end"]
    latest = hi - durations  # duty must end within the service day
    starts = np.empty(n, dtype=int)
    for i in range(n):
        while True:
            if p["profile"] == "uniform":
                s = rng.uniform(lo, latest[i])
            elif p["profile"] == "peak":
                u = rng.random()
                acc = 0.0
                s = None
                for w, mu, sd in PEAKS:
                    acc += w
                    if u < acc:
                        s = rng.normal(mu, sd)
                        break
                if s is None:
                    s = rng.uniform(lo, latest[i])
            else:
                raise ValueError(f"unknown profile {p['profile']}")
            if lo <= s <= latest[i]:
                starts[i] = int(round(s))
                break
    return starts


def generate(seed: int, name: str | None = None, **overrides) -> Instance:
    """Generate one instance. Same (seed, parameters) -> same instance."""
    p = {**DEFAULTS, **overrides}
    rng = np.random.default_rng(seed)
    n, L = p["n"], p["terminals"]

    # Terminals and travel times between terminals
    side = TRAVEL_LEVELS[p["travel"]] / math.sqrt(2)
    xy = rng.uniform(0.0, side, size=(L, 2))
    travel = [[p["turnaround"] + math.ceil(float(np.hypot(*(xy[a] - xy[b])))) for b in range(L)]
              for a in range(L)]

    # Duties
    dlo, dhi = DURATION_CLASSES[p["durations"]]
    durations = rng.integers(dlo, dhi + 1, size=n)
    starts = _sample_starts(rng, n, durations, p)
    order = np.argsort(starts, kind="stable")  # number duties by start time
    starts, durations = starts[order], durations[order]
    ends = starts + durations
    st = rng.integers(0, L, size=n)
    et = rng.integers(0, L, size=n)

    # Compatibility graph
    arcs = [(i, j) for i in range(n) for j in range(n)
            if i != j and ends[i] + travel[et[i]][st[j]] <= starts[j]]
    if p["arc_drop"] > 0:
        keep = rng.random(len(arcs)) >= p["arc_drop"]
        arcs = [a for a, k in zip(arcs, keep) if k]

    inst = Instance(
        name=name or f"gen_n{n}_s{seed}",
        W=p["W"], T=p["T"],
        start=starts.tolist(), end=ends.tolist(),
        arcs=arcs,
        source="generator",
        params={**p, "seed": seed},
        start_terminal=st.tolist(), end_terminal=et.tolist(),
        terminal_travel=travel,
    )
    inst.check()
    return inst


# --------------------------------------------------------------------------- #
# Benchmark suite
# --------------------------------------------------------------------------- #
SCALE_SIZES = [10, 15, 20, 25, 30, 40, 50, 60, 80, 100, 150, 200]
FACTOR_N = 40
FACTORS = {  # one-factor-at-a-time variations around DEFAULTS
    "profile": ["uniform"],
    "durations": ["short", "long"],
    "travel": ["low", "high"],
    "T": [960, 600],          # W/T = 0.5 (working time binds) / 0.8 (spread binds)
    "arc_drop": [0.3, 0.6],   # sparser, non-transitive "given" graph
}
SEEDS = [1, 2, 3]


def suite() -> list[tuple[str, str, dict, int]]:
    """List of (group, name, overrides, seed) defining the benchmark suite."""
    out = []
    for n, seed in itertools.product(SCALE_SIZES, SEEDS):
        out.append(("scale", f"scale_n{n:03d}_s{seed}", {"n": n}, seed))
    for seed in SEEDS:
        out.append(("factor", f"fac_default_s{seed}", {"n": FACTOR_N}, seed))
    for key, values in FACTORS.items():
        for v, seed in itertools.product(values, SEEDS):
            out.append(("factor", f"fac_{key}-{v}_s{seed}", {"n": FACTOR_N, key: v}, seed))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suite", action="store_true", help="generate the full benchmark suite")
    ap.add_argument("--outdir", default=str(ROOT / "data" / "instances"))
    ap.add_argument("--out", help="output file for a single instance")
    ap.add_argument("--seed", type=int, default=1)
    for k, v in DEFAULTS.items():
        ap.add_argument(f"--{k.replace('_', '-')}", type=type(v), default=v)
    a = ap.parse_args()

    if a.suite:
        for group, name, ov, seed in suite():
            inst = generate(seed, name=name, **ov)
            inst.to_json(Path(a.outdir) / group / f"{name}.json")
        print(f"wrote {len(suite())} instances to {a.outdir}")
    else:
        ov = {k: getattr(a, k) for k in DEFAULTS}
        inst = generate(a.seed, **ov)
        out = a.out or str(Path(a.outdir) / f"{inst.name}.json")
        inst.to_json(out)
        print(f"wrote {out}: n={inst.n}, arcs={len(inst.arcs)}")


if __name__ == "__main__":
    main()
