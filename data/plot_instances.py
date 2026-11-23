"""Figure for Section 2: (a) duty concurrency over the day, peak vs uniform profile;
(b) the two cost lower bounds as functions of n on the scale suite.

Writes report/figures/instances.pdf.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data.generator import generate  # noqa: E402

BLUE, ORANGE = "#2a78d6", "#eb6834"   # categorical slots 1-2 (validated pair)
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3de"

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "legend.frameon": False, "lines.linewidth": 2,
})


def active_profile(inst, grid):
    s, e = np.array(inst.start), np.array(inst.end)
    return [int(((s <= t) & (t < e)).sum()) for t in grid]


def main() -> None:
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 2.7), constrained_layout=True)

    # (a) concurrency over the day, n = 100, averaged over 3 seeds
    grid = np.arange(300, 1441, 5)
    for prof, col, ls in [("peak", BLUE, "-"), ("uniform", ORANGE, "--")]:
        prof_avg = np.mean([active_profile(generate(s, n=100, profile=prof), grid) for s in (1, 2, 3)], axis=0)
        a.plot(grid / 60, prof_avg, color=col, ls=ls, label=prof)
    a.set_xlabel("time of day [h]")
    a.set_ylabel("duties running")
    a.set_xticks(range(6, 25, 3))
    a.set_ylim(bottom=0)
    a.set_ylim(0, a.get_ylim()[1] * 1.1)
    a.legend(loc="upper center", ncol=2)
    a.set_title("(a) concurrency, $n=100$", fontsize=9, color=INK, loc="left")

    # (b) lower bounds vs n
    df = pd.read_csv(ROOT / "results" / "instances_summary.csv")
    sc = df[df.group == "scale"].groupby("n")[["lb_path", "lb_work"]].mean()
    b.plot(sc.index, sc.lb_path, color=BLUE, ls="-", marker="o", ms=4, label=r"$0.6\,\rho(G)$")
    b.plot(sc.index, sc.lb_work, color=ORANGE, ls="--", marker="s", ms=4, label=r"$\Sigma_i\, d_i/W$")
    b.set_xlabel("$n$")
    b.set_ylabel("lower bound on cost")
    b.set_ylim(bottom=0)
    b.legend(loc="upper left")
    b.set_title("(b) cost lower bounds, scale suite", fontsize=9, color=INK, loc="left")

    out = ROOT / "report" / "figures" / "instances.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
