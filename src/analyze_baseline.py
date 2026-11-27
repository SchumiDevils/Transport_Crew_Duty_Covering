"""Tables and figures of Section 4 from results/raw/baseline.csv.

Writes report/tables/base_*.tex and report/figures/base_*.pdf, and prints a
textual summary used to write the bottleneck analysis.
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
TAB = ROOT / "report" / "tables"
FIG = ROOT / "report" / "figures"
MODELS = ["AF", "CF", "CFM"]
LABEL = {"AF": "(AF)", "CF": "(CF)", "CFM": "(CF-M)"}
# categorical slots 1-3 of the validated palette + distinct markers/lines
STYLE = {"AF": ("#2a78d6", "o", "-"), "CF": ("#eb6834", "s", "--"), "CFM": ("#1baf7a", "^", ":")}
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3de"
plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "legend.frameon": False, "lines.linewidth": 2,
})


def load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["solved"] = df.status == "optimal"
    df["cls"] = df.instance.str.replace(r"_s\d+$", "", regex=True)
    # time used: runtime if run, else time limit (too large / error count as failures)
    df["time"] = np.where(df.solved, df.runtime_sec, df.time_limit)
    # reference optimum: best proven optimum of an *exact* model on that instance
    exact = df.solved & ((df.model == "AF") | (df.transitive == True))  # noqa: E712
    zopt = df[exact].groupby("instance")["obj"].min()
    df["zopt"] = df.instance.map(zopt)
    # best known feasible value of an exact model, and best known bound
    feas = df[(df.model == "AF") | (df.transitive == True)]  # noqa: E712
    df["zbest"] = df.instance.map(feas.groupby("instance")["obj"].min())
    df["zbest"] = np.fmin(df.zbest, df.greedy_ub)
    for c in ("lp", "root"):
        col = "lp_bound" if c == "lp" else "root_bound"
        # bound 0 = node/LP not finished: no bound, excluded from the means
        df[f"{c}_gap"] = (df.zopt - df[col].where(df[col] > 0)) / df.zopt
    df["final_gap"] = np.where(df.solved, 0.0, (df.obj - df.bound) / df.obj)
    df["vs_greedy"] = (df.obj - df.greedy_ub) / df.greedy_ub
    return df


def fmt(x, f="{:.1f}", na="--"):
    return na if pd.isna(x) else f.format(x)


def table_scale(df: pd.DataFrame) -> str:
    sc = df[df.group == "scale"]
    lines = [r"\begin{tabular}{r" + "rrr" * len(MODELS) + "}", r"\toprule",
             " & " + " & ".join(rf"\multicolumn{{3}}{{c}}{{{LABEL[m]}}}" for m in MODELS) + r" \\",
             " ".join(rf"\cmidrule(lr){{{2 + 3 * k}-{4 + 3 * k}}}" for k in range(len(MODELS))),
             "$n$ & " + " & ".join(["solved & time & gap"] * len(MODELS)) + r" \\", r"\midrule"]
    for n in sorted(sc.n.dropna().unique()):
        cells = []
        for m in MODELS:
            s = sc[(sc.n == n) & (sc.model == m)]
            if s.empty or (s.status == "too_large").all():
                cells += [r"\multicolumn{3}{c}{too large}"]
                continue
            cells += [f"{int(s.solved.sum())}/{len(s)}", fmt(s.time.mean(), "{:.0f}"),
                      fmt(100 * s.final_gap.mean(), "{:.1f}")]
        lines.append(f"{int(n)} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def table_factor(df: pd.DataFrame) -> str:
    fa = df[df.group == "factor"].copy()
    fa["var"] = fa.cls.str.replace("fac_", "", regex=False)
    order = ["default", "profile-uniform", "durations-short", "durations-long", "travel-low",
             "travel-high", "T-960", "T-600", "arc_drop-0.3", "arc_drop-0.6"]
    lines = [r"\begin{tabular}{l" + "rrr" * len(MODELS) + "}", r"\toprule",
             " & " + " & ".join(rf"\multicolumn{{3}}{{c}}{{{LABEL[m]}}}" for m in MODELS) + r" \\",
             " ".join(rf"\cmidrule(lr){{{2 + 3 * k}-{4 + 3 * k}}}" for k in range(len(MODELS))),
             "Variant & " + " & ".join(["solved & time & gap"] * len(MODELS)) + r" \\", r"\midrule"]
    for v in order:
        cells = []
        for m in MODELS:
            s = fa[(fa["var"] == v) & (fa.model == m)]
            if s.empty:
                cells += ["--"] * 3
                continue
            cells += [f"{int(s.solved.sum())}/{len(s)}", fmt(s.time.mean(), "{:.0f}"),
                      fmt(100 * s.final_gap.mean(), "{:.1f}")]
        lines.append(v.replace("_", r"\_") + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def table_bounds(df: pd.DataFrame) -> str:
    """LP and root gaps w.r.t. the proven optimum, transitive instances solved by some model."""
    sub = df[df.zopt.notna() & (df.transitive == True)]  # noqa: E712
    lines = [r"\begin{tabular}{lrrrrr}", r"\toprule",
             r"Model & instances & LP gap \% & root gap \% & nodes (solved) & solved \\", r"\midrule"]
    for m in MODELS:
        s = sub[sub.model == m]
        sol = s[s.solved]
        lines.append(f"{LABEL[m]} & {len(s)} & {fmt(100 * s.lp_gap.mean())} & "
                     f"{fmt(100 * s.root_gap.mean())} & {fmt(sol.nodes.median(), '{:.0f}')} & "
                     f"{int(s.solved.sum())} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def table_orlib(df: pd.DataFrame) -> str:
    o = df[df.group == "orlib"]
    lines = [r"\begin{tabular}{rrrrrrrrrr}", r"\toprule",
             r" & & \multicolumn{4}{c}{(AF), exact} & \multicolumn{4}{c}{(CF), restriction} \\",
             r"\cmidrule(lr){3-6}\cmidrule(lr){7-10}",
             r"$n$ & LB & greedy & obj & bound & status & greedy$^{\mathrm{pw}}$ & obj & bound & status \\",
             r"\midrule"]

    def st(r):
        return {"optimal": "opt", "time_limit": "TL", "too_large": "large"}.get(r.status, r.status)

    def bd(r):  # bound 0 = root LP not finished: no dual bound
        return "none" if r["bound"] == 0 else fmt(r["bound"])

    for n in sorted(o.n.dropna().unique()):
        a = o[(o.n == n) & (o.model == "AF")].iloc[0]
        c = o[(o.n == n) & (o.model == "CF")].iloc[0]
        lines.append(" & ".join([f"{int(n)}", fmt(a.lb), fmt(a.greedy_ub), fmt(a["obj"]), bd(a), st(a),
                                 fmt(c.greedy_ub), fmt(c["obj"]), bd(c), st(c)]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def figure_scale(df: pd.DataFrame) -> None:
    sc = df[(df.group == "scale") & (df.status != "too_large")]
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 2.8), constrained_layout=True)
    for m in MODELS:
        s = sc[sc.model == m].groupby("n")
        col, mk, ls = STYLE[m]
        t = s.time.median()
        a.plot(t.index, t.values, color=col, marker=mk, ls=ls, ms=4, label=LABEL[m])
        g = s.final_gap.mean() * 100
        b.plot(g.index, g.values, color=col, marker=mk, ls=ls, ms=4, label=LABEL[m])
    tl = float(df.time_limit.max())
    a.axhline(tl, color=MUTED, lw=1, ls=(0, (2, 2)))
    a.text(sc.n.min(), tl * 1.15, "time limit", color=MUTED, fontsize=8)
    a.set_yscale("log")
    a.set_xlabel("$n$")
    a.set_ylabel("time [s] (median of 3 seeds)")
    a.set_title("(a) running time", fontsize=9, color=INK, loc="left")
    a.legend(loc="lower right")
    b.set_xlabel("$n$")
    b.set_ylabel("final gap [%] (mean)")
    b.set_title("(b) gap at the time limit", fontsize=9, color=INK, loc="left")
    b.set_ylim(bottom=0)
    b.legend(loc="upper left")
    fig.savefig(FIG / "base_scale.pdf", metadata={"CreationDate": None})
    fig.savefig(FIG / "base_scale.png", dpi=150)
    plt.close(fig)


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results" / "raw" / "baseline.csv"
    df = load(path)
    TAB.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    (TAB / "base_scale.tex").write_text(table_scale(df))
    (TAB / "base_factor.tex").write_text(table_factor(df))
    (TAB / "base_bounds.tex").write_text(table_bounds(df))
    (TAB / "base_orlib.tex").write_text(table_orlib(df))
    figure_scale(df)

    # --------------------------------------------------------- text summary
    print(f"runs: {len(df)}; status counts:\n{df.groupby(['model', 'status']).size().unstack(fill_value=0)}")
    print("invalid solutions:", int((df.valid == False).sum()))  # noqa: E712
    ok = df[df.obj.notna()]
    print("\nobjective worse than greedy (share of runs with a solution):")
    print(ok.groupby("model").apply(lambda s: (s.obj > s.greedy_ub + 1e-6).mean()).round(2))
    sym = df[df.solved & (df.nodes > 0)]
    print("\nnodes of solved runs (median / max):")
    print(sym.groupby("model").nodes.agg(["median", "max"]))
    print("\nLP / root bound gaps vs optimum, transitive solved instances:")
    sub = df[df.zopt.notna() & (df.transitive == True)]  # noqa: E712
    print(sub.groupby("model")[["lp_gap", "root_gap"]].mean().round(3))
    print("\nLP solve time (mean, s):")
    print(df.groupby("model").lp_sec.mean().round(1))
    print("\nlargest n solved per model (scale group, all 3 seeds):")
    sc = df[df.group == "scale"].groupby(["model", "n"]).solved.all()
    print(sc[sc].reset_index().groupby("model").n.max())


if __name__ == "__main__":
    main()
