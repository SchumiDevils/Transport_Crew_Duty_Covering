"""Tables and figures of Section 6 (baseline vs improvements).

Reads results/raw/{baseline,improved,ablation}.csv, writes
report/tables/cmp_*.tex and report/figures/cmp_*.pdf, and prints a summary.

Fairness rules (same instances, machine, time limit, threads for all runs):
  * a run counts as *solved* if it proves optimality for the problem itself:
    (CF)-type models (CF, CF-M, CF+...) only on transitive graphs, where they
    are exact; on non-transitive graphs they solve a restriction;
  * unsolved runs count with the time limit (300 s).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results" / "raw"
TAB = ROOT / "report" / "tables"
FIG = ROOT / "report" / "figures"
MAIN = ["AF", "CF", "CFM", "CF+", "SP"]
ABL = ["CF+", "CF+noclq", "CF+chain", "CF+nosym", "CF+nostart"]
LABEL = {"AF": "(AF)", "CF": "(CF)", "CFM": "(CF-M)", "CF+": "(CF+)", "SP": "(SP-CG)",
         "CF+noclq": "no cliques", "CF+chain": "spread chain", "CF+nosym": "no symmetry br.",
         "CF+nostart": "no improving cut"}
EXACT_ANY = {"AF", "SP"}     # exact on every DAG
# categorical palette (fixed order); baseline models muted lines, new ones bold
STYLE = {"AF": ("#2a78d6", "o", ":"), "CF": ("#eb6834", "s", ":"), "CFM": ("#1baf7a", "^", ":"),
         "CF+": ("#8a56d6", "D", "-"), "SP": ("#d63a5e", "v", "-")}
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3de"
plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "legend.frameon": False, "lines.linewidth": 2,
})
TL = 300.0


def load() -> pd.DataFrame:
    parts = [pd.read_csv(RAW / f) for f in ("baseline.csv", "improved.csv", "ablation.csv")
             if (RAW / f).exists()]
    df = pd.concat(parts, ignore_index=True)
    df["obj"] = df["obj"].round(6)
    df["bound"] = df["bound"].round(6)
    df["exact"] = df.model.isin(EXACT_ANY) | (df.transitive == True)  # noqa: E712
    df["solved"] = (df.status == "optimal") & df.exact
    df["time"] = np.where(df.solved, df.runtime_sec, TL)
    zopt = df[df.solved].groupby("instance")["obj"]
    df["zopt"] = df.instance.map(zopt.min())
    df["zopt_max"] = df.instance.map(zopt.max())
    # best known solution value: every run's solution is feasible for the
    # problem (restrictions included), so take the minimum over all runs
    df["zbest"] = df.instance.map(df.groupby("instance")["obj"].min())
    df["primal_gap"] = ((df.obj - df.zbest) / df.zbest).fillna(1.0)   # no solution -> 100%
    df["final_gap"] = np.where(df.solved, 0.0, (df.obj - df.bound.clip(lower=0)) / df.obj)
    for c in ("lp_bound", "root_bound"):
        b = df[c].where(df[c] > 0)
        df[c.split("_")[0] + "_gap"] = (df.zopt - b) / df.zopt
    return df


def fmt(x, f="{:.1f}", na="--"):
    return na if pd.isna(x) else f.format(x)


# ------------------------------------------------------------------- tables
def table_scale(df) -> str:
    sc = df[df.group == "scale"]
    lines = [r"\begin{tabular}{r" + "rr" * len(MAIN) + "}", r"\toprule",
             " & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{{LABEL[m]}}}" for m in MAIN) + r" \\",
             " ".join(rf"\cmidrule(lr){{{2 + 2 * k}-{3 + 2 * k}}}" for k in range(len(MAIN))),
             "$n$ & " + " & ".join(["solved & time"] * len(MAIN)) + r" \\", r"\midrule"]
    for n in sorted(sc.n.unique()):
        cells = []
        for m in MAIN:
            s = sc[(sc.n == n) & (sc.model == m)]
            if s.empty or (s.status == "too_large").all():
                cells.append(r"\multicolumn{2}{c}{large}")
            else:
                cells += [f"{int(s.solved.sum())}/{len(s)}", fmt(s.time.mean(), "{:.0f}")]
        lines.append(f"{int(n)} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def table_summary(df) -> str:
    """One line per model over all 76 instances."""
    lines = [r"\begin{tabular}{lrrrrr}", r"\toprule",
             r"Model & solved & too large & mean time [s] & shifted geo.\ mean [s] & primal gap \% \\",
             r"\midrule"]
    for m in MAIN:
        s = df[df.model == m]
        sgm = np.exp(np.log(s.time + 1).mean()) - 1
        lines.append(f"{LABEL[m]} & {int(s.solved.sum())}/{len(s)} & {int((s.status == 'too_large').sum())} & "
                     f"{s.time.mean():.0f} & {sgm:.1f} & {fmt(100 * s.primal_gap.mean())} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def table_bounds(df) -> str:
    sub = df[df.zopt.notna() & (df.transitive == True) & df.model.isin(MAIN)]  # noqa: E712
    lines = [r"\begin{tabular}{lrrrr}", r"\toprule",
             r"Model & LP gap \% & root gap \% & nodes (median) & solved \\", r"\midrule"]
    for m in MAIN:
        s = sub[sub.model == m]
        lines.append(f"{LABEL[m]} & {fmt(100 * s.lp_gap.mean())} & {fmt(100 * s.root_gap.mean())} & "
                     f"{fmt(s[s.solved].nodes.median(), '{:.0f}')} & {int(s.solved.sum())}/{len(s)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def table_orlib(df) -> str:
    o = df[df.group == "orlib"]
    lines = [r"\begin{tabular}{rrrrrrrrrr}", r"\toprule",
             r" & & \multicolumn{2}{c}{best baseline} & \multicolumn{6}{c}{(SP-CG)} \\",
             r"\cmidrule(lr){3-4}\cmidrule(lr){5-10}",
             r"$n$ & greedy & obj & bound & $z_{\mathrm{LP}}$ & bound & obj & time [s] & CG [s] & enum.\ cols \\",
             r"\midrule"]
    for n in sorted(o.n.unique()):
        a = o[(o.n == n) & (o.model == "AF")].iloc[0]
        sp = o[(o.n == n) & (o.model == "SP")]
        sp = sp.iloc[0] if len(sp) else None
        ab = "none" if a["bound"] == 0 else fmt(a["bound"])
        cells = [f"{int(n)}", fmt(a.greedy_ub), fmt(a["obj"]), ab if pd.notna(a["obj"]) else "--"]
        if sp is None:
            cells += ["--"] * 6
        else:
            t = fmt(sp.runtime_sec, "{:.0f}") if sp.status == "optimal" else "TL"
            cells += [fmt(sp.lp_bound, "{:.2f}"), fmt(sp["bound"]), fmt(sp["obj"]), t,
                      fmt(sp.cg_sec, "{:.1f}"), fmt(sp.enum_cols, "{:.0f}")]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def table_ablation(df) -> str:
    ins = df[(df.model == "CF+nostart") & (df.group == "scale")].instance.unique()
    sub = df[df.instance.isin(ins)]
    lines = [r"\begin{tabular}{lrrrrrr}", r"\toprule",
             r"Variant & solved & time [s] & root gap \% & nodes & rows & gap \% \\",
             r"\midrule"]
    for m in ["CF"] + ABL:
        s = sub[sub.model == m]
        if s.empty:
            continue
        lab = "(CF) baseline" if m == "CF" else ("(CF+) full" if m == "CF+" else "(CF+) " + LABEL[m])
        lines.append(f"{lab} & {int(s.solved.sum())}/{len(s)} & {s.time.mean():.0f} & "
                     f"{fmt(100 * s.root_gap.mean())} & {fmt(s.nodes.median(), '{:.0f}')} & "
                     f"{fmt(s.n_rows.mean() / 1000, '{:.0f}k')} & {fmt(100 * s.final_gap.mean())} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------ figures
def figure_scale(df) -> None:
    sc = df[(df.group == "scale") & (df.status != "too_large")]
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 2.8), constrained_layout=True)
    for m in MAIN:
        s = sc[sc.model == m].groupby("n")
        col, mk, ls = STYLE[m]
        lw = 2 if m in ("CF+", "SP") else 1.3
        t = s.time.median()
        a.plot(t.index, t.values, color=col, marker=mk, ls=ls, lw=lw, ms=4, label=LABEL[m])
        gp = s.final_gap.mean() * 100
        b.plot(gp.index, gp.values, color=col, marker=mk, ls=ls, lw=lw, ms=4, label=LABEL[m])
    a.axhline(TL, color=MUTED, lw=1, ls=(0, (2, 2)))
    a.text(sc.n.min(), TL * 1.2, "time limit", color=MUTED, fontsize=8)
    a.set_yscale("log")
    a.set_xlabel("$n$")
    a.set_ylabel("time [s] (median of 3 seeds)")
    a.set_title("(a) running time", fontsize=9, color=INK, loc="left")
    b.set_xlabel("$n$")
    b.set_ylabel("final gap [%] (mean)")
    b.set_title("(b) gap at the time limit", fontsize=9, color=INK, loc="left")
    b.set_ylim(bottom=0)
    h, lab = a.get_legend_handles_labels()
    fig.legend(h, lab, loc="outside upper center", ncol=5)
    fig.savefig(FIG / "cmp_scale.pdf", metadata={"CreationDate": None})
    fig.savefig(FIG / "cmp_scale.png", dpi=150)
    plt.close(fig)


def figure_profile(df) -> None:
    """Dolan-More performance profile of the running time over all instances."""
    d = df[df.model.isin(MAIN)].pivot_table(index="instance", columns="model", values="time",
                                            aggfunc="first")
    sv = df[df.model.isin(MAIN)].pivot_table(index="instance", columns="model", values="solved",
                                             aggfunc="first").fillna(False).astype(bool)
    t = d.where(sv)                          # unsolved -> NaN (ratio = infinity)
    best = t.min(axis=1)
    keep = best.notna()
    r = t[keep].div(best[keep] .clip(lower=0.01), axis=0)
    fig, ax = plt.subplots(figsize=(4.6, 2.9), constrained_layout=True)
    taus = np.logspace(0, 4.5, 400)
    for m in MAIN:
        col, mk, ls = STYLE[m]
        rho = [(r[m] <= tau).mean() for tau in taus]
        ax.plot(taus, rho, color=col, ls=ls, lw=2 if m in ("CF+", "SP") else 1.3, label=LABEL[m])
    ax.set_xscale("log")
    ax.set_ylim(0, 1.02)
    ax.set_xlabel(r"$\tau$ (time / best time)")
    ax.set_ylabel(r"share of instances solved within $\tau$")
    ax.legend(loc="center left", bbox_to_anchor=(0.0, 0.62))
    fig.savefig(FIG / "cmp_profile.pdf", metadata={"CreationDate": None})
    fig.savefig(FIG / "cmp_profile.png", dpi=150)
    plt.close(fig)
    return int(keep.sum())


def main() -> None:
    df = load()
    TAB.mkdir(parents=True, exist_ok=True)
    (TAB / "cmp_scale.tex").write_text(table_scale(df))
    (TAB / "cmp_summary.tex").write_text(table_summary(df))
    (TAB / "cmp_bounds.tex").write_text(table_bounds(df))
    (TAB / "cmp_orlib.tex").write_text(table_orlib(df))
    (TAB / "cmp_ablation.tex").write_text(table_ablation(df))
    figure_scale(df)
    k = figure_profile(df)

    print("runs per model:\n", df.groupby(["model", "status"]).size().unstack(fill_value=0))
    print("invalid solutions:", int((df.valid == False).sum()))  # noqa: E712
    print(df[df.valid == False][["instance", "model", "errors"]].to_string())  # noqa: E712
    bad = df[(df.zopt_max - df.zopt) > 1e-6].drop_duplicates("instance")
    print("instances where proven optima DISAGREE:", len(bad), list(bad.instance))
    print("profile instances (solved by some method):", k, "of", df.instance.nunique())
    sc = df[df.group == "scale"].groupby(["model", "n"]).solved.all()
    print("largest n solved (all seeds):\n", sc[sc].reset_index().groupby("model").n.max())
    sp = df[df.model == "SP"]
    print("SP: solved", int(sp.solved.sum()), "of", len(sp), "| median time", sp.runtime_sec.median().round(2),
          "| max time", sp.runtime_sec.max().round(1))
    print("SP enumeration used on", int(sp.enum_cols.notna().sum()), "instances; LP integral (zLP = opt) on",
          int((abs(sp.lp_bound - sp.zopt) < 1e-6).sum()))
    cp = df[df.model == "CF+"]
    print("CF+ improving-cut: runs ending at greedy value", int((abs(cp.obj - cp.greedy_ub) < 1e-6).sum()),
          "of", int(cp.obj.notna().sum()))
    # restriction loss on non-transitive instances: CF-type optimum vs SP optimum
    nt = df[(df.transitive == False) & (df.status == "optimal")]  # noqa: E712
    pv = nt.pivot_table(index="instance", columns="model", values="obj", aggfunc="first")
    if "SP" in pv:
        print("non-transitive: restriction loss vs SP optimum\n",
              pv.apply(lambda c: c - pv["SP"]).round(2).to_string())


if __name__ == "__main__":
    main()
