"""Compute structural statistics of all instances.

Writes results/instances_summary.csv (one row per instance) and
report/tables/instances.tex (aggregated table for Section 2).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.instance import Instance, summary  # noqa: E402


def main() -> None:
    rows = []
    for f in sorted((ROOT / "data" / "instances").rglob("*.json")):
        s = summary(Instance.from_json(f))
        s["group"] = f.parent.name
        rows.append(s)
    df = pd.DataFrame(rows)
    out = ROOT / "results" / "instances_summary.csv"
    df.to_csv(out, index=False, float_format="%.4f")
    print(f"wrote {out} ({len(df)} instances)")

    # ---- LaTeX tables (averages over the 3 seeds) --------------------------
    tab_dir = ROOT / "report" / "tables"
    tab_dir.mkdir(parents=True, exist_ok=True)
    cols = ["n", "arcs", "density", "eff_density", "d_avg", "ht_eligible", "max_conc", "rho",
            "lb_path", "lb_work"]
    fmt = {"n": "{:.0f}", "arcs": "{:.0f}", "density": "{:.2f}", "eff_density": "{:.2f}",
           "d_avg": "{:.0f}", "ht_eligible": "{:.2f}", "max_conc": "{:.1f}", "rho": "{:.1f}",
           "lb_path": "{:.1f}", "lb_work": "{:.1f}"}
    head = (r"\begin{tabular}{l" + "r" * len(cols) + "}\n\\toprule\n"
            r"Class & $n$ & $|\A|$ & dens. & eff.\ dens. & $\bar d$ & HT-elig. & conc. & $\rho(G)$"
            r" & $0.6\rho$ & $\sum d/W$ \\" "\n\\midrule\n")

    def row(label: str, sub: pd.DataFrame) -> str:
        m = sub[cols].mean()
        return label + " & " + " & ".join(fmt[c].format(m[c]) for c in cols) + r" \\" + "\n"

    sc = df[df.group == "scale"].copy()
    body = "".join(row("default", sc[sc.n == n]) for n in sorted(sc.n.unique()))
    (tab_dir / "inst_scale.tex").write_text(head.replace("Class", "Params") + body + "\\bottomrule\n\\end{tabular}\n")

    fa = df[df.group == "factor"].copy()
    fa["cls"] = fa.name.str.replace(r"_s\d+$", "", regex=True).str.replace("fac_", "", regex=False)
    order = ["default", "profile-uniform", "durations-short", "durations-long", "travel-low",
             "travel-high", "T-960", "T-600", "arc_drop-0.3", "arc_drop-0.6"]
    body = "".join(row(c.replace("_", r"\_"), fa[fa.cls == c]) for c in order if c in set(fa.cls))
    (tab_dir / "inst_factor.tex").write_text(head.replace("Class", "Variant") + body + "\\bottomrule\n\\end{tabular}\n")

    orl = df[df.group == "orlib"].sort_values("n")
    body = "".join(row(r.name, orl[orl.name == r.name]) for r in orl.itertuples())
    (tab_dir / "inst_orlib.tex").write_text(head.replace("Class", "File") + body + "\\bottomrule\n\\end{tabular}\n")
    print(f"wrote LaTeX tables to {tab_dir}")


if __name__ == "__main__":
    main()
