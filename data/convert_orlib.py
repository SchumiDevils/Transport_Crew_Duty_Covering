"""Convert the OR-Library crew scheduling files (Beasley & Cao, 1996) to our JSON format.

Source: http://people.brunel.ac.uk/~mastjjb/jeb/orlib/cspinfo.html
File format: "N time_limit", then N lines "start finish", then lines "i j cost"
(1-based) listing the transition arcs, i.e. the compatibility graph.

Adaptation to Project n. 2:
  * the compatibility graph is taken as given (the arcs of the file);
    transition costs are ignored (our objective has no arc costs);
  * the file's time limit (480) is used as the FT max working time W;
  * the files have no total-time limit: we set T = 720 (W/T = 0.67, as our
    default generated instances).
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.instance import Instance  # noqa: E402

T_ORLIB = 720


def read_orlib(path: Path) -> Instance:
    tok = path.read_text().split()
    n, limit = int(tok[0]), int(tok[1])
    start = [int(tok[2 + 2 * i]) for i in range(n)]
    end = [int(tok[3 + 2 * i]) for i in range(n)]
    rest = tok[2 + 2 * n:]
    arcs = sorted({(int(rest[k]) - 1, int(rest[k + 1]) - 1) for k in range(0, len(rest), 3)})
    inst = Instance(
        name=path.stem, W=limit, T=T_ORLIB, start=start, end=end, arcs=arcs,
        source="orlib", params={"file": path.name, "time_limit": limit, "T": T_ORLIB},
    )
    inst.check()
    return inst


def main() -> None:
    src = ROOT / "data" / "orlib"
    dst = ROOT / "data" / "instances" / "orlib"
    files = sorted(src.glob("csp*.txt"), key=lambda p: int(p.stem[3:]))
    for f in files:
        read_orlib(f).to_json(dst / f"{f.stem}.json")
    print(f"converted {len(files)} files to {dst}")


if __name__ == "__main__":
    main()
