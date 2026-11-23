"""Independent solution validator.

Checks a solution against the instance data only (it does not use any model):
  1. every duty is assigned to exactly one employee;
  2. the duties of each employee, in the given order, form a path in G;
  3. working time  <= W  (FT) or W/2 (HT);
  4. total time    <= T  (FT) or T/2 (HT);
  5. the recomputed cost equals the reported objective.

Solution format (JSON):
  {"instance": name, "objective": float,
   "shifts": [{"type": "FT" | "HT", "duties": [i1, i2, ...]}, ...]}

Usage: python src/validator.py <instance.json> <solution.json>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.instance import COST_FT, COST_HT, Instance  # noqa: E402


def validate(inst: Instance, sol: dict, tol: float = 1e-6) -> list[str]:
    """Return the list of violations (empty list = valid solution)."""
    errors: list[str] = []
    arcs = set(inst.arcs)
    seen = [0] * inst.n
    cost = 0.0
    for k, sh in enumerate(sol["shifts"]):
        typ, duties = sh["type"], sh["duties"]
        if typ not in ("FT", "HT"):
            errors.append(f"shift {k}: unknown type {typ}")
            continue
        if not duties:
            errors.append(f"shift {k}: empty")
            continue
        W = inst.W if typ == "FT" else inst.W // 2
        T = inst.T if typ == "FT" else inst.T // 2
        cost += COST_FT if typ == "FT" else COST_HT
        for i in duties:
            if not 0 <= i < inst.n:
                errors.append(f"shift {k}: unknown duty {i}")
            else:
                seen[i] += 1
        for a, b in zip(duties, duties[1:]):
            if (a, b) not in arcs:
                errors.append(f"shift {k}: ({a},{b}) is not an arc of G")
        work = sum(inst.end[i] - inst.start[i] for i in duties)
        spread = inst.end[duties[-1]] - inst.start[duties[0]]
        if work > W:
            errors.append(f"shift {k} ({typ}): working time {work} > {W}")
        if spread > T:
            errors.append(f"shift {k} ({typ}): total time {spread} > {T}")
    for i, c in enumerate(seen):
        if c != 1:
            errors.append(f"duty {i} covered {c} times")
    if abs(cost - sol["objective"]) > tol:
        errors.append(f"cost mismatch: recomputed {cost:.4f}, reported {sol['objective']:.4f}")
    return errors


def main() -> None:
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(2)
    inst = Instance.from_json(sys.argv[1])
    sol = json.loads(Path(sys.argv[2]).read_text())
    errs = validate(inst, sol)
    if errs:
        print("INVALID")
        for e in errs:
            print("  -", e)
        sys.exit(1)
    print(f"VALID: cost {sol['objective']:.2f}, {len(sol['shifts'])} employees")


if __name__ == "__main__":
    main()
