"""Build the submission archive: the final report and all support files.

    python make_archive.py            -> dist/MDDM_Project2_Group1.zip

Contents: every file tracked by git or not ignored by .gitignore (so no
.venv, caches or LaTeX build files), plus a copy of the report at the top
level of the archive.
"""
from __future__ import annotations

import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NAME = "MDDM_Project2_Group1"
SKIP = {".gitignore"}


def main() -> None:
    files = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard"],
                           cwd=ROOT, capture_output=True, text=True, check=True).stdout.split("\n")
    files = sorted(f for f in files if f and f not in SKIP and not f.endswith(".gitkeep")
                   and not f.startswith("dist/") and (ROOT / f).is_file())
    pdf = ROOT / "report" / "main.pdf"
    if not pdf.exists():
        raise SystemExit("report/main.pdf missing: compile the report first")
    src = [p for p in (ROOT / "report").rglob("*") if p.suffix in {".tex", ".bib", ".pdf", ".png"}
           and p != pdf]
    stale = [p for p in src if p.stat().st_mtime > pdf.stat().st_mtime]
    if stale:
        raise SystemExit("report/main.pdf is older than " + ", ".join(
            str(p.relative_to(ROOT)) for p in stale[:3]) + ": recompile with (cd report && latexmk -g -pdf main.tex)")
    out = ROOT / "dist" / f"{NAME}.zip"
    out.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(pdf, f"{NAME}/report.pdf")
        for f in files:
            z.write(ROOT / f, f"{NAME}/{f}")
    print(f"{out.relative_to(ROOT)}: {len(files) + 1} files, {out.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
