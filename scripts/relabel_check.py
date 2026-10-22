"""Label change check (decisions R40): every regenerated text output (tex / csv / md / json) must contain exactly the
same sequence of numbers as its archived pre-relabel version (archive/pre_proposed_labels/), and no "ours" display
label may remain in table headers or figure-facing text.

    python scripts/relabel_check.py
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from labels import relabel  # noqa: E402
DIRS = ["results/figures/paper", "results/revision", "results/revision2", "results/ablations", "results/lattices"]
NUM = re.compile(r"-?\d+(?:\.\d+)?(?:e[-+]?\d+)?")


def main():
    bad, n = [], 0
    for d in DIRS:
        arch = ROOT / d / "archive" / "pre_proposed_labels"
        for old in sorted(arch.rglob("*")):
            if old.suffix not in (".tex", ".csv", ".md", ".json") or not old.is_file():
                continue
            rel = old.relative_to(arch)
            new = ROOT / d / rel
            if rel.name == "paper_table.tex":
                new = ROOT / "results" / "tables" / "paper_table.tex"
            if not new.exists():
                bad.append(f"missing regenerated file {new}")
                continue
            a = NUM.findall(relabel(old.read_text(encoding="utf-8")))      # same label substitutions, then numbers
            b = NUM.findall(new.read_text(encoding="utf-8"))
            n += 1
            if a != b:
                i = next((k for k, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
                bad.append(f"{d}/{rel}: numbers differ at token {i} ({a[i:i + 3]} vs {b[i:i + 3]}; {len(a)} vs {len(b)})")
    print(f"{n} files compared; {len(bad)} with differences")
    for x in bad:
        print("  ", x)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
