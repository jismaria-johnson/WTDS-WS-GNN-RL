"""Validate the exact special-class algorithms and write results/exact_validation.json.

    python scripts/validate_exact.py

Same check as tests/test_exact_special.py (brute force: 500 instances per algorithm, n <= 12; proven CP-SAT: 300
instances per algorithm, n = 50-200; all 5 weightings). scripts/plot_exact_vs_ours.py refuses to plot unless this
file reports 0 mismatches and 0 unproven ILP references.
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from wtdsws.exact_validation import validate  # noqa: E402

if __name__ == "__main__":
    t0 = time.time()
    rep = validate(n_bf=500, n_ilp=300)
    mism = sum(v[k][m] for v in rep.values() for k in v for m in ("weight_mismatches", "size_mismatches"))
    rep["total_instances"] = sum(v[k]["instances"] for v in rep.values() for k in v)
    rep["total_mismatches"] = mism
    rep["ilp_not_proven"] = sum(v["ilp"]["not_proven"] for k, v in rep.items() if isinstance(v, dict))
    rep["seconds"] = round(time.time() - t0, 1)
    out = ROOT / "results" / "exact_validation.json"
    out.write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))
    sys.exit(1 if mism or rep["ilp_not_proven"] else 0)
