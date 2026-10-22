"""Default generator densities of the app (imported by tests/test_app.py)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))


def dens(sub):
    from components.graph_input import DENSITY
    lab, lo, hi, d, step, is_int = DENSITY[sub]
    return int(d) if is_int else float(d)
