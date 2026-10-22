"""Build the 160-graph validation set (10 per class x scheme, n = 50-100) with proven ILP optima."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wtdsws.seeds import check_disjoint
from wtdsws.validation import get_ws_validation_set

print("seed-range check:")
check_disjoint(verbose=True)
g, info = get_ws_validation_set("data/val_ws.npz", per_cell=10, n_min=50, n_max=100, ilp_time=30.0, threads=8)
import numpy as np
print(f"{len(g)} graphs, all proven optimal; replaced (not proven in 30 s): {info['replaced_not_proven']}; "
      f"ILP time mean {np.mean(info['ilp_runtime']):.2f}s max {np.max(info['ilp_runtime']):.2f}s")
