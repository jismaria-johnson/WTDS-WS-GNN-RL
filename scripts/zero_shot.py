"""Zero-shot: the copied wtds checkpoints (no fine-tuning) on the wtds-ws validation set, + RR."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np

from wtdsws.agent import load_model
from wtdsws.config import resolve_device
from wtdsws.validation import evaluate_policy, get_ws_validation_set

dev = resolve_device("auto")
graphs, info = get_ws_validation_set(ROOT / "data/val_ws.npz")
opt = np.asarray(info["opt"], float)
grr = float(np.mean((np.asarray(info["greedy_rr"]) - opt) / opt))
out = {"device": dev, "val_graphs": len(graphs), "val_gap_weighted_greedy_rr": grr, "seeds": {}}
for s in (0, 1):
    ck = ROOT / f"checkpoints/pretrained/wtds_seed{s}_best.pt"
    m = load_model(ck, dev)
    r = evaluate_policy(m, graphs, opt, dev, env_kwargs=m.env_kwargs, batch_size=40)
    out["seeds"][s] = {"checkpoint": str(ck.relative_to(ROOT)).replace("\\", "/"), **r}
    print(f"seed {s} zero-shot: gap raw {r['gap_raw']:.4f}  gap +RR {r['gap_rr']:.4f}  "
          f"opt% {r['frac_opt_rr']:.2f}  ({r['time']:.1f}s on {dev})")
print(f"weighted greedy + RR gap on the same set: {grr:.4f}")
(ROOT / "results").mkdir(exist_ok=True)
(ROOT / "results/zero_shot.json").write_text(json.dumps(out, indent=2))
