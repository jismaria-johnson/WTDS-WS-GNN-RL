"""Ship the fine-tuned checkpoints and write checkpoints/model_info.json.

For each seed: best.pt = lowest validation gap after RR over all validation points INCLUDING step 0 (= the copied
wtds weights), so if fine-tuning never beats zero-shot the zero-shot weights are what ships (reported as such).
Every number written here is re-measured now on the 160-graph validation set, not copied from logs.
"""
import csv
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch

from wtdsws.agent import load_model
from wtdsws.config import resolve_device
from wtdsws.validation import evaluate_policy, get_ws_validation_set

dev = resolve_device("auto")
graphs, info = get_ws_validation_set(ROOT / "data/val_ws.npz")
opt = np.asarray(info["opt"], float)
grr = float(np.mean((np.asarray(info["greedy_rr"]) - opt) / opt))
SHIP = {0: ROOT / "checkpoints/best.pt", 1: ROOT / "checkpoints/seed1_best.pt"}


def measure(path):
    m = load_model(path, dev)
    return evaluate_policy(m, graphs, opt, dev, env_kwargs=m.env_kwargs, batch_size=20)


seeds = []
for s in (0, 1):
    run = ROOT / f"runs/ws_seed{s}"
    rows = list(csv.DictReader(open(run / "val.csv", newline="")))
    steps = int(float(rows[-1]["step"]))
    ft = [r for r in rows if int(float(r["step"])) > 0]
    best_ft = min(ft, key=lambda r: float(r["gap_rr"]))
    best_ck = ROOT / f"checkpoints/ws_seed{s}/best.pt"
    ck = torch.load(best_ck, map_location="cpu", weights_only=False)
    zero = measure(ROOT / f"checkpoints/pretrained/wtds_seed{s}_best.pt")
    shipped = measure(best_ck)
    shipped_what = "zero-shot (fine-tuning did not improve the validation gap)" if int(ck["step"]) == 0 \
        else f"fine-tuned, step {int(ck['step'])}"
    shutil.copyfile(best_ck, SHIP[s])
    seeds.append({"seed": s, "checkpoint": str(SHIP[s].relative_to(ROOT)).replace("\\", "/"),
                  "pretrained_from": f"checkpoints/pretrained/wtds_seed{s}_best.pt",
                  "finetune_steps": steps, "best_step": int(ck["step"]), "shipped": shipped_what,
                  "zero_shot_val_gap_rr": zero["gap_rr"], "zero_shot_val_gap_raw": zero["gap_raw"],
                  "zero_shot_val_frac_optimal_rr": zero["frac_opt_rr"],
                  "finetuned_best_val_gap_rr": float(best_ft["gap_rr"]),
                  "finetuned_best_step": int(float(best_ft["step"])),
                  "finetuned_last_val_gap_rr": float(rows[-1]["gap_rr"]),
                  "shipped_val_gap_rr": shipped["gap_rr"], "shipped_val_gap_raw": shipped["gap_raw"],
                  "shipped_val_frac_optimal_rr": shipped["frac_opt_rr"],
                  "wall_s": float(rows[-1]["wall_s"])})
    print(f"seed {s}: steps {steps}; zero-shot gap+RR {zero['gap_rr']:.4f}; best fine-tuned {float(best_ft['gap_rr']):.4f} "
          f"(step {best_ft['step']}); shipped {shipped['gap_rr']:.4f} [{shipped_what}]")
assert seeds[0]["finetune_steps"] == seeds[1]["finetune_steps"], "seeds must stop at the same step count"
cfg = json.loads((ROOT / "runs/ws_seed0/config.json").read_text())
out = {"encoder": "mpnn", "hidden": 64, "layers": 3, "readout": "meanmax",
       "parameters": int(sum(p.numel() for p in load_model(SHIP[0], "cpu").parameters())),
       "reward": "weighted: r = -w(v) / mean weight (unchanged from wtds-gnn-rl)", "device": dev,
       "train": {k: cfg["train"][k] for k in ("lr", "eps_start", "eps_end", "eps_frac", "total_steps", "time_box_s",
                                              "buffer_size", "batch_size", "sampler", "curriculum")},
       "val_gap_weighted_greedy_rr": grr, "seeds": seeds,
       "validation_note": "Validation: 160 fixed graphs (10 per class × scheme; ER, BA, chordal, bipartite × degree, "
                          "inverse, random, mixed; n = 50–100), all optima proven by CP-SAT (30 s). Gap = mean "
                          "(W(S) − W*) / W* after redundancy removal. Both seeds were fine-tuned for the same number "
                          "of steps from the copied wtds-gnn-rl checkpoints; best.pt = lowest gap, step 0 included."}
(ROOT / "checkpoints/model_info.json").write_text(json.dumps(out, indent=2))
print(f"weighted greedy + RR gap on the same set: {grr:.4f}\nwrote checkpoints/model_info.json")
