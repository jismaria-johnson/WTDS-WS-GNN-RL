"""Revision step 4/7: GNN inference time on the GPU and with CUDA disabled (CPU only), plus training time per seed.

    python scripts/rev_inference_time.py --device cuda
    CUDA_VISIBLE_DEVICES= python scripts/rev_inference_time.py --device cpu     # CUDA disabled for the process

Graphs: main test grid, graphs gidx 0-2 of every class, random scheme, n in {20, 50, 100, 200, 500} (12 per n).
Time = greedy policy rollout + RR per instance (model seed 0, after one warm-up), same code path as the grid.
The CPU and GPU solutions are compared (same W(S) expected up to float ties). Output: results/revision/inference_<dev>.json
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", choices=["cuda", "cpu"], required=True)
    a = ap.parse_args()
    import torch
    if a.device == "cpu":
        assert not torch.cuda.is_available(), "run with CUDA_VISIBLE_DEVICES= so CUDA is disabled"
    from run_eval import checked, structures
    from wtdsws.generators import make_graph
    from wtdsws.postprocess import redundancy_removal
    from wtdsws.rollout import rollout
    from wtdsws.solve import get_model
    from wtdsws.weights import weighted_graph
    model = get_model(ROOT / "checkpoints" / "best.pt", a.device)
    ek = dict(getattr(model, "env_kwargs", {}) or {})
    items = [it for it in structures("grid", 10, (20, 50, 100, 200, 500)) if it[2] <= 2]
    warm = weighted_graph(make_graph(items[0][3]), "random", items[0][4])
    rollout(model, [warm], a.device, env_kwargs=ek, multi_select="auto")
    res = {}
    for cls, n, gi, spec, wseed in items:
        g = weighted_graph(make_graph(spec), "random", wseed)
        if a.device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        r = rollout(model, [g], a.device, env_kwargs=ek, multi_select="auto")[0]
        S = redundancy_removal(g, r["solution"], scores=r["q"], protected=r["forced"])
        dt = time.perf_counter() - t0
        S, W = checked(g, S, "ours_rr")
        res.setdefault(str(n), []).append(dict(cls=cls, gidx=gi, time=dt, W=W))
    out = {"device": a.device, "torch_threads": torch.get_num_threads(),
           "cpu": os.environ.get("PROCESSOR_IDENTIFIER", ""),
           "gpu": torch.cuda.get_device_name(0) if a.device == "cuda" else None,
           "per_n": {n: dict(mean_s=float(np.mean([x["time"] for x in v])), count=len(v), rows=v)
                     for n, v in res.items()}}
    p = ROOT / "results" / "revision" / f"inference_{a.device}.json"
    p.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print({n: round(v["mean_s"], 4) for n, v in out["per_n"].items()})


if __name__ == "__main__":
    main()
