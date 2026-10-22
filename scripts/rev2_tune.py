"""Revision 2, step 1c: tune guided restarts and both controls on the VALIDATION set only.

    python scripts/rev2_tune.py --worker 0/4      # (run 4 workers, configs are split round-robin)
    python scripts/rev2_tune.py --choose          # merge -> checkpoints/guided_restarts.json

Graphs: data/val_ws.npz (validation seed range, proven optima), the first 3 graphs of each class x scheme cell = 48.
Grids (each config = one anytime run up to 1 s per graph, objective = mean gap at 1 s; gap at 0.16 s also saved):
  guided        beta {0.5, 1, 2, 4} x tau {0.05, 0.1, 0.2} x refresh {never, 10% of n}       24 configs
  random_prior  the same 24 configs (refresh = redraw the random ranks every 10% of n picks)   24 configs
  plain         tau {0.02, 0.05, 0.1, 0.2, 0.3, 0.5} (beta = 0, no refresh)                    6 configs
Ties on the objective -> the first config in grid order. Model: checkpoints/best.pt (seed 0), CPU, 1 torch thread per
worker (all configs run under the same conditions).
"""
import argparse
import csv
import itertools
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402

OUT = ROOT / "results" / "revision2" / "tuning"
BUDGET = 1.0


def configs():
    out = []
    for prior in ("gnn", "random"):
        for beta, tau, ref in itertools.product((0.5, 1.0, 2.0, 4.0), (0.05, 0.1, 0.2), (0.0, 0.1)):
            out.append(dict(prior=prior, beta=beta, tau=tau, refresh=ref))
    for tau in (0.02, 0.05, 0.1, 0.2, 0.3, 0.5):
        out.append(dict(prior="none", beta=0.0, tau=tau, refresh=0.0))
    return out


def val_graphs():
    from wtdsws.io import load_graphs
    graphs = load_graphs(ROOT / "data" / "val_ws.npz")
    info = json.loads((ROOT / "data" / "val_ws.json").read_text())
    seen, out = {}, []
    for i, (g, c, s, o) in enumerate(zip(graphs, info["cls"], info["scheme"], info["opt"])):
        k = seen.get((c, s), 0)
        seen[(c, s)] = k + 1
        if k < 3:
            out.append((i, c, s, g, float(o)))
    assert len(out) == 48 and all(v == 10 for v in seen.values())
    return out


def worker(wi, wk):
    import torch
    torch.set_num_threads(1)
    import rev2_anytime as A
    from wtdsws.seeds import split_of
    from wtdsws.solve import get_model
    model = get_model(ROOT / "checkpoints" / "best.pt", "cpu")
    items = val_graphs()
    assert all(split_of(int(it[3].meta["seed"])) == "val" and split_of(int(it[3].meta["weight_seed"])) == "val"
               for it in items)
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"tune_w{wi}.csv"
    fh = open(out, "w", newline="")
    wr = csv.DictWriter(fh, fieldnames=["prior", "beta", "tau", "refresh", "vidx", "cls", "scheme", "opt",
                                        "w_016", "w_1", "restarts_trace_len"])
    wr.writeheader()
    A.restarts_run(items[0][3], "gnn", model, "cpu", 0.05, dict(beta=1.0, tau=0.1), 0)     # warm-up
    t0 = time.time()
    for ci, cfg in enumerate(configs()):
        if ci % wk != wi:
            continue
        for vi, c, s, g, opt in items:
            tr, _ = A.restarts_run(g, cfg["prior"], model, "cpu", BUDGET, cfg, seed=vi)
            tr = A.verify_trace(g, tr, f"tune {cfg}")
            assert tr[-1][1] >= opt - 1e-6, "below a proven optimum"
            wr.writerow({**cfg, "vidx": vi, "cls": c, "scheme": s, "opt": opt, "w_016": A.at_budget(tr, 0.16)[0],
                         "w_1": A.at_budget(tr, BUDGET)[0], "restarts_trace_len": len(tr)})
        fh.flush()
        print(f"[w{wi}] config {ci} {cfg} done ({time.time() - t0:.0f}s)", flush=True)
    fh.close()


def choose():
    import pandas as pd
    d = pd.concat([pd.read_csv(p) for p in OUT.glob("tune_w*.csv")], ignore_index=True)
    d["gap_1"] = 100 * (d.w_1 - d.opt) / d.opt
    d["gap_016"] = 100 * (d.w_016 - d.opt) / d.opt
    cfgs = configs()
    assert len(d) == 48 * len(cfgs), (len(d), 48 * len(cfgs))
    rows = []
    for ci, c in enumerate(cfgs):
        m = d[(d.prior == c["prior"]) & np.isclose(d.beta, c["beta"]) & np.isclose(d.tau, c["tau"])
              & np.isclose(d.refresh, c["refresh"])]
        assert len(m) == 48
        rows.append({**c, "order": ci, "mean_gap_1s": float(m.gap_1.mean()), "mean_gap_016s": float(m.gap_016.mean()),
                     "mean_gap_1s_degree": float(m[m.scheme == "degree"].gap_1.mean())})
    tab = pd.DataFrame(rows)
    tab.to_csv(OUT / "tuning_summary.csv", index=False)
    chosen = {}
    for prior, name in (("gnn", "ours_guided_restarts"), ("random", "random_prior_restarts"), ("none", "plain_restarts")):
        t = tab[tab.prior == prior].sort_values(["mean_gap_1s", "order"], kind="stable")
        r = t.iloc[0]
        chosen[name] = dict(beta=float(r.beta), tau=float(r.tau), refresh=float(r.refresh),
                            val_mean_gap_1s=float(r.mean_gap_1s), val_mean_gap_016s=float(r.mean_gap_016s))
        print(name, chosen[name])
    meta = dict(objective="mean gap (%) at 1 s on 48 validation graphs (data/val_ws.npz, first 3 per class x scheme)",
                checkpoint="checkpoints/best.pt", device="cpu, 1 torch thread per tuning worker", ls_time_cap=1.0,
                grid={"guided": "beta {0.5,1,2,4} x tau {0.05,0.1,0.2} x refresh {0, 0.1}",
                      "random_prior": "same 24 configs", "plain": "tau {0.02,0.05,0.1,0.2,0.3,0.5}"},
                summary="results/revision2/tuning/tuning_summary.csv", script="scripts/rev2_tune.py")
    (ROOT / "checkpoints" / "guided_restarts.json").write_text(json.dumps({**chosen, "_meta": meta}, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", default=None)
    ap.add_argument("--choose", action="store_true")
    a = ap.parse_args()
    if a.choose:
        choose()
    else:
        wi, wk = map(int, a.worker.split("/"))
        worker(wi, wk)
