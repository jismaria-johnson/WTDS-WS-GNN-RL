"""Revision: new rows on the SAME main test grid as results/raw_grid.csv (n = 20, 50, 100, 200; the n with ILP optima).

    python scripts/rev_eval_grid.py --part baselines         # stronger greedy variants (CPU)
    python scripts/rev_eval_grid.py --part seeds --seeds 2   # ours + RR / ours + RR + LS for new model seeds (GPU)

Instances come from run_eval.structures (same test seeds); each rebuilt instance is checked against the stored row
(same m, same greedy + RR weight). Every solution is checked with both TDS checkers and W(S) recomputed (run_eval.checked).
Output: results/revision/raw_grid_rev_<part>.csv (run_eval.COLS); existing rows are skipped, so runs can be resumed.
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import pandas as pd  # noqa: E402

import run_eval  # noqa: E402
from run_eval import COLS, checked, structures  # noqa: E402
from wtdsws.baselines import greedy_rr  # noqa: E402
from wtdsws.baselines.greedy_plus import (greedy_lookahead, greedy_random_restart, greedy_tiebreak,  # noqa: E402
                                          lp_rounding)
from wtdsws.generators import make_graph  # noqa: E402
from wtdsws.weights import SCHEMES, weighted_graph  # noqa: E402

NS = (20, 50, 100, 200)
OUT = ROOT / "results" / "revision" / "raw_grid_rev.csv"


def ckpt(seed):
    return {0: "checkpoints/best.pt", 1: "checkpoints/seed1_best.pt"}.get(seed, f"checkpoints/ws_seed{seed}/best.pt")


BASELINES = {
    "greedy_tiebreak_rr": lambda g: greedy_tiebreak(g),
    "greedy_tiebreak_rr_ls": lambda g: greedy_tiebreak(g, ls_time=1.0),
    "greedy_lookahead_rr": lambda g: greedy_lookahead(g),
    "greedy_lookahead_rr_ls": lambda g: greedy_lookahead(g, ls_time=1.0),
    "lp_rounding_rr": lambda g: lp_rounding(g),
    "lp_rounding_rr_ls": lambda g: lp_rounding(g, ls_time=1.0),
    "greedy_random_restart_rr": lambda g: greedy_random_restart(g, budget=0.16, seed=0),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["baselines", "seeds"], required=True)
    ap.add_argument("--seeds", default="2,3,4")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = Path(a.out or str(OUT).replace(".csv", f"_{a.part}.csv"))
    out.parent.mkdir(parents=True, exist_ok=True)
    ref = pd.read_csv(ROOT / "results" / "raw_grid.csv")
    ref = ref[ref.method == "greedy_rr"].set_index(["cls", "n", "gidx", "scheme"])
    done = set()
    if out.exists():
        with open(out, newline="") as f:
            done = {(r["cls"], r["n"], r["gidx"], r["scheme"], r["method"], r["model_seed"]) for r in csv.DictReader(f)}
    fh = open(out, "a", newline="")
    w = csv.DictWriter(fh, fieldnames=COLS)
    if not done and out.stat().st_size == 0:
        w.writeheader()
    device = None
    if a.part == "seeds":
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        seeds = [int(s) for s in a.seeds.split(",")]
        for s in seeds:
            run_eval.CKPT[("ours", s)] = ckpt(s)
            assert (ROOT / ckpt(s)).exists(), ckpt(s)
        warm = make_graph(dict(family="er", n=30, avg_deg=4.0, seed=1, scheme="random", weight_seed=1))
        for s in seeds:
            run_eval.rl_methods(warm, "ours", s, 0.0, device)
    items = [it for it in structures("grid", 10, (20, 50, 100, 200, 500)) if it[1] in NS]
    t0, rows = time.time(), 0
    for idx, (cls, n, gi, spec, wseed) in enumerate(items):
        base = make_graph(spec)
        for sch in SCHEMES:
            g = weighted_graph(base, sch, wseed)
            g.name = f"{spec['family']}_n{n}_g{gi}_{sch}"
            r0 = ref.loc[(cls, n, gi, sch)]
            assert int(r0["m"]) == g.num_edges and int(r0["graph_seed"]) == spec["seed"], (cls, n, gi, sch)
            row0 = dict(part="grid", cls=cls, family=spec["family"], scheme=sch, n=n, m=g.num_edges, gidx=gi,
                        graph_seed=spec["seed"], weight_seed=wseed, spec=json.dumps(spec), optimal="", bound="")

            def emit(method, S, rt, ms=""):
                nonlocal rows
                S, W = checked(g, S, method)
                w.writerow({**row0, "method": method, "model_seed": ms, "weight": W, "size": len(S), "runtime": rt,
                            "valid": True, "S": " ".join(map(str, S))})
                rows += 1

            if a.part == "baselines":
                assert abs(greedy_rr(g).weight - float(r0["weight"])) < 1e-9, "greedy_rr differs from raw_grid"
                for m, fn in BASELINES.items():
                    if (cls, str(n), str(gi), sch, m, "") not in done:
                        r = fn(g)
                        emit(m, r.vertices, r.runtime)
            else:
                for s in seeds:
                    if (cls, str(n), str(gi), sch, "ours_rr", str(s)) not in done:
                        for m, S, rt, _ in run_eval.rl_methods(g, "ours", s, 1.0, device):
                            emit(m, S, rt, ms=s)
        fh.flush()
        if (idx + 1) % 20 == 0:
            print(f"  {idx + 1}/{len(items)} structures, {rows} rows, {time.time() - t0:.0f}s", flush=True)
    fh.close()
    print(f"[{a.part}] {rows} new rows -> {out} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
