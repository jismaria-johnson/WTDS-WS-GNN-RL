"""Revision step 5: alpha-sweep of degree-tied weights.

    w(v) = ceil(100 * (d(v)/Delta)^alpha)        alpha > 0   (alpha = 1: the "degree" scheme)
    w(v) = ceil(100 * (delta/d(v))^(-alpha))     alpha < 0   (alpha = -1: the "inverse" scheme)
    w(v) = 100                                   alpha = 0
    clipped to [1, 100]; alpha = +-1 use the exact integer formulas of wtdsws/weights.py.

    python scripts/rev_alpha_sweep.py --part cpu                 # ILP (30 s), greedy + RR, greedy_tiebreak + RR, greedy + RR + LS
    python scripts/rev_alpha_sweep.py --part ours --seeds 0,1,2,3,4

Graphs: ER, BA, chordal, bipartite x n in {50, 100} x 10 graphs, FRESH test-range seeds (asserted unused by every
earlier results file). The same graph structure is used for every alpha. Every solution is checked with both TDS
checkers and W(S) recomputed. Output: results/revision/alpha_sweep_<part>.csv
"""
import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402

import run_eval  # noqa: E402
from run_eval import checked  # noqa: E402
from wtdsws.baselines import greedy_rr, greedy_rr_ls, ilp  # noqa: E402
from wtdsws.baselines.greedy_plus import greedy_tiebreak  # noqa: E402
from wtdsws.generators import make_graph  # noqa: E402
from wtdsws.graph import Graph  # noqa: E402
from wtdsws.graph_classes import CLASSES, class_spec  # noqa: E402
from wtdsws.seeds import MASTER_SEEDS, SEED_RANGES, split_of  # noqa: E402
from wtdsws.weights import degree_weights, inverse_weights  # noqa: E402

ALPHAS = (-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0)
NS = (50, 100)
PER_CELL = 10
MASTER = MASTER_SEEDS["test"] + 5_000          # a new RNG stream for this study
COLS = ["cls", "family", "n", "m", "gidx", "alpha", "graph_seed", "method", "model_seed", "weight", "size", "runtime",
        "valid", "optimal", "bound", "spec", "S"]


def alpha_weights(deg, alpha: float) -> np.ndarray:
    deg = np.asarray(deg, dtype=np.int64)
    if alpha == 1.0:
        return degree_weights(deg)
    if alpha == -1.0:
        return inverse_weights(deg)
    if alpha == 0.0:
        return np.full(len(deg), 100, dtype=np.int64)
    ratio = deg / deg.max() if alpha > 0 else deg.min() / deg
    w = np.array([math.ceil(100.0 * r ** abs(alpha) - 1e-9) for r in ratio], dtype=np.int64)
    return np.clip(w, 1, 100)


def with_alpha(base: Graph, alpha: float) -> Graph:
    meta = dict(base.meta)
    meta.update(scheme=f"alpha{alpha:+.2f}")
    return Graph(base.n, base.indptr, base.indices, alpha_weights(base.degrees, alpha).astype(np.float64),
                 base.name, meta)


def used_seeds():
    seen = set()
    for p in list((ROOT / "results").glob("*.csv")) + list((ROOT / "results" / "revision").glob("*.csv")):
        if p.name.startswith("alpha_sweep"):
            continue
        with open(p, newline="") as fh:
            rd = csv.DictReader(fh)
            if not rd.fieldnames or "graph_seed" not in rd.fieldnames:
                continue
            for r in rd:
                for k in ("graph_seed", "weight_seed"):
                    if r.get(k, "").isdigit():
                        seen.add(int(r[k]))
    return seen


def specs():
    rng = np.random.default_rng(MASTER)
    lo, hi = SEED_RANGES["test"]
    out = []
    for cls in CLASSES:
        for n in NS:
            for i in range(PER_CELL):
                spec = class_spec(cls, n, rng)
                spec["seed"] = int(rng.integers(lo, hi))
                out.append((cls, n, i, spec))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["cpu", "ours"], required=True)
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--ilp-time", type=float, default=30.0)
    a = ap.parse_args()
    items = specs()
    seeds_used = [sp["seed"] for *_, sp in items]
    assert all(split_of(s) == "test" for s in seeds_used)
    clash = set(seeds_used) & used_seeds()
    assert not clash, f"{len(clash)} graph seeds already used by earlier runs"
    out = ROOT / "results" / "revision" / f"alpha_sweep_{a.part}.csv"
    done = set()
    if out.exists():
        done = {(r["cls"], r["n"], r["gidx"], r["alpha"], r["method"], r["model_seed"])
                for r in csv.DictReader(open(out, newline=""))}
    fh = open(out, "a", newline="")
    wr = csv.DictWriter(fh, fieldnames=COLS)
    if not done:
        wr.writeheader()
    if a.part == "ours":
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        seeds = [int(s) for s in a.seeds.split(",")]
        for s in seeds:
            run_eval.CKPT[("ours", s)] = {0: "checkpoints/best.pt", 1: "checkpoints/seed1_best.pt"}.get(
                s, f"checkpoints/ws_seed{s}/best.pt")
            run_eval.rl_methods(make_graph(dict(family="er", n=30, avg_deg=4.0, seed=1, scheme="random",
                                                weight_seed=1)), "ours", s, 0.0, device)
    print(f"{len(items)} fresh graphs (test range, unused) x {len(ALPHAS)} alphas, part {a.part}", flush=True)
    t0 = time.time()
    for k, (cls, n, gi, spec) in enumerate(items):
        base = make_graph(spec)
        for al in ALPHAS:
            g = with_alpha(base, al)
            g.name = f"alpha_{spec['family']}_n{n}_g{gi}_a{al:+.2f}"
            row0 = dict(cls=cls, family=spec["family"], n=n, m=g.num_edges, gidx=gi, alpha=al,
                        graph_seed=spec["seed"], spec=json.dumps(spec))

            def emit(method, S, rt, ms="", optimal="", bound=""):
                S, W = checked(g, S, method)
                wr.writerow({**row0, "method": method, "model_seed": ms, "weight": W, "size": len(S), "runtime": rt,
                             "valid": True, "optimal": optimal, "bound": bound, "S": " ".join(map(str, S))})

            def todo(method, ms=""):
                return (cls, str(n), str(gi), str(al), method, str(ms)) not in done

            if a.part == "cpu":
                for m, fn in (("greedy_rr", greedy_rr), ("greedy_tiebreak_rr", greedy_tiebreak),
                              ("greedy_rr_ls", lambda g_: greedy_rr_ls(g_, ls_time=1.0, seed=0))):
                    if todo(m):
                        r = fn(g)
                        emit(m, r.vertices, r.runtime)
                if todo("ilp"):
                    r = ilp(g, time_limit=a.ilp_time, threads=8)
                    emit("ilp", r.vertices, r.runtime, optimal=bool(r.info["optimal"]), bound=float(r.info["bound"]))
            else:
                for s in seeds:
                    if todo("ours_rr", s):
                        for m, S, rt, _ in run_eval.rl_methods(g, "ours", s, 1.0, device):
                            emit(m, S, rt, ms=s)
        fh.flush()
        if (k + 1) % 10 == 0:
            print(f"  {k + 1}/{len(items)} graphs, {time.time() - t0:.0f}s", flush=True)
    fh.close()
    print(f"done -> {out} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
