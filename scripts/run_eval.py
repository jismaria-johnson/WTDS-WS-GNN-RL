"""Ground-truth evaluation (Section 6).

    python scripts/run_eval.py --part grid      # class x scheme x n in {20,50,100,200,500}
    python scripts/run_eval.py --part subclass  # unit interval + convex bipartite, n in {50,100}

For every (class, n) we draw `per_cell` graph STRUCTURES from the test seed range and evaluate each of them under
all four weight schemes, so schemes are compared on identical graphs. One CSV row per (graph, scheme, method,
model seed). Every solution is checked with is_total_dominating_set (both checker implementations) and its weight
is recomputed from the solution and the weights before the row is written. Rows already in the CSV are skipped, so
the run can be resumed.
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from wtdsws.baselines import greedy, greedy_rr, greedy_rr_ls, ilp
from wtdsws.checker import is_total_dominating_set, verify
from wtdsws.generators import make_graph
from wtdsws.graph_classes import CLASSES, class_spec
from wtdsws.postprocess import local_search, redundancy_removal
from wtdsws.rollout import rollout
from wtdsws.seeds import MASTER_SEEDS, SEED_RANGES, split_of
from wtdsws.solve import get_model
from wtdsws.weights import SCHEMES, weighted_graph

COLS = ["part", "cls", "family", "scheme", "n", "m", "gidx", "graph_seed", "weight_seed", "method", "model_seed",
        "weight", "size", "runtime", "valid", "optimal", "bound", "spec", "S"]
CKPT = {("ours", 0): "checkpoints/best.pt", ("ours", 1): "checkpoints/seed1_best.pt",
        ("zero", 0): "checkpoints/pretrained/wtds_seed0_best.pt", ("zero", 1): "checkpoints/pretrained/wtds_seed1_best.pt"}


def structures(part, per_cell, ns=(20, 50, 100, 200, 500)):
    """[(cls, n, gidx, spec)] with graph seeds from the TEST range (disjoint from train / validation)."""
    rng = np.random.default_rng(MASTER_SEEDS["test"] + (0 if part == "grid" else 1))
    lo, hi = SEED_RANGES["test"]
    out = []
    if part == "grid":
        cells = [(c, None, n) for c in CLASSES for n in ns]
    else:
        cells = [(c, sub, n) for c, sub in (("chordal", "unit_interval"), ("bipartite", "convex_bip")) for n in (50, 100)]
    for cls, sub, n in cells:
        for i in range(per_cell):
            spec = class_spec(cls, n, rng, subtype=sub)
            spec["seed"] = int(rng.integers(lo, hi))
            out.append((cls, n, i, spec, int(rng.integers(lo, hi))))
    return out


def checked(g, S, method):
    S = sorted(int(v) for v in set(S))
    ok = verify(g, S) and is_total_dominating_set(g.adj, S)
    if not ok:
        raise AssertionError(f"{method}: INVALID solution on {g.name}")
    return S, float(g.w[S].sum())                       # W(S) recomputed from S and the weights


def rl_methods(g, kind, seed, ls_time, device):
    model = get_model(ROOT / CKPT[(kind, seed)], device)
    t0 = time.perf_counter()
    r = rollout(model, [g], device, env_kwargs=dict(getattr(model, "env_kwargs", {}) or {}), multi_select="auto")[0]
    S_rr = redundancy_removal(g, r["solution"], scores=r["q"], protected=r["forced"])
    t_rr = time.perf_counter() - t0
    out = [(f"{kind}_rr", S_rr, t_rr, {})]
    if kind == "ours":
        S_ls = local_search(g, S_rr, time_limit=ls_time, seed=seed)
        out.append(("ours_rr_ls", S_ls, time.perf_counter() - t0, {}))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["grid", "subclass"], default="grid")
    ap.add_argument("--per-cell", type=int, default=10)
    ap.add_argument("--ilp-time", type=float, default=30.0)
    ap.add_argument("--ilp-max-n", type=int, default=200)
    ap.add_argument("--ilp-threads", type=int, default=8)
    ap.add_argument("--ls-time", type=float, default=1.0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--smoke", action="store_true", help="pipeline test: 1 graph per cell, n <= 50, zero-shot "
                                                         "checkpoints stand in for ours, output in results/smoke/")
    a = ap.parse_args()
    ns = (20, 50, 100, 200, 500)
    if a.smoke:
        a.per_cell, ns, a.ilp_max_n = 1, (20, 50), 20
        CKPT[("ours", 0)], CKPT[("ours", 1)] = CKPT[("zero", 0)], CKPT[("zero", 1)]
        a.out = a.out or str(ROOT / "results" / "smoke" / f"raw_{a.part}.csv")
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(a.out or ROOT / "results" / f"raw_{a.part}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        with open(out, newline="") as f:
            for r in csv.DictReader(f):
                done.add((r["cls"], r["family"], r["n"], r["gidx"], r["scheme"], r["method"], r["model_seed"]))
    fh = open(out, "a", newline="")
    w = csv.DictWriter(fh, fieldnames=COLS)
    if not done:
        w.writeheader()
    items = structures(a.part, a.per_cell, ns)
    assert all(split_of(sp["seed"]) == "test" and split_of(ws) == "test" for *_, sp, ws in items)
    print(f"[{a.part}] {len(items)} graph structures x {len(SCHEMES)} schemes on {device}; "
          f"all graph/weight seeds in the test range {SEED_RANGES['test']}", flush=True)
    # warm up the GPU models so the first timed rollout is not penalised
    warm = make_graph(dict(family="er", n=30, avg_deg=4.0, seed=1, scheme="random", weight_seed=1))
    for k in CKPT:
        rl_methods(warm, k[0], k[1], 0.0, device)
    t_start, n_rows = time.time(), 0
    for idx, (cls, n, gi, spec, wseed) in enumerate(items):
        base = make_graph(spec)                            # class asserted inside make_graph
        for sch in SCHEMES:
            g = weighted_graph(base, sch, wseed)
            g.name = f"{spec['family']}_n{n}_g{gi}_{sch}"
            row0 = dict(part=a.part, cls=cls, family=spec["family"], scheme=sch, n=n, m=g.num_edges, gidx=gi,
                        graph_seed=spec["seed"], weight_seed=wseed, spec=json.dumps(spec))

            def emit(method, S, rt, model_seed="", optimal="", bound=""):
                nonlocal n_rows
                S, W = checked(g, S, method)
                w.writerow({**row0, "method": method, "model_seed": model_seed, "weight": W, "size": len(S),
                            "runtime": rt, "valid": True, "optimal": optimal, "bound": bound,
                            "S": " ".join(map(str, S))})
                n_rows += 1

            def todo(method, ms=""):
                return (cls, spec["family"], str(n), str(gi), sch, method, str(ms)) not in done

            if todo("greedy"):
                r = greedy(g)
                emit("greedy", r.vertices, r.runtime)
            if todo("greedy_rr"):
                r = greedy_rr(g)
                emit("greedy_rr", r.vertices, r.runtime)
            if todo("greedy_rr_ls"):
                r = greedy_rr_ls(g, ls_time=a.ls_time, seed=0)
                emit("greedy_rr_ls", r.vertices, r.runtime)
            for kind in ("zero", "ours"):
                for s in (0, 1):
                    if todo(f"{kind}_rr", s):
                        for method, S, rt, _ in rl_methods(g, kind, s, a.ls_time, device):
                            emit(method, S, rt, model_seed=s)
            if n <= a.ilp_max_n and todo("ilp"):
                r = ilp(g, time_limit=a.ilp_time, threads=a.ilp_threads)
                emit("ilp", r.vertices, r.runtime, optimal=bool(r.info["optimal"]), bound=float(r.info["bound"]))
            fh.flush()
        if (idx + 1) % 10 == 0:
            print(f"  {idx + 1}/{len(items)} structures ({cls}, n={n}) rows {n_rows} "
                  f"elapsed {time.time() - t_start:.0f}s", flush=True)
    fh.close()
    print(f"[{a.part}] done: {n_rows} new rows in {time.time() - t_start:.0f}s -> {out}", flush=True)


if __name__ == "__main__":
    main()
