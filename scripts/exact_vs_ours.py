"""Exact vs ours on graph classes with exact algorithms.

    python scripts/exact_vs_ours.py            # -> results/exact_vs_ours.csv (resumable)

Classes (panels): trees (5 random labelled trees + 5 BA m = 1 per n), unit-interval, convex bipartite, 2-trees;
n in {20, 50, 100, 200, 500, 1000}; 10 graphs per (class, n). Graph and weight seeds are NEW draws from the TEST
seed range (a fresh master seed; asserted disjoint from the seeds of results/raw_grid.csv and raw_subclass.csv).
Each graph gets 5 weightings: unit, degree, inverse, random, mixed (random/mixed use the stored weight_seed).

Methods per (graph, weighting):
  exact         tree DP | our unit-interval DP | CP-SAT ILP (60 s; convex bipartite, 2-trees), all lexicographic
                (min W, then min |S|)
  ilp_check     trees and unit-interval only: the same lexicographic CP-SAT ILP, a cross-check of the DP on every
                graph (the run aborts if a proven ILP disagrees with the DP)
  greedy_rr     weighted greedy + redundancy removal
  ours_rr       GNN+RL + RR, checkpoints/best.pt (seed 0) and checkpoints/seed1_best.pt (seed 1)
  ours_rr_ls    ours_rr + 1 s local search (table only)
Every solution is checked with is_total_dominating_set (both checker implementations) and W(S) is recomputed
from the set and the weights before the row is written.
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402

from wtdsws.baselines import greedy_rr  # noqa: E402
from wtdsws.baselines.exact_special import ilp_exact, tree_tds, unit_interval_tds  # noqa: E402
from wtdsws.checker import is_total_dominating_set, verify  # noqa: E402
from wtdsws.exact_validation import WEIGHTINGS, with_scheme  # noqa: E402
from wtdsws.generators import _build_nx, make_graph  # noqa: E402
from wtdsws.graph_classes import (class_spec, is_convex_bipartite, is_k_tree,  # noqa: E402
                                  is_unit_interval_realisation)
from wtdsws.postprocess import local_search, redundancy_removal  # noqa: E402
from wtdsws.rollout import rollout  # noqa: E402
from wtdsws.seeds import MASTER_SEEDS, SEED_RANGES, split_of  # noqa: E402
from wtdsws.solve import get_model  # noqa: E402

CLASSES = ("tree", "unit_interval", "convex_bip", "two_tree")
NS = (20, 50, 100, 200, 500, 1000)
PER_CELL = 10
MASTER = MASTER_SEEDS["test"] + 2          # grid used +0, subclass +1: a new, unused stream
CKPT = {0: ROOT / "checkpoints" / "best.pt", 1: ROOT / "checkpoints" / "seed1_best.pt"}
COLS = ["cls", "family", "n", "m", "gidx", "scheme", "graph_seed", "weight_seed", "method", "model_seed", "size",
        "weight", "runtime", "valid", "proven_optimal", "exact_weight", "exact_size", "optimal_weight_reached",
        "same_size_as_exact", "exact_proven", "status", "spec", "S"]


def specs():
    """[(cls, n, gidx, spec, weight_seed)] - deterministic."""
    rng = np.random.default_rng(MASTER)
    lo, hi = SEED_RANGES["test"]
    out = []
    for cls in CLASSES:
        for n in NS:
            for i in range(PER_CELL):
                if cls == "tree":
                    spec = dict(family="tree", n=n) if i < PER_CELL // 2 else dict(family="ba", n=n, m=1)
                elif cls == "unit_interval":
                    spec = class_spec("chordal", n, rng, subtype="unit_interval")
                elif cls == "convex_bip":
                    spec = class_spec("bipartite", n, rng, subtype="convex_bip")
                else:
                    spec = dict(family="ktree", n=n, k=2, cls="chordal")
                spec["seed"] = int(rng.integers(lo, hi))
                out.append((cls, n, i, spec, int(rng.integers(lo, hi))))
    return out


def old_test_seeds():
    seen = set()
    for f in ("raw_grid.csv", "raw_subclass.csv"):
        p = ROOT / "results" / f
        if p.exists():
            with open(p, newline="") as fh:
                for r in csv.DictReader(fh):
                    seen.update((int(r["graph_seed"]), int(r["weight_seed"])))
    return seen


def build(cls, spec):
    """(Graph, networkx graph, interval points or None) with the class asserted."""
    G = _build_nx(dict(spec))
    g = make_graph(spec)                                   # the project's own pipeline (asserts its class checks)
    assert list(G.nodes()) == list(range(spec["n"])) and g.n == spec["n"]
    assert {tuple(sorted(e)) for e in G.edges()} == {tuple(e) for e in g.edges().tolist()}, "nx / Graph mismatch"
    assert min(d for _, d in G.degree()) > 0, "isolated vertex"
    pts = None
    if cls == "tree":
        assert nx.is_tree(G), "not a tree"
    elif cls == "unit_interval":
        pts = G.graph["points"]
        assert nx.is_chordal(G) and is_unit_interval_realisation(G, pts, G.graph["r"]), "not unit interval"
    elif cls == "convex_bip":
        n1 = G.graph["n1"]
        assert nx.is_bipartite(G) and is_convex_bipartite(G, list(range(n1)), list(range(n1, G.number_of_nodes())))
    else:
        assert nx.is_chordal(G) and is_k_tree(G, 2), "not a 2-tree"
    return g, G, pts


def checked(g, S, method):
    S = sorted(int(v) for v in set(S))
    if not (verify(g, S) and is_total_dominating_set(g.adj, S)):
        raise AssertionError(f"{method}: INVALID solution on {g.name}")
    return S, float(g.w[S].sum())


def ours(g, seed, device, ls_time):
    model = get_model(CKPT[seed], device)
    t0 = time.perf_counter()
    r = rollout(model, [g], device, env_kwargs=dict(getattr(model, "env_kwargs", {}) or {}), multi_select="auto")[0]
    S_rr = redundancy_removal(g, r["solution"], scores=r["q"], protected=r["forced"])
    t_rr = time.perf_counter() - t0
    S_ls = local_search(g, S_rr, time_limit=ls_time, seed=seed)
    return [("ours_rr", S_rr, t_rr), ("ours_rr_ls", S_ls, time.perf_counter() - t0)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results" / "exact_vs_ours.csv"))
    ap.add_argument("--ilp-time", type=float, default=60.0)
    ap.add_argument("--ls-time", type=float, default=1.0)
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for p in CKPT.values():
        assert p.exists(), p
    items = specs()
    seeds = [s for *_, sp, ws in items for s in (sp["seed"], ws)]
    assert all(split_of(s) == "test" for s in seeds), "seed outside the test range"
    clash = set(seeds) & old_test_seeds()
    assert not clash, f"{len(clash)} seeds reused from earlier test runs"
    out = Path(a.out)
    done = set()
    if out.exists():
        with open(out, newline="") as f:
            done = {(r["cls"], r["n"], r["gidx"]) for r in csv.DictReader(f)}
    fh = open(out, "a", newline="")
    wr = csv.DictWriter(fh, fieldnames=COLS)
    if not done:
        wr.writeheader()
    print(f"{len(items)} graphs x {len(WEIGHTINGS)} weightings on {device}; seeds in the test range "
          f"{SEED_RANGES['test']}, none shared with earlier test runs", flush=True)
    warm = with_scheme(make_graph(dict(family="tree", n=40, seed=1)), "random", 1)
    for s in CKPT:
        ours(warm, s, device, 0.0)
    t_start = time.time()
    for k, (cls, n, gi, spec, wseed) in enumerate(items):
        if (cls, str(n), str(gi)) in done:
            continue
        base, G, pts = build(cls, spec)
        rows = []
        for sch in WEIGHTINGS:
            g = with_scheme(base, sch, wseed)
            g.name = f"{cls}_n{n}_g{gi}_{sch}"
            row0 = dict(cls=cls, family=spec["family"], n=n, m=g.num_edges, gidx=gi, scheme=sch,
                        graph_seed=spec["seed"], weight_seed=wseed, spec=json.dumps(spec))
            res = []                                       # (method, model_seed, S, runtime, proven, status)
            if cls == "tree":
                r = tree_tds(g)
                res.append(("exact", "", r.vertices, r.runtime, True, "dp"))
            elif cls == "unit_interval":
                r = unit_interval_tds(g, pts)
                res.append(("exact", "", r.vertices, r.runtime, True, "dp"))
            e = ilp_exact(g, a.ilp_time, threads=a.threads)
            res.append(("exact" if cls in ("convex_bip", "two_tree") else "ilp_check", "", e.vertices, e.runtime,
                        bool(e.info["optimal"]), e.info["status"]))
            r = greedy_rr(g)
            res.append(("greedy_rr", "", r.vertices, r.runtime, "", ""))
            for s in CKPT:
                for m, S, rt in ours(g, s, device, a.ls_time):
                    res.append((m, s, S, rt, "", ""))
            ex = None
            for m, ms, S, rt, proven, status in res:
                S, W = checked(g, S, m)
                if m == "exact":
                    ex = (W, len(S), proven)
                if m == "ilp_check" and proven and (W, len(S)) != (ex[0], ex[1]):
                    raise AssertionError(f"DP {ex[:2]} != proven ILP {(W, len(S))} on {g.name}: fix the DP")
                if m == "ilp_check" and not proven and W < ex[0] - 1e-9:
                    raise AssertionError(f"ILP found W {W} below the DP {ex[0]} on {g.name}: fix the DP")
                rows.append({**row0, "method": m, "model_seed": ms, "size": len(S), "weight": W,
                             "runtime": f"{rt:.6f}", "valid": True, "proven_optimal": proven,
                             "exact_weight": ex[0], "exact_size": ex[1],
                             "optimal_weight_reached": abs(W - ex[0]) < 1e-9, "same_size_as_exact": len(S) == ex[1],
                             "exact_proven": ex[2], "status": status, "S": " ".join(map(str, S))})
        for row in rows:
            wr.writerow(row)
        fh.flush()
        if (k + 1) % 10 == 0:
            print(f"  {k + 1}/{len(items)} graphs ({cls}, n={n}) elapsed {time.time() - t_start:.0f}s", flush=True)
    fh.close()
    print(f"done in {time.time() - t_start:.0f}s -> {out}", flush=True)


if __name__ == "__main__":
    main()
