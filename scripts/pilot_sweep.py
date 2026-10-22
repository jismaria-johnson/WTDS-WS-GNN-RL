"""STEP 1 of the exact-vs-ours plot update: a pilot sweep that finds where exact, ours and greedy + RR differ most.

    python scripts/pilot_sweep.py              # -> results/pilot_sweep.csv (resumable)

The pilot graphs are drawn from the PILOT seed range (SEED_RANGES["pilot"]), disjoint from train / val / test, so
choosing plot settings on them never touches test data. The final plots use fresh test-range graphs
(scripts/exact_vs_ours_final.py).

Settings (class, parameter):
  tree            random labelled tree | BA m = 1
  unit_interval   average degree 3 | 6 | 10
  convex_bip      average degree 3 | 6 | 10
  two_tree        the ktree generator with k = 2 (as generated)
n values: grid A = {100, ..., 1000} step 100, grid B = {200, ..., 2000} step 200. The two grids share graphs at the
common n (200, 400, ..., 1000), so 15 distinct n per setting. 5 graphs per (setting, n), 5 weightings each.

Methods: exact (tree DP | our unit-interval DP | CP-SAT ILP 60 s), ilp_check (the same lexicographic ILP on trees
and unit-interval graphs: the run aborts if a proven ILP disagrees with the DP), greedy_rr, ours_rr (best.pt).
Every solution is checked with both TDS checkers and W(S) is recomputed from the set before the row is written.
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

import numpy as np  # noqa: E402

from exact_vs_ours import build, checked  # noqa: E402
from wtdsws.baselines import greedy_rr  # noqa: E402
from wtdsws.baselines.exact_special import ilp_exact, tree_tds, unit_interval_tds  # noqa: E402
from wtdsws.exact_validation import WEIGHTINGS, with_scheme  # noqa: E402
from wtdsws.generators import make_graph  # noqa: E402
from wtdsws.postprocess import redundancy_removal  # noqa: E402
from wtdsws.rollout import rollout  # noqa: E402
from wtdsws.seeds import SEED_RANGES, split_of  # noqa: E402
from wtdsws.solve import get_model  # noqa: E402

PILOT_MASTER = 777_100
GRID_A = tuple(range(100, 1001, 100))
GRID_B = tuple(range(200, 2001, 200))
NS = tuple(sorted(set(GRID_A) | set(GRID_B)))
PER_CELL = 5
SETTINGS = (("tree", "random_tree"), ("tree", "ba_m1"),
            ("unit_interval", "deg3"), ("unit_interval", "deg6"), ("unit_interval", "deg10"),
            ("convex_bip", "deg3"), ("convex_bip", "deg6"), ("convex_bip", "deg10"),
            ("two_tree", "k2"))
CKPT = ROOT / "checkpoints" / "best.pt"
COLS = ["cls", "param", "family", "n", "m", "gidx", "scheme", "graph_seed", "weight_seed", "method", "size", "weight",
        "runtime", "valid", "proven_optimal", "status", "exact_weight", "exact_size", "exact_proven"]


def make_spec(cls, param, n, seed):
    if cls == "tree":
        return dict(family="tree", n=n, seed=seed) if param == "random_tree" else dict(family="ba", n=n, m=1, seed=seed)
    if cls == "two_tree":
        return dict(family="ktree", n=n, k=2, cls="chordal", seed=seed)
    d = float(param[3:])
    fam, c = ("unit_interval", "chordal") if cls == "unit_interval" else ("convex_bip", "bipartite")
    return dict(family=fam, n=n, avg_deg=d, cls=c, seed=seed)


def specs(master=PILOT_MASTER, split="pilot", settings=SETTINGS, ns=NS, per_cell=PER_CELL):
    """[(cls, param, n, gidx, spec, weight_seed)] - deterministic, every seed in the given split's range."""
    rng = np.random.default_rng(master)
    lo, hi = SEED_RANGES[split]
    out = []
    for cls, param in settings:
        for n in ns:
            for i in range(per_cell):
                spec = make_spec(cls, param, n, int(rng.integers(lo, hi)))
                out.append((cls, param, n, i, spec, int(rng.integers(lo, hi))))
    return out


def ours_rr(model, g, device):
    t0 = time.perf_counter()
    r = rollout(model, [g], device, env_kwargs=dict(getattr(model, "env_kwargs", {}) or {}), multi_select="auto")[0]
    S = redundancy_removal(g, r["solution"], scores=r["q"], protected=r["forced"])
    return S, time.perf_counter() - t0


def ours_rr_batched(model, graphs, device):
    """GPU: one batched rollout over all weightings of a graph (identical sets to one-by-one rollouts; checked on
    5 weightings of a 1000-vertex 2-tree). Runtime per graph = batched rollout time / #graphs + its own RR time."""
    t0 = time.perf_counter()
    rs = rollout(model, graphs, device, env_kwargs=dict(getattr(model, "env_kwargs", {}) or {}), multi_select="auto")
    t_roll = (time.perf_counter() - t0) / len(graphs)
    out = []
    for g, r in zip(graphs, rs):
        t1 = time.perf_counter()
        S = redundancy_removal(g, r["solution"], scores=r["q"], protected=r["forced"])
        out.append((S, t_roll + time.perf_counter() - t1))
    return out


def solve_all(cls, g, pts, ilp_time, threads, extra):
    """[(method, S, runtime, proven, status)] with exact first."""
    res = []
    if cls == "tree":
        r = tree_tds(g)
        res.append(("exact", r.vertices, r.runtime, True, "dp"))
    elif cls == "unit_interval":
        r = unit_interval_tds(g, pts)
        res.append(("exact", r.vertices, r.runtime, True, "dp"))
    e = ilp_exact(g, ilp_time, threads=threads)
    rt, status = e.runtime, e.info["status"]
    if not e.info["optimal"]:                    # CP-SAT run-to-run variance: one retry with another random seed
        e2 = ilp_exact(g, ilp_time, threads=threads, seed=1)
        rt += e2.runtime
        status = f"{status}+retry:{e2.info['status']}"
        if e2.info["optimal"] or e2.weight < e.weight:
            e = e2
    res.append(("exact" if cls in ("convex_bip", "two_tree") else "ilp_check", e.vertices, rt,
                bool(e.info["optimal"]), status))
    r = greedy_rr(g)
    res.append(("greedy_rr", r.vertices, r.runtime, "", ""))
    res.extend(extra(g))
    return res


def check_rows(g, res, row0):
    """Verify every set (both checkers), recompute W, cross-check DP vs ILP; return CSV rows."""
    rows, ex = [], None
    for m, S, rt, proven, status in res:
        S, W = checked(g, S, m)
        if m == "exact":
            ex = (W, len(S), proven)
        if m == "ilp_check" and proven and (W, len(S)) != ex[:2]:
            raise AssertionError(f"DP {ex[:2]} != proven ILP {(W, len(S))} on {g.name}: fix the DP")
        if m == "ilp_check" and not proven and W < ex[0] - 1e-9:
            raise AssertionError(f"ILP found W {W} below the DP {ex[0]} on {g.name}: fix the DP")
        if ex[2] and W < ex[0] - 1e-9:
            raise AssertionError(f"{m} beat a proven optimum on {g.name}")
        rows.append({**row0, "method": m, "size": len(S), "weight": W, "runtime": f"{rt:.6f}", "valid": True,
                     "proven_optimal": proven, "status": status, "exact_weight": ex[0], "exact_size": ex[1],
                     "exact_proven": ex[2], "S": " ".join(map(str, S))})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results" / "pilot_sweep.csv"))
    ap.add_argument("--ilp-time", type=float, default=60.0)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--redo-unproven", action="store_true",
                    help="re-solve graphs with an unproven ILP row (e.g. a timeout under CPU contention); the new "
                         "rows are appended and pilot_choose keeps the later copy")
    ap.add_argument("--ns", default="", help="comma list: run only these n (parallel workers)")
    ap.add_argument("--only", default="", help="comma list cls/param: run only these settings (parallel workers)")
    a = ap.parse_args()
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = get_model(CKPT, device)
    items = specs()                                      # seeds are drawn for ALL settings, then filtered
    if a.only:
        keep = {tuple(x.split("/")) for x in a.only.split(",")}
        items = [it for it in items if it[:2] in keep]
    if a.ns:
        items = [it for it in items if it[2] in {int(x) for x in a.ns.split(",")}]
    seeds = [s for *_, sp, ws in items for s in (sp["seed"], ws)]
    assert all(split_of(s) == "pilot" for s in seeds), "pilot seed outside the pilot range"
    out = Path(a.out)
    done = set()
    if out.exists():
        with open(out, newline="") as f:
            rows_ = list(csv.DictReader(f))
        done = {(r["cls"], r["param"], r["n"], r["gidx"]) for r in rows_}
        if a.redo_unproven:
            bad = {(r["cls"], r["param"], r["n"], r["gidx"]) for r in rows_
                   if r["method"] in ("exact", "ilp_check") and r["proven_optimal"] != "True"}
            done -= bad
            print(f"re-solving {len(bad)} graphs with an unproven ILP row", flush=True)
        cnt = {}
        for r in rows_:
            k = (r["cls"], r["param"], r["n"], r["gidx"])
            cnt[k] = cnt.get(k, 0) + 1
        part = {k for k, c in cnt.items() if c % (len(WEIGHTINGS) * (4 if k[0] in ("tree", "unit_interval") else 3))}
        done -= part                                         # a graph cut off mid-write (process stopped) is redone
        if part:
            print(f"re-solving {len(part)} incompletely written graphs", flush=True)
    new_file = not out.exists() or out.stat().st_size == 0
    fh = open(out, "a", newline="")
    wr = csv.DictWriter(fh, fieldnames=COLS, extrasaction="ignore")   # the pilot does not store S
    if new_file:
        wr.writeheader()
    print(f"{len(items)} pilot graphs x {len(WEIGHTINGS)} weightings on {device}; seeds in {SEED_RANGES['pilot']}",
          flush=True)
    ours_rr(model, with_scheme(make_graph(dict(family="tree", n=40, seed=1)), "random", 1), device)   # warm-up
    t_start = time.time()
    for k, (cls, param, n, gi, spec, wseed) in enumerate(items):
        if (cls, param, str(n), str(gi)) in done:
            continue
        base, G, pts = build(cls, spec)
        rows = []
        gs = [with_scheme(base, sch, wseed) for sch in WEIGHTINGS]
        ob = ours_rr_batched(model, gs, device)                 # GPU, all weightings at once
        for j, (sch, g) in enumerate(zip(WEIGHTINGS, gs)):
            g.name = f"pilot_{cls}_{param}_n{n}_g{gi}_{sch}"
            row0 = dict(cls=cls, param=param, family=spec["family"], n=n, m=g.num_edges, gidx=gi, scheme=sch,
                        graph_seed=spec["seed"], weight_seed=wseed)
            res = solve_all(cls, g, pts, a.ilp_time, a.threads,
                            lambda g_, j=j: [("ours_rr", *ob[j], "", "")])
            rows.extend(check_rows(g, res, row0))
        for row in rows:
            wr.writerow(row)
        fh.flush()
        if (k + 1) % 25 == 0:
            print(f"  {k + 1}/{len(items)} ({cls}/{param}, n={n}) elapsed {time.time() - t_start:.0f}s", flush=True)
    fh.close()
    print(f"done in {time.time() - t_start:.0f}s -> {out}", flush=True)


if __name__ == "__main__":
    main()
