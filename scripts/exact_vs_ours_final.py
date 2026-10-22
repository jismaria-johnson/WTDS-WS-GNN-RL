"""STEP 3 of the exact-vs-ours plot update: the final run on FRESH test-range graphs with the settings chosen by the
pilot (results/chosen_settings.json, scripts/pilot_choose.py).

    python scripts/exact_vs_ours_final.py      # -> results/exact_vs_ours_final.csv (resumable)

Per class: the chosen parameter and n grid, 10 NEW graphs per n, 5 weightings each. Graph and weight seeds come from
the TEST range with a new master seed (MASTER_SEEDS["test"] + 3; +0 grid, +1 subclass, +2 the first exact-vs-ours
run); the script asserts that none of them was used by results/raw_grid.csv, raw_subclass.csv or exact_vs_ours.csv,
and that no pilot graph is reused (the pilot range is disjoint by construction, and asserted again here).

Methods: exact (tree DP | our unit-interval DP | CP-SAT ILP 60 s), ilp_check (trees, unit-interval), greedy_rr,
ours_rr and ours_rr_ls (1 s) for both checkpoints (best.pt, seed1_best.pt). Every solution passes both TDS checkers
and W(S) is recomputed from the set before the row is written (scripts/pilot_sweep.check_rows).
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

from exact_vs_ours import CKPT, build, ours  # noqa: E402
from wtdsws.postprocess import local_search, redundancy_removal  # noqa: E402
from wtdsws.rollout import rollout  # noqa: E402
from wtdsws.solve import get_model  # noqa: E402
from pilot_sweep import check_rows, solve_all, specs  # noqa: E402
from wtdsws.exact_validation import WEIGHTINGS, with_scheme  # noqa: E402
from wtdsws.generators import make_graph  # noqa: E402
from wtdsws.seeds import MASTER_SEEDS, SEED_RANGES, split_of  # noqa: E402

MASTER = MASTER_SEEDS["test"] + 3
PER_CELL = 10
CLASSES = ("tree", "unit_interval", "convex_bip", "two_tree")
COLS = ["cls", "param", "family", "n", "m", "gidx", "scheme", "graph_seed", "weight_seed", "method", "model_seed",
        "size", "weight", "runtime", "valid", "proven_optimal", "status", "exact_weight", "exact_size", "exact_proven",
        "S"]


def final_specs(chosen=ROOT / "results" / "chosen_settings.json", classes=CLASSES):
    ch = json.loads(Path(chosen).read_text(encoding="utf-8"))["chosen"]
    out = []
    for i, cls in enumerate(CLASSES):                       # one RNG stream per class: independent of the others
        if cls not in classes:
            continue
        c = ch[cls]
        out += specs(master=MASTER * 10 + i, split="test", settings=((cls, c["param"]),), ns=tuple(c["ns"]),
                     per_cell=PER_CELL)
    return out


def ours_batched(graphs, seed, device, ls_time):
    """GPU: one batched rollout of checkpoint `seed` over all weightings of a graph, then RR and LS per graph.
    Runtime per graph = (batched rollout time) / (#graphs) + its own RR time [+ LS time]: the amortised GPU cost."""
    model = get_model(CKPT[seed], device)
    t0 = time.perf_counter()
    rs = rollout(model, graphs, device, env_kwargs=dict(getattr(model, "env_kwargs", {}) or {}), multi_select="auto")
    t_roll = (time.perf_counter() - t0) / len(graphs)
    out = []
    for g, r in zip(graphs, rs):
        t1 = time.perf_counter()
        S_rr = redundancy_removal(g, r["solution"], scores=r["q"], protected=r["forced"])
        t_rr = t_roll + time.perf_counter() - t1
        t2 = time.perf_counter()
        S_ls = local_search(g, S_rr, time_limit=ls_time, seed=seed)
        out.append([("ours_rr", S_rr, t_rr), ("ours_rr_ls", S_ls, t_rr + time.perf_counter() - t2)])
    return out


def used_seeds(own):
    """Graph / weight seeds of every earlier run (and the pilot), except the caller's own output file."""
    seen = set()
    files = ["raw_grid.csv", "raw_subclass.csv", "exact_vs_ours.csv", "pilot_sweep.csv", "exact_vs_ours_final.csv",
             "stress_test.csv"] + [p.name for p in (ROOT / "results").glob("pilot_sweep_w*.csv")]         + [p.name for p in (ROOT / "results").glob("exact_vs_ours_final_part*.csv")]
    for f in files:
        p = ROOT / "results" / f
        if f == own or (own.startswith("exact_vs_ours_final") and f.startswith("exact_vs_ours_final")):
            continue
        if p.exists():
            with open(p, newline="") as fh:
                for r in csv.DictReader(fh):
                    if r["graph_seed"] != "graph_seed":          # a header repeated by a restarted worker
                        seen.update((int(r["graph_seed"]), int(r["weight_seed"])))
    return seen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results" / "exact_vs_ours_final.csv"))
    ap.add_argument("--ilp-time", type=float, default=60.0)
    ap.add_argument("--ls-time", type=float, default=1.0)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--chosen", default=str(ROOT / "results" / "chosen_settings.json"))
    ap.add_argument("--classes", default=",".join(CLASSES))
    ap.add_argument("--shard", default="0/1", help="i/k: this worker runs items i, i+k, ... (parallel workers)")
    a = ap.parse_args()
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for p in CKPT.values():
        assert p.exists(), p
    items = final_specs(a.chosen, a.classes.split(","))
    si, sk = map(int, a.shard.split("/"))
    items = items[si::sk]
    seeds = [s for *_, sp, ws in items for s in (sp["seed"], ws)]
    assert all(split_of(s) == "test" for s in seeds), "seed outside the test range"
    clash = set(seeds) & used_seeds("exact_vs_ours_final.csv")
    assert not clash, f"{len(clash)} seeds reused from earlier runs or the pilot"
    out = Path(a.out)
    done = set()
    if out.exists():
        with open(out, newline="") as f:
            done = {(r["cls"], r["n"], r["gidx"]) for r in csv.DictReader(f)}
    fh = open(out, "a", newline="")
    wr = csv.DictWriter(fh, fieldnames=COLS, extrasaction="ignore")
    if not done:
        wr.writeheader()
    print(f"{len(items)} fresh graphs x {len(WEIGHTINGS)} weightings on {device}; seeds in the test range "
          f"{SEED_RANGES['test']}, none shared with earlier runs or the pilot", flush=True)
    warm = with_scheme(make_graph(dict(family="tree", n=40, seed=1)), "random", 1)
    for s in CKPT:
        ours(warm, s, device, 0.0)
    t_start = time.time()
    for k, (cls, param, n, gi, spec, wseed) in enumerate(items):
        if (cls, str(n), str(gi)) in done:
            continue
        base, G, pts = build(cls, spec)
        rows = []
        gs = []
        for sch in WEIGHTINGS:
            g = with_scheme(base, sch, wseed)
            g.name = f"final_{cls}_{param}_n{n}_g{gi}_{sch}"
            gs.append(g)
        ob = {s: ours_batched(gs, s, device, a.ls_time) for s in CKPT}          # GPU, all weightings at once
        for j, (sch, g) in enumerate(zip(WEIGHTINGS, gs)):
            row0 = dict(cls=cls, param=param, family=spec["family"], n=n, m=g.num_edges, gidx=gi, scheme=sch,
                        graph_seed=spec["seed"], weight_seed=wseed)

            def extra(g_, j=j):
                return [(f"{m}@{s}", S, rt, "", "") for s in CKPT for m, S, rt in ob[s][j]]

            for r in check_rows(g, solve_all(cls, g, pts, a.ilp_time, a.threads, extra), row0):
                m, _, ms = r["method"].partition("@")
                rows.append({**r, "method": m, "model_seed": ms})
        for row in rows:
            wr.writerow(row)
        fh.flush()
        if (k + 1) % 10 == 0:
            print(f"  {k + 1}/{len(items)} ({cls}, n={n}) elapsed {time.time() - t_start:.0f}s", flush=True)
    fh.close()
    print(f"done in {time.time() - t_start:.0f}s -> {out}", flush=True)


if __name__ == "__main__":
    main()
