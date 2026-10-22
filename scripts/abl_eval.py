"""Ablation study: evaluate one variant (reference or A1-A5) on the main test grid.

    python scripts/abl_eval.py --variant ref|a1|a2|a3|a4|a5 [--seeds 0,1] [--ns 20,50,100,200]

Instances: results/raw_grid.csv test grid, n in {20, 50, 100, 200}, 10 graphs per class x n, 4 schemes = 640
(rebuilt with run_eval.structures, same seeds; edge count, graph seed and weight seed checked against the ILP rows).
Reference value per instance: the proven ILP optimum, or CP-SAT's lower bound when not proven (decisions R28).
Per (instance, model seed): greedy policy rollout (exactly as the grid: multi_select="auto"), raw set and RR set, both
checked with both TDS checkers, W recomputed. Time = rollout + RR (no verification), CPU (faster than the GPU for
n <= 200, revision 1). Extra: A1 wasted picks (gain 0 when picked); A2 raw sets missing a degree-1-forced vertex and
policy picks spent on such vertices. Output: results/ablations/eval_<variant>.csv (one row per instance x seed).
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

OUT = ROOT / "results" / "ablations"
NAME = {"a1": "abl_a1", "a2": "abl_a2", "a3": "abl_a3", "a4": "abl_a4", "a5": "abl_a5"}
COLS = ["variant", "model_seed", "cls", "scheme", "n", "gidx", "ref_w", "proven", "w_raw", "size_raw", "w_rr",
        "size_rr", "time", "wasted_picks", "forced_missing", "forced_picked", "picks"]


def ckpt(variant, seed):
    if variant == "ref":
        return ROOT / {0: "checkpoints/best.pt", 1: "checkpoints/seed1_best.pt"}[seed]
    return ROOT / "checkpoints" / "ablations" / f"{NAME[variant]}_seed{seed}" / "best.pt"


def instances(ns):
    import pandas as pd
    from run_eval import structures
    from wtdsws.generators import make_graph
    from wtdsws.weights import SCHEMES, weighted_graph
    ref = pd.read_csv(ROOT / "results" / "raw_grid.csv")
    il = ref[ref.method == "ilp"].set_index(["cls", "n", "gidx", "scheme"])
    out = []
    for cls, n, gi, spec, wseed in structures("grid", 10, (20, 50, 100, 200, 500)):
        if n not in ns:
            continue
        base = make_graph(spec)
        for sch in SCHEMES:
            g = weighted_graph(base, sch, wseed)
            r = il.loc[(cls, n, gi, sch)]
            assert int(r["m"]) == g.num_edges and int(r["graph_seed"]) == spec["seed"] and int(r["weight_seed"]) == wseed
            proven = str(r["optimal"]) == "True"
            out.append(dict(cls=cls, scheme=sch, n=n, gidx=gi, g=g, proven=proven,
                            ref_w=float(r["weight"]) if proven else float(r["bound"])))
    return out


def wasted_and_forced(g, r, use_forced_ref):
    """Wasted picks: policy picks whose open neighbourhood had no uncovered vertex at pick time.
    forced_missing: some degree-1-forced vertex is absent from the raw set (impossible for a TDS; measured anyway).
    forced_picked: policy picks (not pre-selections) that are degree-1-forced vertices."""
    from wtdsws.reductions import forced_vertices
    covered = np.zeros(g.n, bool)
    for v in r["forced"]:
        covered[g.neighbors(v)] = True
    wasted = 0
    for v in r["order"]:
        nb = g.neighbors(v)
        if covered[nb].all():
            wasted += 1
        covered[nb] = True
    fv = set(forced_vertices(g).tolist())
    S = set(r["solution"])
    return wasted, int(not fv <= S), sum(1 for v in r["order"] if v in fv)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True)
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--ns", default="20,50,100,200")
    ap.add_argument("--limit", type=int, default=0, help="first k instances only (smoke)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    from run_eval import checked
    from wtdsws.postprocess import redundancy_removal
    from wtdsws.rollout import rollout
    from wtdsws.solve import get_model
    items = instances(tuple(int(x) for x in a.ns.split(",")))
    if a.limit:
        items = items[:a.limit]
    out = Path(a.out) if a.out else OUT / f"eval_{a.variant}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    fh = open(out, "w", newline="")
    wr = csv.DictWriter(fh, fieldnames=COLS)
    wr.writeheader()
    t_all = time.time()
    for s in map(int, a.seeds.split(",")):
        p = ckpt(a.variant, s)
        if not p.exists():
            print(f"skip seed {s}: {p} missing", flush=True)
            continue
        model = get_model(p, "cpu")
        ek = dict(getattr(model, "env_kwargs", {}) or {})
        rollout(model, [items[0]["g"]], "cpu", env_kwargs=ek, multi_select="auto")            # warm-up
        for it in items:
            g = it["g"]
            t0 = time.perf_counter()
            r = rollout(model, [g], "cpu", env_kwargs=ek, multi_select="auto")[0]
            S_rr = redundancy_removal(g, r["solution"], scores=r["q"], protected=r["forced"], check=False)
            dt = time.perf_counter() - t0
            S_raw, W_raw = checked(g, r["solution"], f"{a.variant} raw")       # both checkers, W recomputed
            S_rr, W_rr = checked(g, S_rr, f"{a.variant} rr")
            if it["proven"]:
                assert W_rr >= it["ref_w"] - 1e-6 and W_raw >= it["ref_w"] - 1e-6, "below a proven optimum"
            wasted, fmiss, fpick = wasted_and_forced(g, r, True)
            wr.writerow(dict(variant=a.variant, model_seed=s, cls=it["cls"], scheme=it["scheme"], n=it["n"],
                             gidx=it["gidx"], ref_w=it["ref_w"], proven=it["proven"], w_raw=W_raw,
                             size_raw=len(S_raw), w_rr=W_rr, size_rr=len(S_rr), time=dt, wasted_picks=wasted,
                             forced_missing=fmiss, forced_picked=fpick, picks=len(r["order"])))
        fh.flush()
        print(f"[{a.variant}] seed {s}: {len(items)} instances ({time.time() - t_all:.0f}s); env {json.dumps(ek)}",
              flush=True)
    fh.close()


if __name__ == "__main__":
    main()
