"""Revision step 1: weighted greedy + RR + the same 1 s local search on the exact-vs-ours graphs.

    python scripts/rev_greedy_ls_exact.py --study first   # the 240 graphs of results/exact_vs_ours.csv
    python scripts/rev_greedy_ls_exact.py --study final   # the 400 graphs of results/exact_vs_ours_final.csv

The graphs are rebuilt with the original scripts' own spec / build functions. Each rebuilt instance is checked against
the stored row (same m; greedy + RR re-run gives the stored W), so the new column is on exactly the same instances.
Every solution passes is_total_dominating_set and W(S) is recomputed. Output: results/revision/greedy_rr_ls_<study>.csv
"""
import argparse
import csv
import importlib.util
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from wtdsws.baselines import greedy_rr, greedy_rr_ls  # noqa: E402
from wtdsws.checker import is_total_dominating_set, verify  # noqa: E402
from wtdsws.exact_validation import WEIGHTINGS, with_scheme  # noqa: E402

COLS = ["cls", "param", "family", "n", "m", "gidx", "scheme", "graph_seed", "weight_seed", "method", "size", "weight",
        "runtime", "valid", "exact_weight", "exact_proven", "S"]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", choices=["first", "final"], required=True)
    ap.add_argument("--ls-time", type=float, default=1.0)
    a = ap.parse_args()
    if a.study == "first":
        mod = load("exact_vs_ours")
        items = [(cls, None, n, gi, spec, ws) for cls, n, gi, spec, ws in mod.specs()]
        ref = pd.read_csv(ROOT / "results" / "exact_vs_ours.csv")
    else:
        mod = load("exact_vs_ours_final")
        items = mod.final_specs()
        ref = pd.read_csv(ROOT / "results" / "exact_vs_ours_final.csv")
    ref["n"] = ref["n"].astype(int)
    ex = ref[ref.method == "exact"].set_index(["cls", "n", "gidx", "scheme"])
    gr = ref[ref.method == "greedy_rr"].set_index(["cls", "n", "gidx", "scheme"])
    out = ROOT / "results" / "revision" / f"greedy_rr_ls_{a.study}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    fh = open(out, "w", newline="")
    wr = csv.DictWriter(fh, fieldnames=COLS)
    wr.writeheader()
    t0, rows = time.time(), 0
    for k, (cls, param, n, gi, spec, wseed) in enumerate(items):
        base = mod.build(cls, spec)[0]
        for sch in WEIGHTINGS:
            g = with_scheme(base, sch, wseed)
            key = (cls, int(n), int(gi), sch)
            e, r0 = ex.loc[key], gr.loc[key]
            assert int(e["m"]) == g.num_edges and int(e["graph_seed"]) == spec["seed"], f"rebuild mismatch {key}"
            rg = greedy_rr(g)
            assert abs(float(g.w[rg.vertices].sum()) - float(r0["weight"])) < 1e-9, f"greedy_rr mismatch {key}"
            r = greedy_rr_ls(g, ls_time=a.ls_time, seed=0)
            S = sorted(set(int(v) for v in r.vertices))
            assert verify(g, S) and is_total_dominating_set(g.adj, S), f"INVALID {key}"
            W = float(g.w[S].sum())                              # recomputed from S and the weights
            assert W <= float(r0["weight"]) + 1e-9               # LS never worsens greedy + RR
            wr.writerow(dict(cls=cls, param=param or "", family=spec["family"], n=n, m=g.num_edges, gidx=gi,
                             scheme=sch, graph_seed=spec["seed"], weight_seed=wseed, method="greedy_rr_ls",
                             size=len(S), weight=W, runtime=r.runtime, valid=True,
                             exact_weight=float(e["exact_weight"]), exact_proven=bool(e["exact_proven"]),
                             S=" ".join(map(str, S))))
            rows += 1
        fh.flush()
        if (k + 1) % 20 == 0:
            print(f"  {k + 1}/{len(items)} graphs, {rows} rows, {time.time() - t0:.0f}s", flush=True)
    fh.close()
    print(f"[{a.study}] {rows} greedy_rr_ls rows, all valid, -> {out} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
