"""Revision step 4: equal wall-clock budgets per instance (0.16 s, 1 s, 5 s).

    python scripts/rev_equal_time.py --group cpu            # greedy restarts, greedy + RR + LS, LP + LS, CP-SAT
    python scripts/rev_equal_time.py --group ours --seeds 0,1,2,3,4 [--device cuda|cpu]

Instances: the main test grid at n = 100 and 200, graphs gidx 0-1 of every class, all four schemes (64 instances,
rebuilt from run_eval.structures and checked against results/raw_grid.csv). Gaps use the proven ILP optimum (30 s run).

Methods (every candidate is checked with both TDS checkers and W recomputed, run_eval.checked):
  ours_bok        greedy policy rollout + RR, then batches of 8 rollouts sampled from softmax(Q / T) (T = 0.03), each
                  + RR, until the budget is used; cheapest kept
  ours_bok_ls     the same, each candidate also gets LS limited to the time left
  grr / grr_ls    greedy_random_restart with the budget (without / with LS per restart)
  greedy_rr_ls    weighted greedy + RR + LS with LS limit = budget (stops at a local optimum)
  lp_rr_ls        LP rounding + RR + LS with LS limit = budget
  cpsat           CP-SAT, 8 workers, time limit = budget, no warm start: best solution found (none -> recorded)
"""
import argparse
import csv
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from run_eval import checked, structures  # noqa: E402
from wtdsws.baselines import greedy_rr_ls, ilp  # noqa: E402
from wtdsws.baselines.greedy_plus import greedy_random_restart, lp_rounding  # noqa: E402
from wtdsws.generators import make_graph  # noqa: E402
from wtdsws.weights import SCHEMES, weighted_graph  # noqa: E402

BUDGETS = (0.16, 1.0, 5.0)
TEMPERATURE = 0.03
COLS = ["cls", "scheme", "n", "gidx", "method", "model_seed", "budget", "weight", "size", "runtime", "candidates",
        "opt", "proven", "found"]


def ckpt(seed):
    return {0: "checkpoints/best.pt", 1: "checkpoints/seed1_best.pt"}.get(seed, f"checkpoints/ws_seed{seed}/best.pt")


def instances():
    ref = pd.read_csv(ROOT / "results" / "raw_grid.csv")
    il = ref[ref.method == "ilp"].set_index(["cls", "n", "gidx", "scheme"])
    out = []
    for cls, n, gi, spec, wseed in structures("grid", 10, (20, 50, 100, 200, 500)):
        if n not in (100, 200) or gi > 1:
            continue
        base = make_graph(spec)
        for sch in SCHEMES:
            g = weighted_graph(base, sch, wseed)
            r = il.loc[(cls, n, gi, sch)]
            assert int(r["m"]) == g.num_edges
            out.append((cls, sch, n, gi, g, float(r["weight"]), str(r["optimal"]) == "True"))
    return out


def ours_anytime(g, model, device, budget, ls, rng):
    from wtdsws.postprocess import local_search, redundancy_removal
    from wtdsws.rollout import rollout
    ek = dict(getattr(model, "env_kwargs", {}) or {})
    t0 = time.perf_counter()
    best, best_w, k = None, float("inf"), 0

    def consider(rs):
        nonlocal best, best_w, k
        for r in rs:
            S = redundancy_removal(g, r["solution"], scores=r["q"], protected=r["forced"])
            if ls:
                left = budget - (time.perf_counter() - t0)
                S = local_search(g, S, time_limit=max(min(1.0, left), 0.0), seed=k)
            S, W = checked(g, S, "ours_anytime")
            k += 1
            if W < best_w - 1e-12:
                best, best_w = S, W

    consider(rollout(model, [g], device, env_kwargs=ek, multi_select="auto"))
    while time.perf_counter() - t0 < budget:
        consider(rollout(model, [g] * 8, device, env_kwargs=ek, mode="sample", temperature=TEMPERATURE,
                         rng=rng, batch_size=8))
    return best, time.perf_counter() - t0, k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", choices=["cpu", "ours"], required=True)
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--device", default=None)
    ap.add_argument("--budgets", default=",".join(map(str, BUDGETS)))
    a = ap.parse_args()
    budgets = [float(b) for b in a.budgets.split(",")]
    tag = a.group if a.group == "cpu" else f"ours_{a.device or 'cuda'}"
    out = ROOT / "results" / "revision" / f"equal_time_{tag}.csv"
    done = set()
    if out.exists():
        done = {(r["cls"], r["scheme"], r["n"], r["gidx"], r["method"], r["model_seed"], r["budget"])
                for r in csv.DictReader(open(out, newline=""))}
    fh = open(out, "a", newline="")
    wr = csv.DictWriter(fh, fieldnames=COLS)
    if not done:
        wr.writeheader()
    items = instances()
    print(f"{len(items)} instances, budgets {budgets}", flush=True)
    t_start = time.time()
    models = {}
    if a.group == "ours":
        import torch
        from wtdsws.solve import get_model
        device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
        for s in map(int, a.seeds.split(",")):
            models[s] = get_model(ROOT / ckpt(s), device)
            ours_anytime(items[0][4], models[s], device, 0.05, False, np.random.default_rng(0))   # warm-up
    for i, (cls, sch, n, gi, g, opt, proven) in enumerate(items):
        base = dict(cls=cls, scheme=sch, n=n, gidx=gi, opt=opt, proven=proven)

        def emit(method, B, S, rt, cand=1, ms=""):
            if (cls, sch, str(n), str(gi), method, str(ms), str(B)) in done:
                return
            if S is None:
                wr.writerow({**base, "method": method, "model_seed": ms, "budget": B, "weight": "", "size": "",
                             "runtime": rt, "candidates": cand, "found": False})
                return
            S, W = checked(g, S, method)
            wr.writerow({**base, "method": method, "model_seed": ms, "budget": B, "weight": W, "size": len(S),
                         "runtime": rt, "candidates": cand, "found": True})

        def todo(method, B, ms=""):
            return (cls, sch, str(n), str(gi), method, str(ms), str(B)) not in done

        for B in budgets:
            if a.group == "cpu":
                if todo("grr", B):
                    r = greedy_random_restart(g, budget=B, seed=0)
                    emit("grr", B, r.vertices, r.runtime, r.info["restarts"])
                if todo("grr_ls", B):
                    r = greedy_random_restart(g, budget=B, ls_time=B, seed=0)
                    emit("grr_ls", B, r.vertices, r.runtime, r.info["restarts"])
                if todo("greedy_rr_ls", B):
                    r = greedy_rr_ls(g, ls_time=B, seed=0)
                    emit("greedy_rr_ls", B, r.vertices, r.runtime)
                if todo("lp_rr_ls", B):
                    r = lp_rounding(g, ls_time=B, seed=0)
                    emit("lp_rr_ls", B, r.vertices, r.runtime)
                if todo("cpsat", B):
                    t0 = time.perf_counter()
                    try:
                        r = ilp(g, time_limit=B, threads=8, warm_start=False)
                        emit("cpsat", B, r.vertices, r.runtime)
                    except RuntimeError:                      # no solution within the budget
                        emit("cpsat", B, None, time.perf_counter() - t0)
            else:
                for s, model in models.items():
                    for method, ls in (("ours_bok", False), ("ours_bok_ls", True)):
                        if todo(method, B, s):
                            S, rt, k = ours_anytime(g, model, device, B, ls, np.random.default_rng(1000 * s + i))
                            emit(method, B, S, rt, k, ms=s)
            fh.flush()
        print(f"  {i + 1}/{len(items)} instances, {time.time() - t_start:.0f}s", flush=True)
    fh.close()
    print(f"done -> {out} ({time.time() - t_start:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
