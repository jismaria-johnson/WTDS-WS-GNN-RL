"""Revision step 6: coverage fixes.

    python scripts/rev_coverage.py --part lb        # gaps to the CP-SAT lower bound on the 16 unproven grid instances
    python scripts/rev_coverage.py --part block     # blocking-rule ablation (no training)

lb: for every grid instance whose 30 s ILP was not proven optimal, gap_LB = (W(S) - LB) / LB with LB = CP-SAT's best
    bound (an upper bound on the true gap). Reported per method, and pooled over all 640 instances with the proven
    optimum where available and LB elsewhere. Uses the stored, already verified rows.
block: the base paper's rule (a picked vertex blocks its neighbours: they can no longer be picked). The trained policy
    (each model seed) picks argmax Q among unselected, unblocked vertices; forced vertices are selected at reset and also
    block their neighbours. The episode ends when every vertex is covered (a TDS: re-checked with
    is_total_dominating_set) or no legal action is left (failure). Main grid, n = 20-200, 640 instances.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REV = ROOT / "results" / "revision"


sys.path.insert(0, str(ROOT / "scripts"))
from labels import DISP  # noqa: E402

def part_lb():
    from stats import KEY, load_grid, tex_table, wide
    d, opt = load_grid()
    U, S = wide(d, opt)
    unp = ~opt["proven"]
    lb = opt["lb"].astype(float)
    ref_all = opt["ilp_w"].where(opt["proven"], lb)
    rows, out = [], {"n_unproven": int(unp.sum()), "instances": []}
    for k in opt[unp].index:
        out["instances"].append(dict(zip(KEY, map(str, k)), ilp_incumbent=float(opt.loc[k, "ilp_w"]),
                                     lb=float(lb[k])))
    meths = [(m, S[m]) for m in ("ours_rr", "ours_rr_ls")] + \
            [(m, U[m]) for m in ("greedy_rr", "greedy_rr_ls", "greedy_tiebreak_rr", "greedy_lookahead_rr",
                                 "lp_rounding_rr", "greedy_random_restart_rr") if m in U]
    meths.append(("ilp_incumbent_30s", opt["ilp_w"]))
    for m, W in meths:
        Wm = W.mean(axis=1) if isinstance(W, pd.DataFrame) else W
        g_unp = 100 * (Wm[unp] - lb[unp]) / lb[unp]
        g_all = 100 * (Wm - ref_all) / ref_all
        out[m] = dict(gap_lb_unproven=float(g_unp.mean()), gap_lb_unproven_max=float(g_unp.max()),
                      gap_all_640=float(g_all.mean()))
        rows.append([DISP.get(m, m), int(unp.sum()), f"{g_unp.mean():.2f}", f"{g_unp.max():.2f}", f"{g_all.mean():.2f}"])
    df = pd.DataFrame(rows, columns=["method", "unproven inst", "mean gap to LB %", "max gap to LB %",
                                     "all 640: gap to opt or LB %"])
    tex_table(df, REV / "tables" / "unproven_lb.tex",
              "The 16 grid instances whose 30 s ILP was not proven optimal: gap to the CP-SAT lower bound (an upper "
              "bound on the true gap), and the mean over all 640 instances using the optimum where proven and the "
              "lower bound otherwise. Proposed = mean over model seeds.", "tab:unproven_lb")
    (REV / "coverage_lb.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(df.to_string(index=False))


def part_block(seeds):
    import torch
    from run_eval import structures
    from wtdsws.checker import is_total_dominating_set
    from wtdsws.env import TDSEnv
    from wtdsws.generators import make_graph
    from wtdsws.rollout import q_values
    from wtdsws.solve import get_model
    from wtdsws.weights import SCHEMES, weighted_graph
    device = "cuda" if torch.cuda.is_available() else "cpu"
    items = [it for it in structures("grid", 10, (20, 50, 100, 200, 500)) if it[1] <= 200]
    res = []
    for s in seeds:
        ck = {0: "checkpoints/best.pt", 1: "checkpoints/seed1_best.pt"}.get(s, f"checkpoints/ws_seed{s}/best.pt")
        model = get_model(ROOT / ck, device)
        ek = {**dict(getattr(model, "env_kwargs", {}) or {}), "mask_mode": "notin"}
        for cls, n, gi, spec, wseed in items:
            base = make_graph(spec)
            for sch in SCHEMES:
                g = weighted_graph(base, sch, wseed)
                env = TDSEnv(g, **ek)
                blocked = np.zeros(g.n, bool)
                for f in np.flatnonzero(env.selected):
                    blocked[g.neighbors(f)] = True
                while not env.done:
                    legal = ~env.selected & ~blocked
                    if not legal.any():
                        break
                    q = q_values(model, [env], device)[0]
                    q = np.where(legal, q, -np.inf)
                    v = int(np.argmax(q))
                    env.step(v)
                    blocked[g.neighbors(v)] = True
                ok = bool(env.done) and is_total_dominating_set(g.adj, env.solution)
                if env.done:
                    assert ok
                res.append(dict(seed=s, cls=cls, scheme=sch, n=n, gidx=gi, valid=ok, steps=len(env.order),
                                covered=float(env.covered.mean())))
    df = pd.DataFrame(res)
    df.to_csv(REV / "blocking_ablation.csv", index=False)
    summ = dict(episodes=len(df), valid_pct=100 * df.valid.mean(), mean_covered_frac_at_stop=df.covered.mean(),
                seeds=seeds, per_scheme={k: 100 * v for k, v in df.groupby("scheme").valid.mean().items()})
    (REV / "blocking_ablation.json").write_text(json.dumps(summ, indent=1), encoding="utf-8")
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["lb", "block"], required=True)
    ap.add_argument("--seeds", default="0,1")
    a = ap.parse_args()
    part_lb() if a.part == "lb" else part_block([int(s) for s in a.seeds.split(",")])
