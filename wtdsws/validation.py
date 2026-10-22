"""Fixed validation set with cached ILP optima, and policy evaluation helpers."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from .baselines import greedy_rr, ilp
from .checker import assert_tds
from .env import TDSEnv
from .generators import GraphSampler
from .io import load_graphs, save_graphs
from .postprocess import redundancy_removal

VAL_SEED = 888_000_888        # disjoint from training seeds (0..) and test-set seeds


def get_validation_set(path="data/val_set.npz", count=200, n_min=50, n_max=100,
                       seed=VAL_SEED, ilp_time=60.0, verbose=True):
    """Returns (graphs, info dict with ILP optimum/proof status, greedy_rr and random baselines)."""
    path = Path(path)
    jpath = path.with_suffix(".json")
    if path.exists() and jpath.exists():
        return load_graphs(path), json.loads(jpath.read_text())
    sampler = GraphSampler(seed, weighted=True)        # TRAIN_WEIGHT_MIX distributions
    graphs = [sampler.sample(n_min, n_max) for _ in range(count)]
    opt, proven, t_ilp = [], [], time.time()
    for i, g in enumerate(graphs):
        r = ilp(g, time_limit=ilp_time)
        opt.append(r.weight)
        proven.append(bool(r.info["optimal"]))
        if verbose and (i + 1) % 20 == 0:
            print(f"  val ILP {i + 1}/{count}  proven so far {sum(proven)}  ({time.time() - t_ilp:.0f}s)", flush=True)
    grr = [greedy_rr(g).weight for g in graphs]        # weighted greedy + RR, W(S)
    rnd = [random_policy_size(g, np.random.default_rng(i)) for i, g in enumerate(graphs)]
    info = dict(opt=opt, proven=proven, greedy_rr=grr, random_rr=rnd,
                specs=[g.meta for g in graphs], ilp_time=ilp_time, seed=seed,
                created=time.strftime("%Y-%m-%d %H:%M:%S"))
    save_graphs(graphs, path)
    jpath.write_text(json.dumps(info))
    return graphs, info


def random_policy_size(g, rng, rr=True, env_kwargs=None) -> int:
    env = TDSEnv(g, **(env_kwargs or {}))
    while not env.done:
        env.step(int(rng.choice(np.flatnonzero(env.valid_mask()))))
    S = env.solution
    if rr:
        S = redundancy_removal(g, S)
    return float(g.w[S].sum())


def evaluate_policy(model, graphs, opt, device, env_kwargs=None, batch_size=50):
    """Greedy rollout on every graph; returns mean optimality gap before/after RR."""
    from .rollout import rollout
    t0 = time.time()
    res = rollout(model, graphs, device, env_kwargs=env_kwargs, batch_size=batch_size)
    raw, rr = [], []
    for g, r, o in zip(graphs, res, opt):
        assert_tds(g, r["solution"], "validation rollout")
        S = redundancy_removal(g, r["solution"], scores=r["q"], protected=r["forced"])
        raw.append(float(g.w[r["solution"]].sum()))     # W(S), recomputed from the solution
        rr.append(float(g.w[S].sum()))
    opt = np.asarray(opt, dtype=float)
    raw, rr = np.asarray(raw, float), np.asarray(rr, float)
    return dict(gap_raw=float(np.mean((raw - opt) / opt)), gap_rr=float(np.mean((rr - opt) / opt)),
                size_raw=float(raw.mean()), size_rr=float(rr.mean()),
                frac_opt_rr=float(np.mean(rr == opt)), time=time.time() - t0)


# ----------------------------------------------------------------- wtds-ws validation set
def get_ws_validation_set(path="data/val_ws.npz", per_cell=10, n_min=50, n_max=100, ilp_time=30.0,
                          threads=4, verbose=True):
    """160 graphs = `per_cell` per (class x scheme), n ~ U[n_min, n_max], seeds from the validation range.

    Every graph keeps its ILP optimum only if CP-SAT PROVES it within `ilp_time`; a graph whose optimum is not
    proven is replaced by a fresh sample of the same cell (the number replaced is recorded in the info file).
    """
    from .graph_classes import CLASSES, ClassSchemeSampler
    from .seeds import MASTER_SEEDS, SEED_RANGES
    from .weights import SCHEMES
    path = Path(path)
    jpath = path.with_suffix(".json")
    if path.exists() and jpath.exists():
        return load_graphs(path), json.loads(jpath.read_text())
    sampler = ClassSchemeSampler(MASTER_SEEDS["val"], seed_range=SEED_RANGES["val"])
    graphs, opt, ilp_t, replaced, t0 = [], [], [], 0, time.time()
    for cls in CLASSES:
        for scheme in SCHEMES:
            k = 0
            while k < per_cell:
                g = sampler.sample(n_min, n_max, cls=cls, scheme=scheme)
                r = ilp(g, time_limit=ilp_time, threads=threads)
                if not r.info["optimal"]:
                    replaced += 1
                    continue
                graphs.append(g)
                opt.append(r.weight)
                ilp_t.append(r.runtime)
                k += 1
            if verbose:
                print(f"  val {cls:<9} {scheme:<8} done: {len(graphs)} graphs, {replaced} replaced "
                      f"({time.time() - t0:.0f}s)", flush=True)
    grr = [greedy_rr(g).weight for g in graphs]
    rnd = [random_policy_size(g, np.random.default_rng(i)) for i, g in enumerate(graphs)]
    info = dict(opt=opt, proven=[True] * len(graphs), greedy_rr=grr, random_rr=rnd, ilp_runtime=ilp_t,
                replaced_not_proven=replaced, cls=[g.meta["cls"] for g in graphs],
                scheme=[g.meta["scheme"] for g in graphs], specs=[g.meta for g in graphs],
                ilp_time=ilp_time, seed=MASTER_SEEDS["val"], created=time.strftime("%Y-%m-%d %H:%M:%S"))
    save_graphs(graphs, path)
    jpath.write_text(json.dumps(info))
    return graphs, info
