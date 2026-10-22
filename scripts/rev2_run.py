"""Revision 2: anytime equal-time runs. One process, one method at a time (run on an otherwise idle machine).

    python scripts/rev2_run.py --step 1 [--methods ...] [--guided-seed1]   # n = 100, 200 main grid, traces up to 5 s
    python scripts/rev2_run.py --make-large                                # build + cache step-2 graphs, LP bounds
    python scripts/rev2_run.py --device-bench                              # CPU vs GPU, one rollout per large n
    python scripts/rev2_run.py --step 2 [--methods ...]                    # n = 500-5000, traces up to 10 s

Step 1 instances: results/raw_grid.csv test grid, n in {100, 200}, gidx 0-4, 4 classes x 4 schemes (160), rebuilt with
run_eval.structures (same seeds) and checked against the stored ILP rows (edge count). Opt = the stored 30 s ILP value.
Step 2 instances: fresh test-range seeds (RNG MASTER_SEEDS["test"] + 6000), classes er/ba/chordal/bipartite (random
subtype as the grid), n in {500, 1000, 2000, 5000}, 3 structures per (class, n), each with degree and random weights
(96 instances). Seeds asserted disjoint from every seed column of earlier results files.

Output (one row per instance x method, resumable): results/revision2/step{1,2}_traces.csv with the verified trace
[[t, W, |S|], ...] as JSON. Every set in every trace is checked with both TDS checkers and W recomputed (outside the
timed region). The GNN device per n is recorded (CPU for n <= 200; step 2 = the faster one in device_bench.json).
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

import rev2_anytime as A  # noqa: E402

OUT = ROOT / "results" / "revision2"
LARGE = ROOT / "data" / "rev2_large.npz"
COLS = ["step", "cls", "scheme", "n", "m", "gidx", "method", "model_seed", "device", "opt", "proven", "max_budget",
        "final_w", "final_t", "first_t", "wall", "trace_len", "extra", "trace"]
STEP1_METHODS = ["guided", "random_prior", "plain", "ours_bok_ls", "lp_ls", "cpsat"]
STEP2_METHODS = ["ours_greedy_ls", "ours_bok_ls", "guided", "plain", "random_prior", "greedy_ls", "lp_ls", "cpsat"]
CKPT = {0: "checkpoints/best.pt", 1: "checkpoints/seed1_best.pt"}
LARGE_NS = (500, 1000, 2000, 5000)


def step1_instances():
    import pandas as pd
    from run_eval import structures
    from wtdsws.generators import make_graph
    from wtdsws.weights import SCHEMES, weighted_graph
    ref = pd.read_csv(ROOT / "results" / "raw_grid.csv")
    il = ref[ref.method == "ilp"].set_index(["cls", "n", "gidx", "scheme"])
    out = []
    for cls, n, gi, spec, wseed in structures("grid", 10, (20, 50, 100, 200, 500)):
        if n not in (100, 200) or gi > 4:
            continue
        base = make_graph(spec)
        for sch in SCHEMES:
            g = weighted_graph(base, sch, wseed)
            r = il.loc[(cls, n, gi, sch)]
            assert int(r["m"]) == g.num_edges and int(r["graph_seed"]) == spec["seed"] and int(r["weight_seed"]) == wseed
            g.name = f"{cls}_n{n}_g{gi}_{sch}"
            out.append(dict(cls=cls, scheme=sch, n=n, gidx=gi, g=g, opt=float(r["weight"]),
                            proven=str(r["optimal"]) == "True"))
    assert len(out) == 160
    return out


def used_seeds():
    import pandas as pd
    seeds = set()
    for p in list((ROOT / "results").rglob("*.csv")):
        if "revision2" in p.parts:
            continue
        try:
            d = pd.read_csv(p, usecols=lambda c: c in ("graph_seed", "weight_seed", "seed"))
        except Exception:
            continue
        for c in d.columns:
            seeds |= set(pd.to_numeric(d[c], errors="coerce").dropna().astype(np.int64).tolist())
    return seeds


def large_specs():
    from wtdsws.graph_classes import CLASSES, class_spec
    from wtdsws.seeds import MASTER_SEEDS, SEED_RANGES
    rng = np.random.default_rng(MASTER_SEEDS["test"] + 6000)
    lo, hi = SEED_RANGES["test"]
    out = []
    for cls in CLASSES:
        for n in LARGE_NS:
            for gi in range(3):
                spec = class_spec(cls, n, rng)
                spec["seed"] = int(rng.integers(lo, hi))
                out.append((cls, n, gi, spec, int(rng.integers(lo, hi))))
    return out


def _build(cls, n, gi, spec, wseed):
    """One structure, both schemes, LP bounds (HiGHS interior point). Runs in a worker process."""
    from wtdsws.baselines.greedy_plus import lp_values
    from wtdsws.generators import make_graph
    from wtdsws.weights import weighted_graph
    t0 = time.time()
    base = make_graph(spec)
    out = []
    for sch in ("degree", "random"):
        g = weighted_graph(base, sch, wseed)
        g.name = f"{cls}_n{n}_g{gi}_{sch}"
        t1 = time.time()
        lp = float(g.w @ lp_values(g, method="highs-ipm")) * (1 - 1e-7)   # interior-point optimum; tiny margin
        out.append((g, dict(cls=cls, n=n, gidx=gi, scheme=sch, spec=spec, weight_seed=wseed, m=g.num_edges,
                            lp_bound=lp, lp_time=time.time() - t1)))
    return out, time.time() - t0


def make_large(workers=8, deadline=360.0):
    """Builds the 48 structures in parallel (no timed experiment runs meanwhile). A structure whose build does not
    finish within `deadline` s is replaced by a fresh spec of the same class and n from the replacement stream
    MASTER_SEEDS["test"] + 6002 (chordal: subtype drawn from {ktree, unit_interval}, see decisions R21); replacements
    are recorded in the json (field `replaced`). Built structures are cached in data/rev2_large_cache/."""
    from multiprocessing import Pool
    from wtdsws.graph_classes import class_spec
    from wtdsws.io import save_graphs
    from wtdsws.seeds import MASTER_SEEDS, SEED_RANGES, split_of
    specs = large_specs()
    used = used_seeds()
    lo, hi = SEED_RANGES["test"]
    rep_rng = np.random.default_rng(MASTER_SEEDS["test"] + 6002)
    cache = ROOT / "data" / "rev2_large_cache"
    cache.mkdir(exist_ok=True)
    todo = {(c, n, gi): (c, n, gi, sp, ws, None) for c, n, gi, sp, ws in specs}
    built, rounds = {}, 0
    while todo:
        rounds += 1
        pool = Pool(workers)
        jobs = {}
        for k, v in todo.items():
            f = cache / f"{k[0]}_{k[1]}_{k[2]}_{v[3]['seed']}.pkl"
            if f.exists():
                import pickle
                built[k] = (pickle.loads(f.read_bytes()), v[5])
                print(f"  {k} from cache", flush=True)
            else:
                jobs[k] = pool.apply_async(_build, v[:5])
        t_end = time.time() + deadline
        for k, job in jobs.items():
            try:
                res, dt = job.get(timeout=max(t_end - time.time(), 0.01))
                built[k] = (res, todo[k][5])
                import pickle
                (cache / f"{k[0]}_{k[1]}_{k[2]}_{todo[k][3]['seed']}.pkl").write_bytes(pickle.dumps(res))
                print(f"  {k} m={res[0][0].num_edges} ({dt:.1f}s)", flush=True)
            except Exception as e:                       # timeout (or failure): replaced below
                print(f"  {k} not built within {deadline:.0f}s ({type(e).__name__}): replaced", flush=True)
        pool.terminate()
        pool.join()
        nxt = {}
        for k in todo:
            if k in built:
                continue
            c, n, gi, sp, ws, prev = todo[k]
            sub = str(rep_rng.choice(["ktree", "unit_interval"])) if c == "chordal" else None   # R21: no chordal_gnp
            spec = class_spec(c, n, rep_rng, subtype=sub)
            spec["seed"] = int(rep_rng.integers(lo, hi))
            nxt[k] = (c, n, gi, spec, int(rep_rng.integers(lo, hi)), (prev or []) + [sp])
        todo = nxt
        assert rounds < 5
    graphs, meta = [], []
    for c, n, gi, *_ in specs:
        res, replaced = built[(c, n, gi)]
        for g, mt in res:
            mt["replaced"] = replaced
            graphs.append(g)
            meta.append(mt)
    new = [mt["spec"]["seed"] for mt in meta] + [mt["weight_seed"] for mt in meta]
    assert all(split_of(x) == "test" for x in new) and not (set(new) & used)
    assert len({(mt["spec"]["seed"], mt["weight_seed"]) for mt in meta}) == len(specs)
    print(f"{len(meta)} instances; seeds in test range, disjoint from {len(used)} earlier seeds", flush=True)
    save_graphs(graphs, LARGE)
    LARGE.with_suffix(".json").write_text(json.dumps(meta, indent=0))


def step2_instances():
    from wtdsws.io import load_graphs
    graphs = load_graphs(LARGE)
    meta = json.loads(LARGE.with_suffix(".json").read_text())
    out = []
    for g, mt in zip(graphs, meta):
        assert g.num_edges == mt["m"] and g.n == mt["n"]
        g.name = f"{mt['cls']}_n{mt['n']}_g{mt['gidx']}_{mt['scheme']}"
        out.append(dict(cls=mt["cls"], scheme=mt["scheme"], n=mt["n"], gidx=mt["gidx"], g=g, opt=mt["lp_bound"],
                        proven=False))
    assert len(out) == 96
    return out


def device_bench():
    """One greedy rollout (seed-0 model) per device on the first ER graph (degree weights) of each large n."""
    import torch
    from wtdsws.rollout import rollout
    from wtdsws.solve import get_model
    items = step2_instances()
    res = {"torch_threads": torch.get_num_threads(), "gpu": torch.cuda.get_device_name(0), "per_n": {}}
    for dev in ("cpu", "cuda"):
        model = get_model(ROOT / CKPT[0], dev)
        rollout(model, [items[0]["g"]], dev, multi_select="auto")                     # warm-up
    for n in LARGE_NS:
        g = next(it["g"] for it in items if it["n"] == n and it["cls"] == "er" and it["scheme"] == "degree")
        row = {}
        for dev in ("cpu", "cuda"):
            model = get_model(ROOT / CKPT[0], dev)
            ek = dict(getattr(model, "env_kwargs", {}) or {})
            t0 = time.perf_counter()
            r = rollout(model, [g], dev, env_kwargs=ek, multi_select="auto")[0]
            if dev == "cuda":
                torch.cuda.synchronize()
            row[dev] = time.perf_counter() - t0
            row[f"{dev}_W"] = float(g.w[r["solution"]].sum())
        row["chosen"] = min(("cpu", "cuda"), key=lambda d: row[d])
        res["per_n"][str(n)] = row
        print(n, row, flush=True)
    (OUT / "device_bench.json").write_text(json.dumps(res, indent=1))


def device_for(n):
    if n <= 200:
        return "cpu"
    return json.loads((OUT / "device_bench.json").read_text())["per_n"][str(n)]["chosen"]


def run(step, methods, guided_seed1=False, only_n=None, smoke=False):
    from wtdsws.solve import get_model
    tuned = json.loads((ROOT / "checkpoints" / "guided_restarts.json").read_text())
    params = {"guided": tuned["ours_guided_restarts"], "random_prior": tuned["random_prior_restarts"],
              "plain": tuned["plain_restarts"]}
    items = step1_instances() if step == 1 else step2_instances()
    budget = 5.0 if step == 1 else 10.0
    out = OUT / f"step{step}_traces.csv"
    if smoke:                                             # pipeline check: 2 instances, 0.3 s, separate file
        items, budget, out = items[:2], 0.3, OUT / "smoke" / f"step{step}_traces.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        done = {(r["cls"], r["scheme"], r["n"], r["gidx"], r["method"], r["model_seed"])
                for r in csv.DictReader(open(out, newline=""))}
    fh = open(out, "a", newline="")
    wr = csv.DictWriter(fh, fieldnames=COLS)
    if not done:
        wr.writeheader()
    models = {}

    def model_for(seed, dev):
        if (seed, dev) not in models:
            models[(seed, dev)] = get_model(ROOT / CKPT[seed], dev)
            A.ours_greedy_ls(items[0]["g"], models[(seed, dev)], dev, 0.01)               # warm-up (untimed)
        return models[(seed, dev)]

    warm, wdev = items[0]["g"], device_for(items[0]["n"])   # untimed warm-up of every code path (imports, caches)
    wm = model_for(0, wdev)
    A.restarts_run(warm, "gnn", wm, wdev, 0.05, params["guided"], 0)
    A.restarts_run(warm, "random", None, "cpu", 0.05, params["random_prior"], 0)
    A.ours_bok_ls(warm, wm, wdev, 0.05, np.random.default_rng(0))
    A.greedy_ls(warm, 0.05), A.lp_ls(warm, 0.05), A.cpsat(warm, 0.5)
    for i, it in enumerate(items):
        it["idx"] = i                                        # fixed instance index = restart / sampling seed
    if step == 1:                                            # cut order: gidx 4 last (cut 4 = 4 graphs per cell)
        items = sorted(items, key=lambda it: (it["gidx"] == 4, it["idx"]))
    else:                                                    # cut order: n = 5000 gidx 2 last, then random scheme
        items = sorted(items, key=lambda it: (it["n"] == 5000 and it["gidx"] == 2, it["scheme"] == "random",
                                              it["n"], it["idx"]))
    t_start = time.time()
    for it in items:
        i = it["idx"]
        if only_n and it["n"] not in only_n:
            continue
        g, n = it["g"], it["n"]
        dev = device_for(n)
        jobs = [(m, 0) for m in methods]
        if guided_seed1:                                     # seed-1 checkpoint on every second instance only
            jobs = [("guided", 1)] if i % 2 == 0 else []
        for method, ms in jobs:
            gnn = method in ("guided", "ours_bok_ls", "ours_greedy_ls")
            key = (it["cls"], it["scheme"], str(n), str(it["gidx"]), method, str(ms) if gnn else "")
            if key in done:
                continue
            model = model_for(ms, dev) if gnn else None
            t0 = time.perf_counter()
            if method == "guided":
                tr, ex = A.restarts_run(g, "gnn", model, dev, budget, params["guided"], seed=i)
            elif method == "random_prior":
                tr, ex = A.restarts_run(g, "random", None, dev, budget, params["random_prior"], seed=i)
            elif method == "plain":
                tr, ex = A.restarts_run(g, "none", None, dev, budget, params["plain"], seed=i)
            elif method == "ours_bok_ls":
                tr, ex = A.ours_bok_ls(g, model, dev, budget, np.random.default_rng(1000 * ms + i))
            elif method == "ours_greedy_ls":
                tr, ex = A.ours_greedy_ls(g, model, dev, budget)
            elif method == "greedy_ls":
                tr, ex = A.greedy_ls(g, budget)
            elif method == "lp_ls":
                tr, ex = A.lp_ls(g, budget)
            elif method == "cpsat":
                tr, ex = A.cpsat(g, budget)
            else:
                raise ValueError(method)
            wall = time.perf_counter() - t0
            vt = A.verify_trace(g, tr, method)                       # both checkers + W recomputed, untimed
            if it["proven"] or step == 2:
                assert not vt or vt[-1][1] >= it["opt"] - 1e-6, f"{method} below the optimum / LP bound"
            wr.writerow(dict(step=step, cls=it["cls"], scheme=it["scheme"], n=n, m=g.num_edges, gidx=it["gidx"],
                             method=method, model_seed=ms if gnn else "", device=dev if gnn else "cpu",
                             opt=it["opt"], proven=it["proven"], max_budget=budget,
                             final_w=vt[-1][1] if vt else "", final_t=vt[-1][0] if vt else "",
                             first_t=vt[0][0] if vt else "", wall=wall, trace_len=len(vt), extra=json.dumps(ex),
                             trace=json.dumps([[round(t, 5), W, s] for t, W, s in vt])))
            fh.flush()
        print(f"  [{i}] {g.name} done ({time.time() - t_start:.0f}s)", flush=True)
    fh.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=int, choices=[1, 2])
    ap.add_argument("--methods", default=None)
    ap.add_argument("--guided-seed1", action="store_true")
    ap.add_argument("--only-n", default=None)
    ap.add_argument("--make-large", action="store_true")
    ap.add_argument("--device-bench", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.make_large:
        make_large()
    elif a.device_bench:
        device_bench()
    else:
        ms = a.methods.split(",") if a.methods else (STEP1_METHODS if a.step == 1 else STEP2_METHODS)
        run(a.step, ms, a.guided_seed1, [int(x) for x in a.only_n.split(",")] if a.only_n else None, a.smoke)
