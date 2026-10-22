"""Lattice experiment (base paper: grid, triangular, hexagonal lattices; G(m, k) grids vs known optima).

    python scripts/lat_run.py [--no-ls-min-n 400] [--mixed-seeds 3] [--skip-tri900]

Instances
  gmk  rectangular grids G(m, k), 4 <= m <= 10, m <= k <= 10, unit weights (28 graphs)
  lat  grid / tri / hex lattices (existing generators: generators.lattice_dims + make_graph), n ~ 25, 64, 100, 225,
       400, 900; weightings unit, degree, inverse (1 instance each), random and mixed (3 weight seeds each, fresh test
       range, stream MASTER_SEEDS["test"] + 7000, asserted unused by every earlier results file)
  Vertex and edge counts are asserted: n = r c (grid) or generators._lattice_count; m from Euler's formula for the
  plane lattice patch, m = n + faces - 1 with faces = (r-1)(c-1) (grid), a b triangles (tri), a b hexagons (hex).
Methods (per instance): ours_rr and ours_rr_ls (1 s LS) for the 5 main-model seeds, greedy_rr, greedy_rr_ls (1 s),
  ilp = CP-SAT 60 s, 8 workers, greedy warm start (baselines.ilp defaults): proven flag and lower bound recorded.
  gmk: ours_rr, greedy_rr and ilp only. GNN device: CPU for n <= 200, CUDA above (revision measurements).
Closed form (unit square grids): gamma_t(n x n) = (2a+1)(2a+r), n = 4a + r; asserted equal to CP-SAT where proven.
Every set is checked with both TDS checkers and W recomputed (run_eval.checked). Output: results/lattices/raw.csv.
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

OUT = ROOT / "results" / "lattices"
TARGETS = (25, 64, 100, 225, 400, 900)
KINDS = ("grid", "tri", "hex")
SEEDS = (0, 1, 2, 3, 4)
COLS = ["part", "kind", "rows", "cols", "n_target", "n", "m", "scheme", "wseed_idx", "weight_seed", "method",
        "model_seed", "device", "weight", "size", "runtime", "proven", "bound", "closed_form"]


def ckpt(s):
    return ROOT / {0: "checkpoints/best.pt", 1: "checkpoints/seed1_best.pt"}.get(s, f"checkpoints/ws_seed{s}/best.pt")


def gt_grid(n):
    a, r = divmod(n, 4)
    return (2 * a + 1) * (2 * a + r)


def base_lattice(kind, a, b):
    from wtdsws.generators import _lattice_count, make_graph
    g = make_graph(dict(family=kind, rows=a, cols=b))
    if kind == "grid":
        n_exp, faces = a * b, (a - 1) * (b - 1)
    else:
        n_exp, faces = _lattice_count(kind, a, b), a * b
    assert g.n == n_exp, (kind, a, b, g.n, n_exp)
    assert g.num_edges == g.n + faces - 1, (kind, a, b, g.num_edges, g.n + faces - 1)
    assert g.degrees.min() >= 1
    return g


def weight_seeds():
    from rev2_run import used_seeds
    from wtdsws.seeds import MASTER_SEEDS, SEED_RANGES, split_of
    rng = np.random.default_rng(MASTER_SEEDS["test"] + 7000)
    lo, hi = SEED_RANGES["test"]
    out = {(k, t, sch, i): int(rng.integers(lo, hi)) for k in KINDS for t in TARGETS for sch in ("random", "mixed")
           for i in range(3)}
    used = used_seeds() | lattice_free_seeds()
    assert all(split_of(s) == "test" for s in out.values()) and not (set(out.values()) & used)
    assert len(set(out.values())) == len(out)
    return out


def lattice_free_seeds():
    """Seeds of revision-2 / ablation files (not scanned by used_seeds) and the step-2 graphs."""
    s = set()
    p = ROOT / "data" / "rev2_large.json"
    if p.exists():
        for mt in json.loads(p.read_text()):
            s |= {int(mt["spec"]["seed"]), int(mt["weight_seed"])}
    return s


def instances(mixed_seeds=3, skip_tri900=False):
    from wtdsws.exact_validation import with_scheme
    from wtdsws.generators import lattice_dims
    items = []
    for m in range(4, 11):
        for k in range(m, 11):
            g = with_scheme(base_lattice("grid", m, k), "unit", None)
            items.append(dict(part="gmk", kind="grid", rows=m, cols=k, n_target=m * k, scheme="unit", wseed_idx="",
                              weight_seed="", g=g, order=(0, 0, m * k)))
    ws = weight_seeds()
    for kind in KINDS:
        for t in TARGETS:
            a, b = lattice_dims(kind, t)
            base = base_lattice(kind, a, b)
            for sch in ("unit", "degree", "inverse", "random", "mixed"):
                for i in (range(3) if sch in ("random", "mixed") else [None]):
                    if sch == "mixed" and i >= mixed_seeds:
                        continue
                    seed = ws[(kind, t, sch, i)] if i is not None else None
                    g = with_scheme(base, sch, seed)
                    last = kind == "tri" and t == 900
                    if last and skip_tri900:
                        continue
                    items.append(dict(part="lat", kind=kind, rows=a, cols=b, n_target=t, scheme=sch,
                                      wseed_idx="" if i is None else i, weight_seed="" if seed is None else seed,
                                      g=g, order=(1, int(last) * 2 + int(sch == "mixed" and i == 2), base.n)))
    items.sort(key=lambda it: it["order"])
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-ls-min-n", type=int, default=0, help="cut (3): skip LS variants for n >= this")
    ap.add_argument("--mixed-seeds", type=int, default=3)
    ap.add_argument("--skip-tri900", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    from run_eval import checked
    from wtdsws.baselines import greedy_rr, greedy_rr_ls, ilp
    from wtdsws.postprocess import local_search, redundancy_removal
    from wtdsws.rollout import rollout
    from wtdsws.solve import get_model
    items = instances(a.mixed_seeds, a.skip_tri900)
    if a.limit:
        items = items[:a.limit]
    out = Path(a.out) if a.out else OUT / "raw.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        done = {(r["part"], r["kind"], r["rows"], r["cols"], r["scheme"], r["wseed_idx"], r["method"], r["model_seed"])
                for r in csv.DictReader(open(out, newline=""))}
    fh = open(out, "a", newline="")
    wr = csv.DictWriter(fh, fieldnames=COLS)
    if not done:
        wr.writeheader()
    models = {}
    for dev in ("cpu", "cuda"):
        for s in SEEDS:
            models[(s, dev)] = get_model(ckpt(s), dev)
            rollout(models[(s, dev)], [items[0]["g"]], dev, multi_select="auto")           # warm-up
    t_start = time.time()
    for idx, it in enumerate(items):
        g = it["g"]
        dev = "cpu" if g.n <= 200 else "cuda"
        row0 = {k: it[k] for k in ("part", "kind", "rows", "cols", "n_target", "scheme", "wseed_idx", "weight_seed")}
        row0.update(n=g.n, m=g.num_edges)
        do_ls = it["part"] == "lat" and not (a.no_ls_min_n and g.n >= a.no_ls_min_n)

        def todo(method, ms=""):
            return (it["part"], it["kind"], str(it["rows"]), str(it["cols"]), it["scheme"], str(it["wseed_idx"]),
                    method, str(ms)) not in done

        def emit(method, S, rt, ms="", device="cpu", proven="", bound="", cf=""):
            S, W = checked(g, S, method)                       # both checkers, W recomputed
            wr.writerow({**row0, "method": method, "model_seed": ms, "device": device, "weight": W, "size": len(S),
                         "runtime": rt, "proven": proven, "bound": bound, "closed_form": cf})
            return W

        if todo("ilp"):
            r = ilp(g, time_limit=60.0, threads=8)
            cf = gt_grid(it["rows"]) if (it["kind"] == "grid" and it["rows"] == it["cols"] and it["scheme"] == "unit") else ""
            if cf != "" and r.info["optimal"]:
                assert r.size == cf, f"closed form mismatch {it['rows']}x{it['cols']}: CP-SAT {r.size} vs {cf}"
            W_opt = emit("ilp", r.vertices, r.runtime, proven=bool(r.info["optimal"]), bound=float(r.info["bound"]), cf=cf)
        for s in SEEDS:
            if todo("ours_rr", s):
                model = models[(s, dev)]
                t0 = time.perf_counter()
                rr_ = rollout(model, [g], dev, env_kwargs=dict(model.env_kwargs or {}), multi_select="auto")[0]
                S = redundancy_removal(g, rr_["solution"], scores=rr_["q"], protected=rr_["forced"], check=False)
                t_rr = time.perf_counter() - t0
                emit("ours_rr", S, t_rr, s, dev)
                if do_ls:
                    S2 = local_search(g, S, time_limit=1.0, seed=s)
                    emit("ours_rr_ls", S2, time.perf_counter() - t0, s, dev)
        if todo("greedy_rr"):
            r = greedy_rr(g)
            emit("greedy_rr", r.vertices, r.runtime)
            if do_ls:
                r = greedy_rr_ls(g, ls_time=1.0, seed=0)
                emit("greedy_rr_ls", r.vertices, r.runtime)
        fh.flush()
        print(f"  [{idx + 1}/{len(items)}] {it['part']} {it['kind']} {it['rows']}x{it['cols']} n={g.n} {it['scheme']}"
              f"{it['wseed_idx']} ({time.time() - t_start:.0f}s)", flush=True)
    fh.close()


if __name__ == "__main__":
    main()
