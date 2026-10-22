"""STEP 5 (optional) of the exact-vs-ours plot update: a STRESS TEST on trees built to be hard for weighted greedy.
Never mixed into the main plots or tables.

    python scripts/stress_test.py              # -> results/stress_test.csv, results/stress_test_summary.json,
                                               #    results/figures/exact_vs_ours/stress_test.{png,pdf}

Family "spider trees": h = max(1, n // 400) hubs on a path; the other vertices form legs (paths) of length
U{2..5}, each attached to a random hub. Degree weights make the hubs expensive (w = 100) and leg vertices cheap.
Chosen from a small exploration on PILOT-range seeds (docs/decisions.md): BA m = 1 trees with degree weights were
NOT hard for greedy (greedy exactly optimal on 6/6), spiders were (greedy gap ~6-7%).
Seeds: TEST range, master MASTER_SEEDS["test"] + 4, asserted unused by every earlier run.
Methods and checks as in scripts/exact_vs_ours_final.py (tree DP + lexicographic ILP cross-check, greedy + RR,
ours + RR and ours + RR + LS for both checkpoints; both TDS checkers, W recomputed).
"""
import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from exact_vs_ours import CKPT, ours  # noqa: E402
from exact_vs_ours_final import COLS, used_seeds  # noqa: E402
from pilot_sweep import check_rows, solve_all  # noqa: E402
from wtdsws.exact_validation import with_scheme  # noqa: E402
from wtdsws.graph import Graph  # noqa: E402
from wtdsws.seeds import MASTER_SEEDS, SEED_RANGES, split_of  # noqa: E402

MASTER = MASTER_SEEDS["test"] + 4
NS = (250, 500, 1000, 1500, 2000)
PER_CELL = 10
SCHEME = "degree"
RES = ROOT / "results"
FIG = RES / "figures" / "exact_vs_ours"
INK, TEAL, GREY = "#1F2328", "#0F766E", "#9CA3AF"
EPS = 1e-9


def spider_tree(n, seed):
    rng = np.random.default_rng(seed)
    h = max(1, n // 400)
    T = nx.Graph()
    T.add_nodes_from(range(h))
    nx.add_path(T, range(h))
    v = h
    while v < n:
        prev = int(rng.integers(h))
        for _ in range(int(rng.integers(2, 6))):
            if v >= n:
                break
            T.add_edge(prev, v)
            prev, v = v, v + 1
    assert nx.is_tree(T) and T.number_of_nodes() == n and min(d for _, d in T.degree()) > 0
    return T


def run():
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(MASTER)
    lo, hi = SEED_RANGES["test"]
    items = [(n, i, int(rng.integers(lo, hi)), int(rng.integers(lo, hi))) for n in NS for i in range(PER_CELL)]
    seeds = [s for _, _, a, b in items for s in (a, b)]
    assert all(split_of(s) == "test" for s in seeds)
    assert not set(seeds) & used_seeds("stress_test.csv"), "stress seeds reused from an earlier run"
    out = RES / "stress_test.csv"
    with open(out, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=COLS, extrasaction="ignore")
        wr.writeheader()
        for n, gi, gseed, wseed in items:
            g = with_scheme(Graph.from_networkx(spider_tree(n, gseed)), SCHEME, wseed)
            g.name = f"stress_spider_n{n}_g{gi}"
            row0 = dict(cls="stress_spider", param=f"h={max(1, n // 400)}", family="spider_tree", n=n,
                        m=g.num_edges, gidx=gi, scheme=SCHEME, graph_seed=gseed, weight_seed=wseed)

            def extra(g_):
                return [(f"{m}@{s}", S, rt, "", "") for s in CKPT for m, S, rt in ours(g_, s, device, 1.0)]

            for r in check_rows(g, solve_all("tree", g, None, 60.0, 8, extra), row0):
                m, _, ms = r["method"].partition("@")
                wr.writerow({**r, "method": m, "model_seed": ms})
            fh.flush()
    return out


def analyse():
    df = pd.read_csv(RES / "stress_test.csv")
    assert df.valid.astype(str).str.lower().eq("true").all()
    df["opt"] = (df.weight - df.exact_weight).abs() < EPS
    df["extra"] = df.weight - df.exact_weight
    df["ratio"] = df.weight / df.exact_weight
    df["gap"] = 100 * df.extra / df.exact_weight
    key = ["n", "gidx"]
    pg = {m: df[df.method == m].groupby(key)[["size", "weight", "extra", "ratio", "gap", "opt"]].mean()
          for m in ("exact", "ours_rr", "ours_rr_ls", "greedy_rr")}
    rows = []
    for n in NS:
        r = dict(n=n, graphs=PER_CELL)
        for m, d in pg.items():
            d = d.loc[n]
            r.update({f"W_{m}": d.weight.mean(), f"S_{m}": d["size"].mean(), f"gap_{m}_pct": d.gap.mean(),
                      f"ratio_{m}": d.ratio.mean(), f"opt_{m}_pct": 100 * d.opt.mean()})
        rows.append(r)
    t = pd.DataFrame(rows)
    t.round(4).to_csv(RES / "tables" / "stress_test.csv", index=False)
    chk = df[df.method == "ilp_check"]
    s = dict(label="STRESS TEST (adversarial for greedy) - not part of the main results", family=__doc__.split(
        "Family")[1].split("Chosen")[0].strip(), scheme=SCHEME, graphs=len(NS) * PER_CELL, solutions_checked=len(df),
        ilp_crosscheck_agree=int((chk.proven_optimal.astype(str).eq("True") & (chk.extra.abs() < EPS)
                                  & (chk["size"] == chk.exact_size)).sum()), ilp_crosscheck_runs=len(chk),
        per_n=t.round(4).to_dict("records"))
    (RES / "stress_test_summary.json").write_text(json.dumps(s, indent=2), encoding="utf-8")
    fig, axs = plt.subplots(1, 3, figsize=(18, 5.6))
    for ax, (col, lab) in zip(axs, (("weight", "Total weight W(S)"), ("extra", "Extra cost  W(S) − W_exact"),
                                    ("ratio", "W(S) / W_exact"))):
        for m, lab_m, c, mk, ls, dx in (("exact", "Exact (tree DP)", INK, "o", "-", 0),
                                        ("ours_rr", "Ours (GNN+RL + RR, 2 seeds)", TEAL, "s", "-", 12),
                                        ("greedy_rr", "Greedy + RR", GREY, "^", "--", -12)):
            g = pg[m][col].groupby(level=0)
            ax.errorbar(np.array(NS) + dx, g.mean().loc[list(NS)], yerr=g.std(ddof=0).loc[list(NS)], color=c, ls=ls,
                        lw=2, marker=mk, ms=7, capsize=3, label=lab_m,
                        markeredgecolor="white" if m != "exact" else INK)
        ax.set_xticks(NS)
        ax.set_xlabel("Number of nodes (n)")
        ax.set_ylabel(lab)
        ax.grid(True, color="#E5E7EB")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        lo_, hi_ = ax.get_ylim()
        ax.set_ylim(lo_ - 0.16 * (hi_ - lo_), hi_ + 0.3 * (hi_ - lo_))
    axs[0].legend(loc="upper left", frameon=False)
    o, gr = pg["ours_rr"], pg["greedy_rr"]
    axs[2].text(0.98, 0.03, f"Ours optimal in {100 * o.opt.mean():.0f}% | Greedy optimal in {100 * gr.opt.mean():.0f}%",
                transform=axs[2].transAxes, ha="right", va="bottom",
                bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#D1D5DB"))
    fig.suptitle("Stress test (adversarial for greedy) · spider trees, degree weights · NOT part of the main plots",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"stress_test.{ext}", dpi=300)
    plt.close(fig)
    print(t.round(3).to_string(index=False))
    return s


if __name__ == "__main__":
    t0 = time.time()
    if "--analyse-only" not in sys.argv:
        run()
    analyse()
    print(f"done in {time.time() - t0:.0f}s")
