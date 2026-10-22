"""STEPS 1-2 of the exact-vs-ours plot update: summarise the pilot sweep and choose the plot settings by rule.

    python scripts/pilot_choose.py

Reads results/pilot_sweep.csv (scripts/pilot_sweep.py) and writes
  results/pilot_sweep_per_graph.csv     one row per (setting, n, graph, weighting)
  results/pilot_sweep_settings.csv      one row per (class, param, grid, n, weighting): means of the gaps / differences
  results/pilot_sweep_choice.csv        one row per (class, param, grid): the selection score and the constraints
  results/chosen_settings.json          the chosen (param, grid, weighting) per class
  results/figures/exact_vs_ours/pilot_separation_heatmap.{png,pdf}

Definitions (per graph and weighting, exact optimum proven):
  gap_greedy % = 100 (W_greedy - W_exact) / W_exact,  gap_ours % likewise (ours = ours + RR, best.pt)
  SEPARATION   = gap_greedy - gap_ours   (percentage points; > 0 means ours is closer to the optimum)
  dW = W_greedy - W_ours,  dS = |S|_greedy - |S|_ours

Rule (per class):
  1. candidates = (param, grid); a candidate is eligible iff every ours solution is valid and the exact method is
     proven optimal on >= 95% of its instances;
  2. score = mean SEPARATION over all proven instances of the candidate (its n values x 5 graphs x 5 weightings);
     the eligible candidate with the largest score is chosen (ties: larger grid B, then the parameter listed first);
  3. weighting for Plots 2-5 = the non-unit weighting with the largest mean SEPARATION within the chosen candidate.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from pilot_sweep import GRID_A, GRID_B, SETTINGS  # noqa: E402

RES = ROOT / "results"
FIG = RES / "figures" / "exact_vs_ours"
GRIDS = {"A": GRID_A, "B": GRID_B}
CLASSES = ("tree", "unit_interval", "convex_bip", "two_tree")
TITLE = {"tree": "Trees", "unit_interval": "Unit-interval", "convex_bip": "Convex bipartite", "two_tree": "2-trees"}
PARAM_LABEL = {"random_tree": "random labelled", "ba_m1": "BA m=1", "deg3": "avg deg 3", "deg6": "avg deg 6",
               "deg10": "avg deg 10", "k2": "as generated"}
WEIGHTINGS = ("unit", "degree", "inverse", "random", "mixed")
MIN_PROVEN = 0.95


def to_bool(s):
    return s.astype(str).str.lower().eq("true")


def per_graph(df):
    key = ["cls", "param", "n", "gidx", "scheme"]
    ex = df[df.method == "exact"].set_index(key)
    gr = df[df.method == "greedy_rr"].set_index(key)
    o = df[df.method == "ours_rr"].set_index(key)
    pg = pd.DataFrame({"proven": ex["proven_optimal"], "S_exact": ex["size"], "W_exact": ex["weight"],
                       "S_greedy": gr["size"], "W_greedy": gr["weight"], "S_ours": o["size"], "W_ours": o["weight"],
                       "ours_valid": o["valid"]}).reset_index()
    pg["gap_greedy"] = 100 * (pg.W_greedy - pg.W_exact) / pg.W_exact
    pg["gap_ours"] = 100 * (pg.W_ours - pg.W_exact) / pg.W_exact
    pg["separation"] = pg.gap_greedy - pg.gap_ours
    pg["dW"] = pg.W_greedy - pg.W_ours
    pg["dS"] = pg.S_greedy - pg.S_ours
    pg["extra_W_ours"] = pg.W_ours - pg.W_exact
    pg["extra_W_greedy"] = pg.W_greedy - pg.W_exact
    pg["opt_ours"] = (pg.W_ours - pg.W_exact).abs() < 1e-9
    pg["opt_greedy"] = (pg.W_greedy - pg.W_exact).abs() < 1e-9
    return pg


def with_grid(pg):
    parts = []
    for gname, ns in GRIDS.items():
        d = pg[pg.n.isin(ns)].copy()
        d["grid"] = gname
        parts.append(d)
    return pd.concat(parts, ignore_index=True)


def merge_workers():
    """The pilot ran as one main process plus parallel workers (--only, results/pilot_sweep_w*.csv). Merge them into
    results/pilot_sweep.csv; a (setting, n, graph, weighting, method) row present twice (the main process was stopped
    after it began a worker's setting) keeps the worker's copy. The worker files are kept."""
    parts = [pd.read_csv(p) for p in sorted(RES.glob("pilot_sweep_w*.csv")) if p.stat().st_size > 0]
    if not parts:
        return
    raw = RES / "pilot_sweep_main_raw.csv"                  # the main process's file, kept as written
    if not raw.exists():
        raw.write_bytes((RES / "pilot_sweep.csv").read_bytes())
    main_ = pd.read_csv(raw)
    key = ["cls", "param", "n", "gidx", "scheme", "method"]
    df = pd.concat([main_] + parts, ignore_index=True)
    df = df[df.cls != "cls"].copy()                         # a header line repeated by a restarted worker
    for c in ("n", "m", "gidx", "graph_seed", "weight_seed", "size", "weight", "runtime", "exact_weight", "exact_size"):
        df[c] = pd.to_numeric(df[c])
    for c in ("valid", "proven_optimal", "exact_proven"):
        df[c] = df[c].astype(str)
    df = df.drop_duplicates(key, keep="last")
    order = {s: i for i, s in enumerate(SETTINGS)}
    df["_o"] = [order[(c, p)] for c, p in zip(df.cls, df.param)]
    df.sort_values(["_o", "n", "gidx"], kind="stable").drop(columns="_o").to_csv(RES / "pilot_sweep.csv", index=False)


def main():
    merge_workers()
    df = pd.read_csv(RES / "pilot_sweep.csv")
    for c in ("valid", "proven_optimal", "exact_proven"):
        df[c] = to_bool(df[c])
    assert df.valid.all(), "an invalid solution in the pilot"
    expected = {(c, p) for c, p in SETTINGS}
    got = set(map(tuple, df[["cls", "param"]].drop_duplicates().values))
    assert got == expected, f"pilot incomplete: missing {expected - got}"
    pg = per_graph(df)
    # a graph counts only if all 5 weightings have exact, greedy and ours rows
    full = pg.dropna(subset=["W_exact", "W_greedy", "W_ours"]).groupby(["cls", "param", "n", "gidx"]).scheme.nunique()
    full = full[full == len(WEIGHTINGS)].reset_index()[["cls", "param", "n", "gidx"]]
    pg = pg.merge(full, on=["cls", "param", "n", "gidx"])
    pg.round(6).to_csv(RES / "pilot_sweep_per_graph.csv", index=False)
    pgg = with_grid(pg)
    pr = pgg[pgg.proven]

    agg = dict(instances=("W_exact", "size"), separation=("separation", "mean"), gap_greedy=("gap_greedy", "mean"),
               gap_ours=("gap_ours", "mean"), dW=("dW", "mean"), dS=("dS", "mean"),
               extra_W_ours=("extra_W_ours", "mean"), extra_W_greedy=("extra_W_greedy", "mean"),
               opt_ours_pct=("opt_ours", "mean"), opt_greedy_pct=("opt_greedy", "mean"))
    st = pr.groupby(["cls", "param", "grid", "n", "scheme"]).agg(**agg).reset_index()
    for c in ("opt_ours_pct", "opt_greedy_pct"):
        st[c] *= 100
    st.round(4).to_csv(RES / "pilot_sweep_settings.csv", index=False)

    rows = []
    for (cls, param, grid), d in pgg.groupby(["cls", "param", "grid"]):
        p = d[d.proven]
        graphs_done = d.groupby("n").gidx.nunique().reindex(GRIDS[grid], fill_value=0)
        rows.append(dict(cls=cls, param=param, grid=grid, complete=bool((graphs_done == 5).all()),
                         missing_graphs=int((5 - graphs_done).sum()), instances=len(d),
                         proven_pct=100 * d.proven.mean(),
                         ours_all_valid=bool(d.ours_valid.astype(str).str.lower().eq("true").all()),
                         separation=p.separation.mean(), separation_std=p.separation.std(ddof=0),
                         gap_greedy=p.gap_greedy.mean(), gap_ours=p.gap_ours.mean(), dW=p.dW.mean(), dS=p.dS.mean(),
                         **{f"sep_{w}": p[p.scheme == w].separation.mean() for w in WEIGHTINGS}))
    ch = pd.DataFrame(rows)
    ch["eligible"] = ch.complete & ch.ours_all_valid & (ch.proven_pct >= 100 * MIN_PROVEN)
    order = {p: i for i, (_, p) in enumerate(SETTINGS)}
    ch["_o"] = ch.param.map(order)
    ch = ch.sort_values(["cls", "separation", "grid", "_o"], ascending=[True, False, False, True])
    chosen = {}
    for cls in CLASSES:
        c = ch[(ch.cls == cls) & ch.eligible]
        assert len(c), f"no eligible pilot setting for {cls}"
        best = c.iloc[0]
        sep_w = {w: float(best[f"sep_{w}"]) for w in WEIGHTINGS if w != "unit"}
        wbest = max(sep_w, key=lambda w: sep_w[w])
        chosen[cls] = dict(param=best.param, param_label=PARAM_LABEL[best.param], grid=best.grid,
                           ns=list(GRIDS[best.grid]), weighting=wbest, separation=float(best.separation),
                           separation_by_weighting={w: float(best[f"sep_{w}"]) for w in WEIGHTINGS},
                           proven_pct=float(best.proven_pct), gap_greedy=float(best.gap_greedy),
                           gap_ours=float(best.gap_ours), dW=float(best.dW), dS=float(best.dS),
                           runner_up=None if len(c) < 2 else dict(param=c.iloc[1].param, grid=c.iloc[1].grid,
                                                                  separation=float(c.iloc[1].separation)))
    ch["chosen"] = [chosen[r.cls]["param"] == r.param and chosen[r.cls]["grid"] == r.grid for r in ch.itertuples()]
    ch.drop(columns="_o").round(4).to_csv(RES / "pilot_sweep_choice.csv", index=False)
    meta = dict(rule=__doc__.split("Rule (per class):")[1].strip(), min_proven=MIN_PROVEN,
                pilot_instances=int(len(pg)), pilot_graphs=int(pg.groupby(["cls", "param", "n", "gidx"]).ngroups),
                pilot_unproven=int((~pg.proven).sum()), chosen=chosen)
    (RES / "chosen_settings.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    heatmap(ch, chosen)
    print(ch.drop(columns="_o").round(3).to_string(index=False))
    print(json.dumps(chosen, indent=1))


def heatmap(ch, chosen):
    FIG.mkdir(parents=True, exist_ok=True)
    left = np.array([[chosen[c]["separation_by_weighting"][w] for w in WEIGHTINGS] for c in CLASSES])
    full = pd.concat([ch[ch.cls == c].sort_values(["_o", "grid"]) for c in CLASSES])
    right = full[[f"sep_{w}" for w in WEIGHTINGS]].to_numpy(float)
    lim = max(np.nanmax(np.abs(left)), np.nanmax(np.abs(right)), 1e-6)
    fig, axs = plt.subplots(2, 1, figsize=(11, 3.2 + 0.55 * (len(CLASSES) + len(full))), layout="constrained",
                            gridspec_kw=dict(height_ratios=[len(CLASSES), len(full)]))
    for ax, M, ylab, title in (
            (axs[0], left, [f"{TITLE[c]}\n{chosen[c]['param_label']}, grid {chosen[c]['grid']}" for c in CLASSES],
             "Chosen setting per class"),
            (axs[1], right, [f"{TITLE[r.cls]} · {PARAM_LABEL[r.param]} · {r.grid}" + ("  ★" if r.chosen else "")
                             + ("" if r.eligible else "  (incomplete: not evaluated)" if not r.complete
                                else "  (ineligible)") for r in full.itertuples()],
             "All pilot settings (★ = chosen)")):
        im = ax.imshow(M, cmap="RdBu", vmin=-lim, vmax=lim, aspect="auto")
        ax.set_xticks(range(len(WEIGHTINGS)), WEIGHTINGS)
        ax.set_yticks(range(len(ylab)), ylab, fontsize=10)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                ax.text(j, i, f"{M[i, j]:+.2f}", ha="center", va="center", fontsize=9,
                        color="white" if abs(M[i, j]) > 0.6 * lim else "#1F2328")
        ax.set_title(title, fontsize=12)
        for sp in ax.spines.values():
            sp.set_visible(False)
    cb = fig.colorbar(im, ax=axs, shrink=0.6, pad=0.03)
    cb.set_label("SEPARATION = greedy gap % − ours gap % (pp)\nblue: ours closer to the optimum")
    fig.suptitle("Pilot sweep · separation between greedy + RR and ours + RR\n(pilot seeds only, 5 graphs per n)",
                 fontsize=13, fontweight="bold")
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"pilot_separation_heatmap.{ext}", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
