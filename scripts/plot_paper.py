"""Paper figures and table for exact vs ours, re-drawn from the EXISTING results/exact_vs_ours_final.csv only
(no solver, no training; the CSV is read, never written).

    python scripts/plot_paper.py

Outputs (results/figures/paper/, 300 dpi PNG + vector PDF):
  plot1_cardinality_unit       |S| vs n, unit weights
  plot2_cardinality_weighted   |S| vs n, degree weights (the setting chosen by the pilot for every class)
  plot3_total_weight           W(S) vs n, degree weights
  plot5_ratio                  W(S) / W_exact vs n, degree weights (exact = 1.0)
  paper_main                   3 x 4: rows Plot 1 / Plot 3 / Plot 5, columns the classes; one shared legend
Table: results/tables/paper_table.tex (n = 100, 500, 1000; degree weights).
Summary of the box values: results/paper_beats_greedy.json.

Text box "Proposed beats Greedy in X% of graphs [(ties Y%)]": over all (graph, checkpoint) pairs of the panel (all n,
both model seeds), X = share where ours + RR is STRICTLY better than greedy + RR on the same graph and weighting:
smaller |S| in Plot 1 (unit weights), smaller W(S) in Plots 2, 3 and 5 (W is the objective, also in Plot 2).
Ties = equal value. X is rounded DOWN to one decimal (never in our favour); ties are rounded to nearest.
Curves: mean over graphs (ours first averaged over the two checkpoints), error bars = std over graphs.
"""
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

RES = ROOT / "results"
OUT = RES / "figures" / "paper"
CLASSES = ("tree", "unit_interval", "convex_bip", "two_tree")
TITLE = {"tree": "Trees", "unit_interval": "Unit-interval graphs", "convex_bip": "Convex bipartite graphs",
         "two_tree": "2-trees"}
EXACT_LABEL = {"tree": "Exact (tree DP)", "unit_interval": "Exact (our interval DP)", "convex_bip": "Exact (ILP)",
               "two_tree": "Exact (ILP)"}
INK, TEAL, GREY = "#1F2328", "#0F766E", "#9CA3AF"
WEIGHTED = "degree"
NOTE2 = "With weighted graphs, equal-cost sets can differ in size; total weight (Plot 3) is the objective."
YLAB = {"S": "Size of the set |S|", "W": "Total weight W(S)", "ratio": "Approximation ratio W(S) / W_exact"}
SERIES = (("exact", INK, "o", "-", 0.0), ("ours", TEAL, "s", "-", 0.012), ("greedy", GREY, "^", "--", -0.012))
plt.rcParams.update({"font.size": 11, "axes.titlesize": 12.5, "axes.labelsize": 11, "legend.fontsize": 10.5,
                     "xtick.labelsize": 10, "ytick.labelsize": 10.5, "font.family": "DejaVu Sans"})


def load():
    df = pd.read_csv(RES / "exact_vs_ours_final.csv")
    for c in ("valid", "proven_optimal"):
        df[c] = df[c].astype(str).str.lower().eq("true")
    assert df["valid"].all()
    return df


def per_graph(df):
    key = ["cls", "n", "scheme", "gidx"]
    ex = df[df.method == "exact"].set_index(key)
    gr = df[df.method == "greedy_rr"].set_index(key)
    o = df[df.method == "ours_rr"].groupby(key)
    pg = pd.DataFrame({"proven": ex["proven_optimal"], "S_exact": ex["size"], "W_exact": ex["weight"],
                       "S_greedy": gr["size"], "W_greedy": gr["weight"], "S_ours": o["size"].mean(),
                       "W_ours": o["weight"].mean()}).reset_index()
    for m in ("exact", "ours", "greedy"):
        pg[f"ratio_{m}"] = pg[f"W_{m}"] / pg.W_exact
    return pg


def pairs(df):
    """One row per (graph, weighting, checkpoint): ours + RR vs greedy + RR on the same graph."""
    key = ["cls", "n", "scheme", "gidx"]
    o = df[df.method == "ours_rr"][key + ["model_seed", "size", "weight"]]
    g = df[df.method == "greedy_rr"][key + ["size", "weight"]]
    return o.merge(g, on=key, suffixes=("_ours", "_greedy"), validate="many_to_one")


def beats(pr, cls, scheme, by):
    d = pr[(pr.cls == cls) & (pr.scheme == scheme)]
    a, b = d[f"{by}_ours"], d[f"{by}_greedy"]
    win, tie = 100 * (a < b - 1e-9).mean(), 100 * ((a - b).abs() <= 1e-9).mean()
    return dict(beats_pct=math.floor(win * 10) / 10, ties_pct=round(tie, 1), pairs=int(len(d)),
                beats_exact=float(win))


def box_text(b, weight_word=False):
    s = f"Proposed beats Greedy{' (total weight)' if weight_word else ''} in {b['beats_pct']:.1f}% of graphs"
    return s + (f" (ties {b['ties_pct']:.1f}%)" if b["ties_pct"] > 0 else "")


def panel(ax, pg, cls, scheme, metric, box, legend=True, title=None):
    d = pg[(pg.cls == cls) & (pg.scheme == scheme) & pg.proven]
    ns = sorted(d.n.unique())
    step = max(ns) - min(ns)
    for ser, col, mk, ls, dx in SERIES:
        g = d.groupby("n")[f"{metric}_{ser}"]
        m, s = g.mean().loc[ns].to_numpy(), g.std(ddof=0).loc[ns].to_numpy()
        lab = {"exact": EXACT_LABEL[cls], "ours": "Proposed + RR (2 seeds)", "greedy": "Greedy + RR"}[ser]
        ax.errorbar([n + dx * step for n in ns], m, yerr=s, color=col, ls=ls, lw=2, marker=mk,
                    ms=6.5 if mk != "^" else 7.5, capsize=3, elinewidth=1.2, label=lab,
                    zorder=3 if ser != "greedy" else 2, markeredgecolor="white" if ser != "exact" else INK,
                    markeredgewidth=0.9)
    ax.set_xticks(ns)
    ax.set_xticklabels([str(n) for n in ns], rotation=45)
    ax.set_title(title or TITLE[cls])
    ax.set_xlabel("Number of nodes (n)")
    ax.set_ylabel(YLAB[metric])
    ax.grid(True, color="#E5E7EB", lw=0.8)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    lo, hi = ax.get_ylim()                                   # free strips: legend on top, the box at the bottom
    ax.set_ylim(lo - 0.16 * (hi - lo), hi + (0.34 if legend else 0.06) * (hi - lo))
    if legend:
        ax.legend(loc="upper left", frameon=False)
    ax.text(0.98, 0.03, box, transform=ax.transAxes, ha="right", va="bottom", fontsize=10, color=INK,
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#D1D5DB"), zorder=5)


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=300)
    plt.close(fig)


def figure(pg, pr, name, title, metric, scheme, by, weight_word=False, note=None):
    fig, axs = plt.subplots(2, 2, figsize=(13, 10 if note else 9.6))
    out = {}
    for ax, cls in zip(axs.flat, CLASSES):
        b = beats(pr, cls, scheme, by)
        out[cls] = b
        panel(ax, pg, cls, scheme, metric, box_text(b, weight_word),
              title=f"{TITLE[cls]} ({'unit' if scheme == 'unit' else scheme} weights)")
    fig.suptitle(title, fontsize=14, fontweight="bold", color=INK)
    if note:
        fig.text(0.5, 0.012, note, ha="center", va="bottom", fontsize=11, color="#4B5563")
        fig.tight_layout(rect=(0, 0.035, 1, 0.97))
    else:
        fig.tight_layout(rect=(0, 0, 1, 0.97))
    save(fig, name)
    return out


def paper_main(pg, pr):
    fig, axs = plt.subplots(3, 4, figsize=(22, 15))
    rows = (("S", "unit", "size", "|S|, unit weights"), ("W", WEIGHTED, "weight", "W(S), degree weights"),
            ("ratio", WEIGHTED, "weight", "W(S) / W_exact, degree weights"))
    for r, (metric, sch, by, lab) in enumerate(rows):
        for c, cls in enumerate(CLASSES):
            panel(axs[r, c], pg, cls, sch, metric, box_text(beats(pr, cls, sch, by)), legend=False,
                  title=f"{TITLE[cls]}\n{lab}" if r == 0 else lab)
    handles = [Line2D([], [], color=INK, ls="-", marker="o", lw=2, label="Exact (tree DP / interval DP / ILP)"),
               Line2D([], [], color=TEAL, ls="-", marker="s", lw=2, mec="white", label="Proposed + RR (2 seeds)"),
               Line2D([], [], color=GREY, ls="--", marker="^", lw=2, mec="white", label="Greedy + RR")]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, fontsize=13, bbox_to_anchor=(0.5, 0.995))
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    save(fig, "paper_main")


def table(pg, pr):
    short = {"tree": "Trees", "unit_interval": "Unit-interval", "convex_bip": "Convex bipartite", "two_tree": "2-trees"}
    lines = [r"\begin{tabular}{lr rrr rr r}", r"\toprule",
             r" & & \multicolumn{3}{c}{mean $W(S)$} & \multicolumn{2}{c}{mean $W/W_{\mathrm{exact}}$} & \\",
             r"class & $n$ & Exact & Proposed + RR & Greedy + RR & Proposed + RR & Greedy + RR & Proposed beats Greedy (\%) \\", r"\midrule"]
    rows = []
    for cls in CLASSES:
        for n in (100, 500, 1000):
            d = pg[(pg.cls == cls) & (pg.scheme == WEIGHTED) & (pg.n == n) & pg.proven]
            b = beats(pr[pr.n == n], cls, WEIGHTED, "weight")
            rows.append(dict(cls=cls, n=n, W_exact=d.W_exact.mean(), W_ours=d.W_ours.mean(), W_greedy=d.W_greedy.mean(),
                             ratio_ours=d.ratio_ours.mean(), ratio_greedy=d.ratio_greedy.mean(), **b))
            lines.append(f"{short[cls]} & {n} & {d.W_exact.mean():.1f} & {d.W_ours.mean():.1f} & "
                         f"{d.W_greedy.mean():.1f} & {d.ratio_ours.mean():.3f} & {d.ratio_greedy.mean():.3f} & "
                         f"{b['beats_pct']:.1f}" + (f" (ties {b['ties_pct']:.1f})" if b["ties_pct"] > 0 else "")
                         + r" \\")
        if cls != CLASSES[-1]:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"% degree weights; 10 graphs per n; Proposed = mean of 2 checkpoints; Proposed beats Greedy = share of "
              r"(graph, checkpoint) pairs with strictly smaller W(S), rounded down; from results/exact_vs_ours_final.csv"]
    (RES / "tables" / "paper_table.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = load()
    pg, pr = per_graph(df), pairs(df)
    assert len(pr) == 2 * len(pg), "expected two checkpoints per graph and weighting"
    res = {
        "plot1_cardinality_unit (|S|, unit)": figure(pg, pr, "plot1_cardinality_unit",
                                                     "Plot 1 · Cardinality |S| — unit weights (all w = 1)",
                                                     "S", "unit", "size"),
        "plot2_cardinality_weighted (box: W, degree)": figure(
            pg, pr, "plot2_cardinality_weighted", "Plot 2 · Cardinality |S| — degree weights", "S", WEIGHTED,
            "weight", weight_word=True, note=NOTE2),
        "plot3_total_weight (W, degree)": figure(pg, pr, "plot3_total_weight",
                                                 "Plot 3 · Total weight W(S) — degree weights", "W", WEIGHTED,
                                                 "weight"),
        "plot5_ratio (W, degree)": figure(pg, pr, "plot5_ratio",
                                          "Plot 5 · Approximation ratio W(S) / W_exact — degree weights (exact = 1.0)",
                                          "ratio", WEIGHTED, "weight")}
    paper_main(pg, pr)
    res["paper_table_n100_500_1000"] = table(pg, pr)
    (RES / "paper_beats_greedy.json").write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")
    for k, v in res.items():
        if isinstance(v, dict):
            print(k, {c: (b["beats_pct"], b["ties_pct"]) for c, b in v.items()})


if __name__ == "__main__":
    main()
