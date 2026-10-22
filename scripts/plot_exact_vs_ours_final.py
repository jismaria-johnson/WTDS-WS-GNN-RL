"""STEPS 4 and 6 of the exact-vs-ours plot update: Plots 1-5, the combined paper figure, tables and a summary, all
from results/exact_vs_ours_final.csv (fresh test-range graphs, settings from results/chosen_settings.json).

    python scripts/plot_exact_vs_ours_final.py

Outputs (results/figures/exact_vs_ours/, 300 dpi PNG + PDF):
  final_plot1_cardinality_unit      |S|, unit weights
  final_plot2_cardinality_chosen    |S|, the chosen weighting per class (named in each panel title)
  final_plot3_weight_chosen         W(S), chosen weighting
  final_plot4_extra_cost_chosen     W(S) - W_exact, chosen weighting (exact = 0)
  final_plot5_ratio_chosen          W(S) / W_exact, chosen weighting (exact = 1)
  final_combined_paper_figure       rows Plot 1, Plot 3, Plot 4; columns the four classes
Tables: results/tables/exact_vs_ours_final.csv (every weighting) and .tex (unit + chosen weighting).
Summary: results/exact_vs_ours_final_summary.json.

Aggregation per (class, n, weighting): only graphs whose exact optimum is PROVEN (hollow Exact marker and a
"k/10 proven" note otherwise). Ours = mean of the two checkpoints per graph; points = mean over graphs, error bars =
std over graphs. "Ours optimal in X%" = share of (graph, checkpoint) runs of ours + RR with W(S) = W_exact;
"Greedy optimal in Y%" = share of graphs with W_greedy = W_exact (with unit weights, W = |S|).
Axes are linear: the chosen n grids are evenly spaced, and a log y axis would compress the gaps that Plots 4-5 show.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

RES = ROOT / "results"
FIG = RES / "figures" / "exact_vs_ours"
TAB = RES / "tables"
CLASSES = ("tree", "unit_interval", "convex_bip", "two_tree")
TITLE = {"tree": "Trees", "unit_interval": "Unit-interval graphs", "convex_bip": "Convex bipartite graphs",
         "two_tree": "2-trees"}
EXACT_LABEL = {"tree": "Exact (tree DP)", "unit_interval": "Exact (our interval DP)",
               "convex_bip": "Exact (ILP)", "two_tree": "Exact (ILP)"}
SCHEME_LABEL = {"unit": "unit", "degree": "degree", "inverse": "inverse degree", "random": "random",
                "mixed": "mixed"}
INK, TEAL, GREY = "#1F2328", "#0F766E", "#9CA3AF"
CAPTION2 = ("With weighted graphs, several optimal sets can have the same weight but different sizes, so |S| can "
            "differ even when ours is optimal (see Plots 3–5).")
EPS = 1e-9
plt.rcParams.update({"font.size": 11, "axes.titlesize": 12.5, "axes.labelsize": 11, "legend.fontsize": 10.5,
                     "xtick.labelsize": 10, "ytick.labelsize": 10.5, "font.family": "DejaVu Sans"})
YLAB = {"S": "|S|", "W": "Total weight W(S)", "extra": "Extra cost  W(S) − W_exact", "ratio": "W(S) / W_exact"}


def gate():
    p = RES / "exact_validation.json"
    if not p.exists():
        sys.exit("results/exact_validation.json missing: run scripts/validate_exact.py first")
    v = json.loads(p.read_text())
    if v["total_mismatches"] or v["ilp_not_proven"]:
        sys.exit(f"exact-method validation failed ({v['total_mismatches']} mismatches): fix the algorithm first")
    return v


def merge_shards():
    """The final run ran as parallel shards (results/exact_vs_ours_final_part*.csv). Merge them (shards are kept)."""
    parts = sorted(RES.glob("exact_vs_ours_final_part*.csv"))
    if parts:
        df = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
        key = ["cls", "n", "gidx", "scheme", "method", "model_seed"]
        assert not df.duplicated(key).any(), "a (graph, weighting, method) row occurs in two shards"
        df.sort_values(["cls", "n", "gidx"], kind="stable").to_csv(RES / "exact_vs_ours_final.csv", index=False)


def load():
    merge_shards()
    df = pd.read_csv(RES / "exact_vs_ours_final.csv")
    for c in ("valid", "proven_optimal", "exact_proven"):
        df[c] = df[c].astype(str).str.lower().eq("true")
    assert df["valid"].all()
    for r in df.itertuples():                       # W(S) was recomputed before writing; recompute |S| once more
        assert len(str(r.S).split()) == r.size, f"|S| mismatch in row {r.Index}"
    return df


def per_graph(df):
    key = ["cls", "n", "scheme", "gidx"]
    ex = df[df.method == "exact"].set_index(key)
    gr = df[df.method == "greedy_rr"].set_index(key)
    od = df[df.method == "ours_rr"].copy()
    od["opt"] = (od.weight - od.exact_weight).abs() < EPS
    o = od.groupby(key)
    ld = df[df.method == "ours_rr_ls"].copy()
    ld["opt"] = (ld.weight - ld.exact_weight).abs() < EPS
    ol = ld.groupby(key)
    out = pd.DataFrame({
        "param": ex["param"], "proven": ex["proven_optimal"], "S_exact": ex["size"], "W_exact": ex["weight"],
        "t_exact": ex["runtime"], "S_greedy": gr["size"], "W_greedy": gr["weight"], "t_greedy": gr["runtime"],
        "S_ours": o["size"].mean(), "W_ours": o["weight"].mean(), "t_ours": o["runtime"].mean(),
        "opt_ours": o["opt"].mean(), "W_ours_ls": ol["weight"].mean(), "S_ours_ls": ol["size"].mean(),
        "opt_ours_ls": ol["opt"].mean(), "t_ours_ls": ol["runtime"].mean()}).reset_index()
    seeds = od.pivot_table(index=key, columns="model_seed", values="weight")
    seeds.columns = [f"W_ours_seed{c}" for c in seeds.columns]
    out = out.merge(seeds.reset_index(), on=key)
    out["opt_greedy"] = ((out.W_greedy - out.W_exact).abs() < EPS).astype(float)
    for m in ("ours", "greedy", "ours_ls"):
        out[f"gap_{m}"] = 100 * (out[f"W_{m}"] - out.W_exact) / out.W_exact
        out[f"extra_{m}"] = out[f"W_{m}"] - out.W_exact
        out[f"ratio_{m}"] = out[f"W_{m}"] / out.W_exact
    out["extra_exact"], out["ratio_exact"] = 0.0, 1.0
    return out


def series_stats(d, metric, ser):
    col = f"{metric}_{ser}"
    return d[col].mean(), d[col].std(ddof=0)


def panel(ax, pg, cls, scheme, metric, title=None, legend=True):
    d = pg[(pg.cls == cls) & (pg.scheme == scheme)]
    ns = sorted(d.n.unique())
    step = (max(ns) - min(ns)) or 1
    for ser, lab, col, mk, ls, dx in (("exact", EXACT_LABEL[cls], INK, "o", "-", 0.0),
                                      ("ours", "Ours (GNN+RL + RR, 2 seeds)", TEAL, "s", "-", 0.012),
                                      ("greedy", "Greedy + RR", GREY, "^", "--", -0.012)):
        m, s, hollow = [], [], []
        for n in ns:
            c = d[d.n == n]
            p = c[c.proven]
            mu, sd = series_stats(p, metric, ser) if len(p) else (np.nan, np.nan)
            m.append(mu)
            s.append(sd)
            if len(p) < len(c):
                hollow.append((n, mu, len(p), len(c)))
        xs = [n + dx * step for n in ns]
        ax.errorbar(xs, m, yerr=s, color=col, ls=ls, lw=2, marker=mk, ms=6.5 if mk != "^" else 7.5, capsize=3,
                    elinewidth=1.2, label=lab, zorder=3 if ser != "greedy" else 2,
                    markeredgecolor="white" if ser != "exact" else INK, markeredgewidth=0.9)
        if ser == "exact":
            for n, mu, k, tot in hollow:
                ax.plot([n], [mu], "o", ms=10, mfc="white", mec=INK, mew=1.6, zorder=4)
                ax.annotate(f"{k}/{tot} proven", (n, mu), textcoords="offset points", xytext=(5, -12), fontsize=9)
    ax.set_xticks(ns)
    ax.set_xticklabels([str(n) for n in ns], rotation=45 if len(ns) > 6 else 0)
    ax.set_title(title or f"{TITLE[cls]} ({SCHEME_LABEL[scheme]} weights)")
    ax.set_xlabel("Number of nodes (n)")
    ax.set_ylabel(YLAB[metric])
    ax.grid(True, color="#E5E7EB", lw=0.8)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    lo, hi = ax.get_ylim()                                   # free strips: legend on top, the % box at the bottom
    ax.set_ylim(lo - 0.16 * (hi - lo), hi + (0.34 if legend else 0.08) * (hi - lo))
    if legend:
        ax.legend(loc="upper left", frameon=False)
    p = d[d.proven]
    box = f"Ours optimal in {100 * p.opt_ours.mean():.0f}% | Greedy optimal in {100 * p.opt_greedy.mean():.0f}%"
    ax.text(0.98, 0.03, box, transform=ax.transAxes, ha="right", va="bottom", fontsize=10, color=INK,
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#D1D5DB"), zorder=5)


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=300)
    plt.close(fig)


def figure(pg, chosen, metric, unit, name, title, caption=None):
    fig, axs = plt.subplots(2, 2, figsize=(13, 10 if caption else 9.6))
    for ax, cls in zip(axs.flat, CLASSES):
        sch = "unit" if unit else chosen[cls]["weighting"]
        panel(ax, pg, cls, sch, metric,
              title=f"{TITLE[cls]} · {chosen[cls]['param_label']} · {SCHEME_LABEL[sch]} weights")
    fig.suptitle(title, fontsize=14, fontweight="bold", color=INK)
    if caption:
        fig.text(0.5, 0.012, caption, ha="center", va="bottom", fontsize=11, color="#4B5563", wrap=True)
        fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    else:
        fig.tight_layout(rect=(0, 0, 1, 0.97))
    save(fig, name)


def combined(pg, chosen):
    fig, axs = plt.subplots(3, 4, figsize=(22, 14.5))
    rows = (("Plot 1 · |S|, unit weights", "S", True), ("Plot 3 · W(S)", "W", False),
            ("Plot 4 · extra cost W(S) − W_exact", "extra", False))
    for r, (lab, metric, unit) in enumerate(rows):
        for c, cls in enumerate(CLASSES):
            sch = "unit" if unit else chosen[cls]["weighting"]
            t = f"{lab} ({SCHEME_LABEL[sch]} weights)" if not unit else lab
            panel(axs[r, c], pg, cls, sch, metric, title=(f"{TITLE[cls]} · {chosen[cls]['param_label']}\n" if r == 0
                                                          else "") + t, legend=(c == 0))
    fig.suptitle("Exact vs ours vs greedy + RR on graph classes with exact algorithms (fresh test graphs)",
                 fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save(fig, "final_combined_paper_figure")


def table(pg, chosen):
    rows = []
    for cls in CLASSES:
        for sch in ("unit", "degree", "inverse", "random", "mixed"):
            for n, d in pg[(pg.cls == cls) & (pg.scheme == sch)].groupby("n"):
                p = d[d.proven]
                seed_cols = [c for c in p.columns if c.startswith("W_ours_seed")]
                rows.append(dict(
                    cls=cls, param=chosen[cls]["param"], scheme=sch, plotted=sch in ("unit", chosen[cls]["weighting"]),
                    n=int(n), graphs=len(d), proven=len(p),
                    S_exact=p.S_exact.mean(), S_ours=p.S_ours.mean(), S_ours_ls=p.S_ours_ls.mean(),
                    S_greedy=p.S_greedy.mean(), W_exact=p.W_exact.mean(), W_ours=p.W_ours.mean(),
                    W_ours_seed_std=float(np.std([p[c].mean() for c in seed_cols])), W_ours_ls=p.W_ours_ls.mean(),
                    W_greedy=p.W_greedy.mean(), gap_ours_pct=p.gap_ours.mean(), gap_ours_ls_pct=p.gap_ours_ls.mean(),
                    gap_greedy_pct=p.gap_greedy.mean(), ratio_ours=p.ratio_ours.mean(),
                    ratio_ours_ls=p.ratio_ours_ls.mean(), ratio_greedy=p.ratio_greedy.mean(),
                    opt_ours_pct=100 * p.opt_ours.mean(), opt_ours_ls_pct=100 * p.opt_ours_ls.mean(),
                    opt_greedy_pct=100 * p.opt_greedy.mean(), t_exact_s=p.t_exact.mean(), t_ours_s=p.t_ours.mean(),
                    t_ours_ls_s=p.t_ours_ls.mean(), t_greedy_s=p.t_greedy.mean()))
    t = pd.DataFrame(rows)
    t.round(4).to_csv(TAB / "exact_vs_ours_final.csv", index=False)
    short = {"tree": "Trees", "unit_interval": "Unit-int.", "convex_bip": "Convex bip.", "two_tree": "2-trees"}
    lines = [r"\begin{tabular}{llr rrr rrr rrr rrr rrr}", r"\toprule",
             r" & & & \multicolumn{3}{c}{$|S|$} & \multicolumn{3}{c}{$W$} & \multicolumn{3}{c}{gap $W$ (\%)} & "
             r"\multicolumn{3}{c}{optimal (\%)} & \multicolumn{3}{c}{time (s)} \\",
             r"class & weights & $n$ & exact & ours & greedy & exact & ours & greedy & ours & +LS & greedy & "
             r"ours & +LS & greedy & exact & ours & greedy \\", r"\midrule"]
    for r in t[t.plotted].itertuples():
        lines.append(f"{short[r.cls]} & {r.scheme} & {r.n} & {r.S_exact:.1f} & {r.S_ours:.1f} & {r.S_greedy:.1f} & "
                     f"{r.W_exact:.1f} & {r.W_ours:.1f} & {r.W_greedy:.1f} & {r.gap_ours_pct:.2f} & "
                     f"{r.gap_ours_ls_pct:.2f} & {r.gap_greedy_pct:.2f} & {r.opt_ours_pct:.0f} & "
                     f"{r.opt_ours_ls_pct:.0f} & {r.opt_greedy_pct:.0f} & {r.t_exact_s:.3f} & {r.t_ours_s:.3f} & "
                     f"{r.t_greedy_s:.3f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TAB / "exact_vs_ours_final.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return t


def summary(df, pg, t, chosen, val):
    ex = df[df.method == "exact"]
    chk = df[df.method == "ilp_check"]
    s = dict(validation=val, graphs=int(df.groupby(["cls", "n", "gidx"]).ngroups), solutions_checked=int(len(df)),
             all_valid=bool(df.valid.all()), exact_runs=int(len(ex)),
             exact_not_proven=[dict(cls=r.cls, n=int(r.n), gidx=int(r.gidx), scheme=r.scheme)
                               for r in ex[~ex.proven_optimal].itertuples()],
             ilp_crosscheck=dict(runs=int(len(chk)), proven=int(chk.proven_optimal.sum()),
                                 agree=int((chk.proven_optimal & ((chk.weight - chk.exact_weight).abs() < EPS)
                                            & (chk["size"] == chk.exact_size)).sum())),
             chosen=chosen, per_class={})
    for cls in CLASSES:
        w = chosen[cls]["weighting"]
        sub = t[(t.cls == cls) & (t.scheme == w)]
        last = sub[sub.n == sub.n.max()].iloc[0].to_dict()
        p = pg[(pg.cls == cls) & pg.proven]
        s["per_class"][cls] = dict(
            weighting=w, largest_n=last,
            unit_all_n=dict(opt_ours_pct=100 * p[p.scheme == "unit"].opt_ours.mean(),
                            opt_greedy_pct=100 * p[p.scheme == "unit"].opt_greedy.mean(),
                            gap_ours_pct=p[p.scheme == "unit"].gap_ours.mean(),
                            gap_greedy_pct=p[p.scheme == "unit"].gap_greedy.mean()),
            chosen_all_n=dict(opt_ours_pct=100 * p[p.scheme == w].opt_ours.mean(),
                              opt_greedy_pct=100 * p[p.scheme == w].opt_greedy.mean(),
                              opt_ours_ls_pct=100 * p[p.scheme == w].opt_ours_ls.mean(),
                              gap_ours_pct=p[p.scheme == w].gap_ours.mean(),
                              gap_ours_ls_pct=p[p.scheme == w].gap_ours_ls.mean(),
                              gap_greedy_pct=p[p.scheme == w].gap_greedy.mean(),
                              separation_pp=(p[p.scheme == w].gap_greedy - p[p.scheme == w].gap_ours).mean(),
                              ours_better_graphs=int((p[p.scheme == w].W_ours < p[p.scheme == w].W_greedy - EPS).sum()),
                              greedy_better_graphs=int((p[p.scheme == w].W_greedy < p[p.scheme == w].W_ours - EPS)
                                                       .sum()),
                              graphs=int((p.scheme == w).sum())))
    (RES / "exact_vs_ours_final_summary.json").write_text(json.dumps(s, indent=2, default=float), encoding="utf-8")
    return s


def main():
    val = gate()
    FIG.mkdir(parents=True, exist_ok=True)
    chosen = json.loads((RES / "chosen_settings.json").read_text(encoding="utf-8"))["chosen"]
    df = load()
    pg = per_graph(df)
    figure(pg, chosen, "S", True, "final_plot1_cardinality_unit", "Plot 1 · Cardinality |S| — unit weights (all w = 1)")
    figure(pg, chosen, "S", False, "final_plot2_cardinality_chosen",
           "Plot 2 · Cardinality |S| — weighted graphs (weighting chosen by the pilot)", caption=CAPTION2)
    figure(pg, chosen, "W", False, "final_plot3_weight_chosen", "Plot 3 · Total weight W(S) — weighted graphs")
    figure(pg, chosen, "extra", False, "final_plot4_extra_cost_chosen",
           "Plot 4 · Extra cost over the optimum: W(S) − W_exact (exact = 0)")
    figure(pg, chosen, "ratio", False, "final_plot5_ratio_chosen",
           "Plot 5 · Approximation ratio W(S) / W_exact (exact = 1.0)")
    combined(pg, chosen)
    t = table(pg, chosen)
    s = summary(df, pg, t, chosen, val)
    print(json.dumps({k: s[k] for k in ("graphs", "solutions_checked", "exact_not_proven", "ilp_crosscheck")}, indent=1))
    for cls, v in s["per_class"].items():
        L = v["largest_n"]
        print(f"{cls:<14} {v['weighting']:<8} n={L['n']}: W exact {L['W_exact']:.1f} ours {L['W_ours']:.1f} "
              f"greedy {L['W_greedy']:.1f} | gap ours {L['gap_ours_pct']:.2f}% greedy {L['gap_greedy_pct']:.2f}% | "
              f"opt ours {L['opt_ours_pct']:.0f}% greedy {L['opt_greedy_pct']:.0f}%")
        print("   all n:", {k: round(x, 3) for k, x in v["chosen_all_n"].items()})


if __name__ == "__main__":
    main()
