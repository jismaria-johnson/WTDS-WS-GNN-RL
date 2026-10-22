"""Plots and tables for exact vs ours (reads results/exact_vs_ours.csv only).

    python scripts/plot_exact_vs_ours.py

Refuses to run unless results/exact_validation.json reports 0 mismatches and 0 unproven ILP references.
Series are dodged horizontally by a few percent (log x axis) so that overlapping markers stay visible.
Aggregation per (class, n, scheme): only graphs whose exact optimum is PROVEN are used for every method;
a cell with an unproven graph gets a hollow Exact marker and a "k/10 proven" note. Points = mean over graphs,
error bars = std over graphs; "ours" is first averaged over the two model seeds per graph.
"% optimal" = share of (graph, model seed) runs of ours + RR that match the exact value.
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
SCHEMES = ("degree", "inverse", "random", "mixed")
SCHEME_LABEL = {"unit": "unit weights", "degree": "degree", "inverse": "inverse degree", "random": "random",
                "mixed": "mixed"}
INK, TEAL, GREY = "#1F2328", "#0F766E", "#9CA3AF"
CAPTION2 = ("With weighted graphs, several optimal sets can have the same minimum weight but different sizes, so |S| "
            "can differ from the exact method even when ours is optimal. See Plot 3.")
plt.rcParams.update({"font.size": 11, "axes.titlesize": 12.5, "axes.labelsize": 11, "legend.fontsize": 11,
                     "xtick.labelsize": 11, "ytick.labelsize": 11, "font.family": "DejaVu Sans"})


def gate():
    p = RES / "exact_validation.json"
    if not p.exists():
        sys.exit("results/exact_validation.json missing: run scripts/validate_exact.py first")
    v = json.loads(p.read_text())
    if v["total_mismatches"] or v["ilp_not_proven"]:
        sys.exit(f"exact-method validation failed ({v['total_mismatches']} mismatches): fix the algorithm first")
    return v


def load():
    df = pd.read_csv(RES / "exact_vs_ours.csv")
    assert df["valid"].all()
    for c in ("proven_optimal", "exact_proven", "optimal_weight_reached", "same_size_as_exact"):
        df[c] = df[c].astype(str).str.lower().eq("true")
    return df


def per_graph(df):
    """One row per (cls, n, scheme, gidx): exact, greedy, ours (mean of seeds), ours+LS, runtimes, optimal flags."""
    key = ["cls", "n", "scheme", "gidx"]
    ex = df[df.method == "exact"].set_index(key)
    gr = df[df.method == "greedy_rr"].set_index(key)
    o = df[df.method == "ours_rr"].groupby(key)
    ol = df[df.method == "ours_rr_ls"].groupby(key)
    out = pd.DataFrame({
        "proven": ex["proven_optimal"], "S_exact": ex["size"], "W_exact": ex["weight"], "t_exact": ex["runtime"],
        "S_greedy": gr["size"], "W_greedy": gr["weight"], "t_greedy": gr["runtime"],
        "S_ours": o["size"].mean(), "W_ours": o["weight"].mean(), "t_ours": o["runtime"].mean(),
        "opt_W_ours": o["optimal_weight_reached"].mean(), "same_S_ours": o["same_size_as_exact"].mean(),
        "W_ours_ls": ol["weight"].mean(), "opt_W_ours_ls": ol["optimal_weight_reached"].mean(),
        "t_ours_ls": ol["runtime"].mean()}).reset_index()
    for m in ("ours", "greedy", "ours_ls"):
        out[f"gap_{m}"] = 100 * (out[f"W_{m}"] - out["W_exact"]) / out["W_exact"]
    return out


def cells(pg, cls, scheme):
    """Per n: (n, n_proven, n_total, {series: (mean, std)}) over proven graphs."""
    rows = []
    for n, d in pg[(pg.cls == cls) & (pg.scheme == scheme)].groupby("n"):
        p = d[d.proven]
        stats = {c: (p[c].mean(), p[c].std(ddof=0)) for c in ("S_exact", "S_ours", "S_greedy", "W_exact", "W_ours",
                                                               "W_greedy")}
        rows.append((int(n), len(p), len(d), stats))
    return rows


def panel(ax, pg, cls, scheme, what, box):
    rows = [r for r in cells(pg, cls, scheme) if r[1] > 0]
    ns = [r[0] for r in rows]
    if not rows:                                             # nothing proven / not run yet: say so, plot nothing
        ax.set_title(TITLE[cls])
        ax.text(0.5, 0.5, "pending", transform=ax.transAxes, ha="center", va="center", color=GREY, fontsize=14)
        ax.legend(handles=[], frameon=False)
        return
    key = "S" if what == "size" else "W"
    for ser, lab, col, mk, ls, dx in (("exact", EXACT_LABEL[cls], INK, "o", "-", 1.0),
                                      ("ours", "Ours (GNN+RL + RR)", TEAL, "s", "-", 1.07),
                                      ("greedy", "Greedy + RR", GREY, "^", "--", 0.935)):
        xs = [n_ * dx for n_ in ns]                         # small horizontal dodge (log x) so markers stay visible
        m = np.array([r[3][f"{key}_{ser}"][0] for r in rows])
        s = np.array([r[3][f"{key}_{ser}"][1] for r in rows])
        ax.errorbar(xs, m, yerr=s, color=col, ls=ls, lw=2, marker=mk, ms=7 if mk != "^" else 8, capsize=3,
                    elinewidth=1.2, label=lab, zorder=3 if ser != "greedy" else 2,
                    markeredgecolor="white" if ser != "exact" else INK, markeredgewidth=1.0)
        if ser == "exact":                                   # hollow marker where some optimum is not proven
            for n_, (_, k, tot, st) in zip(ns, rows):
                if k < tot:
                    ax.plot([n_], [st[f"{key}_exact"][0]], "o", ms=11, mfc="white", mec=INK, mew=1.6, zorder=4)
                    ax.annotate(f"{k}/{tot} proven", (n_, st[f"{key}_exact"][0]), textcoords="offset points",
                                xytext=(6, -12), fontsize=11, color=INK)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(ns)
    ax.set_xticklabels([str(n) for n in ns])
    ax.minorticks_off()
    ax.set_title(TITLE[cls])
    ax.set_xlabel("Number of nodes (n)")
    ax.set_ylabel("|S|" if what == "size" else "Total weight W(S)")
    ax.grid(True, color="#E5E7EB", lw=0.8)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(loc="upper left", frameon=False)
    ax.text(0.97, 0.04, box, transform=ax.transAxes, ha="right", va="bottom", fontsize=11, color=INK,
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#D1D5DB"))


def pct_opt(pg, cls, scheme, col):
    d = pg[(pg.cls == cls) & (pg.scheme == scheme) & pg.proven]
    return 100 * d[col].mean()


def box_text(pg, cls, scheme, plot):
    if plot == 1:
        return f"Ours optimal in {pct_opt(pg, cls, scheme, 'same_S_ours'):.0f}% of graphs"
    if plot == 3:
        return f"Ours optimal in {pct_opt(pg, cls, scheme, 'opt_W_ours'):.0f}% of graphs"
    return (f"Ours optimal (W) in {pct_opt(pg, cls, scheme, 'opt_W_ours'):.0f}% of graphs\n"
            f"same |S| as exact in {pct_opt(pg, cls, scheme, 'same_S_ours'):.0f}%")


def figure(pg, scheme, plot, fname, title, caption=None):
    fig, axs = plt.subplots(2, 2, figsize=(12, 9.6 if caption else 9))
    for ax, cls in zip(axs.flat, CLASSES):
        panel(ax, pg, cls, scheme, "weight" if plot == 3 else "size", box_text(pg, cls, scheme, plot))
    fig.suptitle(title, fontsize=14, fontweight="bold", color=INK)
    if caption:
        fig.text(0.5, 0.015, caption, ha="center", va="bottom", fontsize=11, color="#4B5563", wrap=True)
        fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    else:
        fig.tight_layout(rect=(0, 0, 1, 0.97))
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"{fname}.{ext}", dpi=300)
    plt.close(fig)


def combined(pg):
    fig, axs = plt.subplots(3, 4, figsize=(20, 14))
    rows = [(1, "unit", "size", "Plot 1 · |S|, unit weights"), (2, "random", "size", "Plot 2 · |S|, random weights"),
            (3, "random", "weight", "Plot 3 · W(S), random weights")]
    for r, (plot, sch, what, lab) in enumerate(rows):
        for c, cls in enumerate(CLASSES):
            ax = axs[r, c]
            panel(ax, pg, cls, sch, what, box_text(pg, cls, sch, plot))
            ax.set_title(f"{TITLE[cls]}\n{lab}" if r == 0 else lab, fontsize=12)
            if c > 0:
                ax.get_legend().remove()
    fig.suptitle("Exact vs ours on graph classes with exact algorithms", fontsize=15, fontweight="bold")
    fig.text(0.5, 0.008, "Row 2: " + CAPTION2, ha="center", fontsize=11, color="#4B5563")
    fig.tight_layout(rect=(0, 0.025, 1, 0.97))
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"combined_paper_figure.{ext}", dpi=300)
    plt.close(fig)


def table(pg):
    rows = []
    for cls in CLASSES:
        for sch in ("unit",) + SCHEMES:
            for n, d in pg[(pg.cls == cls) & (pg.scheme == sch)].groupby("n"):
                p = d[d.proven]
                rows.append(dict(cls=cls, scheme=sch, n=int(n), graphs=len(d), proven=len(p),
                                 S_exact=p.S_exact.mean(), S_ours=p.S_ours.mean(), W_exact=p.W_exact.mean(),
                                 W_ours=p.W_ours.mean(), gap_W_pct=p.gap_ours.mean(),
                                 opt_W_pct=100 * p.opt_W_ours.mean(), t_exact_s=p.t_exact.mean(),
                                 t_ours_s=p.t_ours.mean(), gap_W_ours_ls_pct=p.gap_ours_ls.mean(),
                                 opt_W_ours_ls_pct=100 * p.opt_W_ours_ls.mean(), t_ours_ls_s=p.t_ours_ls.mean(),
                                 gap_W_greedy_pct=p.gap_greedy.mean()))
    t = pd.DataFrame(rows)
    t.round(4).to_csv(TAB / "exact_vs_ours.csv", index=False)
    lines = [r"\begin{tabular}{llrrrrrrrrr}", r"\toprule",
             r"class & scheme & $n$ & $|S|$ exact & $|S|$ ours & $W$ exact & $W$ ours & gap $W$ (\%) & "
             r"opt.\ $W$ (\%) & $t$ exact (s) & $t$ ours (s) \\", r"\midrule"]
    for r in rows:
        lines.append(f"{TITLE[r['cls']]} & {r['scheme']} & {r['n']} & {r['S_exact']:.1f} & {r['S_ours']:.1f} & "
                     f"{r['W_exact']:.1f} & {r['W_ours']:.1f} & {r['gap_W_pct']:.2f} & {r['opt_W_pct']:.0f} & "
                     f"{r['t_exact_s']:.3f} & {r['t_ours_s']:.3f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TAB / "exact_vs_ours.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return t


def summary(df, pg, t, val):
    s = {"validation": val, "graphs": int(df.groupby(["cls", "n", "gidx"]).ngroups),
         "solutions_checked": int(len(df)), "all_valid": bool(df["valid"].all())}
    ex = df[df.method == "exact"]
    np_ = ex[~ex.proven_optimal]
    s["exact_runs"] = int(len(ex))
    s["exact_not_proven"] = [dict(cls=r.cls, n=int(r.n), gidx=int(r.gidx), scheme=r.scheme) for r in np_.itertuples()]
    chk = df[df.method == "ilp_check"]
    s["ilp_crosscheck"] = dict(runs=int(len(chk)), proven=int(chk.proven_optimal.sum()),
                               agree=int((chk.proven_optimal & chk.optimal_weight_reached & chk.same_size_as_exact).sum()))
    pct = {}
    for cls in CLASSES:
        d = pg[(pg.cls == cls) & pg.proven]
        u, wd = d[d.scheme == "unit"], d[d.scheme != "unit"]
        pct[cls] = dict(unit_opt_pct=100 * u.opt_W_ours.mean(), weighted_opt_W_pct=100 * wd.opt_W_ours.mean(),
                        weighted_same_S_pct=100 * wd.same_S_ours.mean(),
                        per_scheme_opt_W_pct={sc: 100 * d[d.scheme == sc].opt_W_ours.mean() for sc in SCHEMES},
                        unit_opt_pct_ours_ls=100 * u.opt_W_ours_ls.mean(),
                        weighted_opt_W_pct_ours_ls=100 * wd.opt_W_ours_ls.mean(),
                        unit_opt_pct_greedy=100 * (u.W_greedy == u.W_exact).mean(),
                        weighted_opt_W_pct_greedy=100 * (wd.W_greedy == wd.W_exact).mean(),
                        mean_gap_unit_pct=u.gap_ours.mean(), mean_gap_weighted_pct=wd.gap_ours.mean(),
                        mean_gap_unit_greedy_pct=u.gap_greedy.mean(), mean_gap_weighted_greedy_pct=wd.gap_greedy.mean())
    s["optimal_pct"] = pct
    key = t[t.n.isin([100, 1000]) & t.scheme.isin(["unit", "random"])]
    s["headline"] = key[["cls", "scheme", "n", "S_exact", "S_ours", "W_exact", "W_ours", "gap_W_pct", "opt_W_pct"]] \
        .round(3).to_dict("records")
    (RES / "exact_vs_ours_summary.json").write_text(json.dumps(s, indent=2), encoding="utf-8")
    return s


def main():
    val = gate()
    FIG.mkdir(parents=True, exist_ok=True)
    df = load()
    pg = per_graph(df)
    figure(pg, "unit", 1, "plot1_cardinality_unit", "Plot 1 · Cardinality |S| — unit weights (all w = 1)")
    for sch in SCHEMES:
        figure(pg, sch, 2, f"plot2_cardinality_{sch}",
               f"Plot 2 · Cardinality |S| — weighted graphs ({SCHEME_LABEL[sch]} weights)", caption=CAPTION2)
        figure(pg, sch, 3, f"plot3_weight_{sch}",
               f"Plot 3 · Total weight W(S) — weighted graphs ({SCHEME_LABEL[sch]} weights)")
    combined(pg)
    t = table(pg)
    s = summary(df, pg, t, val)
    print(json.dumps({k: s[k] for k in ("graphs", "solutions_checked", "exact_not_proven", "ilp_crosscheck")}, indent=1))
    print(json.dumps(s["optimal_pct"], indent=1))
    print(pd.DataFrame(s["headline"]).to_string())


if __name__ == "__main__":
    main()
