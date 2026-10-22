"""Aggregate the evaluation CSVs into results/summary.json, CSV + LaTeX tables and figures.

    python scripts/analyze.py

Before aggregating, EVERY row is re-verified independently: the graph is rebuilt from its stored spec, the
weights are re-assigned from (scheme, weight seed), the stored solution must pass is_total_dominating_set, and the
stored weight must equal W(S) recomputed from the solution. Gaps use only instances whose ILP optimum was PROVEN.
"""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from wtdsws.checker import is_total_dominating_set
from wtdsws.generators import make_graph
from wtdsws.graph_classes import CLASSES
from wtdsws.weights import SCHEMES, weighted_graph

RES = ROOT / ("results/smoke" if "--smoke" in sys.argv else "results")
FIG = RES / "figures"
TAB = RES / "tables"
TEAL, AMBER, SLATE, INK, MUTED, LINE, PAPER = "#0F766E", "#D97706", "#64748B", "#1F2328", "#6B6F76", "#E4E0D8", "#F7F5F0"
CLS_LABEL = {"er": "ER", "ba": "BA", "chordal": "Chordal", "bipartite": "Bipartite"}
SCH_LABEL = {"degree": "Degree", "inverse": "Inverse", "random": "Random", "mixed": "Mixed"}
SEEDED = ("ours_rr", "ours_rr_ls", "zero_rr")
SINGLE = ("greedy", "greedy_rr", "greedy_rr_ls")
NAMES = {"greedy": "weighted_greedy", "greedy_rr": "weighted_greedy_rr", "greedy_rr_ls": "weighted_greedy_rr_ls",
         "zero_rr": "wtds_zero_shot+RR", "ours_rr": "ours+RR", "ours_rr_ls": "ours+RR+LS", "ilp": "ILP (30 s)"}


def load(part):
    p = RES / f"raw_{part}.csv"
    if not p.exists():
        return []
    with open(p, newline="") as f:
        return list(csv.DictReader(f))


def reverify(rows):
    """Rebuild each graph from its spec and check every stored solution and weight."""
    cache = {}
    for r in rows:
        key = (r["spec"], r["scheme"], r["weight_seed"])
        if key not in cache:
            base = make_graph(json.loads(r["spec"]))
            cache[key] = weighted_graph(base, r["scheme"], int(r["weight_seed"]))
            if len(cache) > 400:
                cache.pop(next(iter(cache)))
        g = cache[key]
        S = [int(v) for v in r["S"].split()]
        if not is_total_dominating_set(g.adj, S):
            raise AssertionError(f"re-verification FAILED: {r['method']} on {r['spec']} {r['scheme']}")
        W = float(g.w[S].sum())
        if abs(W - float(r["weight"])) > 1e-6 or len(S) != int(r["size"]):
            raise AssertionError(f"stored weight {r['weight']} != recomputed {W}")
    return len(rows)


def gkey(r):
    return (r["cls"], r["family"], int(r["n"]), int(r["gidx"]), r["scheme"])


def build(rows):
    """opt[graph] = proven optimum (or None); val[(method, seed)][graph] = (W, runtime)."""
    opt, ilp_rt, val = {}, {}, defaultdict(dict)
    for r in rows:
        if r["method"] == "ilp":
            opt[gkey(r)] = float(r["weight"]) if r["optimal"] == "True" else None
            ilp_rt[gkey(r)] = float(r["runtime"])
        else:
            val[(r["method"], r["model_seed"])][gkey(r)] = (float(r["weight"]), float(r["runtime"]))
    return opt, ilp_rt, val


def gaps(val, opt, method, seed, sel):
    """Per-graph gaps on proven instances among graphs matching `sel`."""
    d = val.get((method, seed), {})
    return [(W - opt[k]) / opt[k] for k, (W, _) in d.items() if sel(k) and opt.get(k) is not None]


def stat(val, opt, method, sel):
    """mean gap; for seeded methods mean +- std (ddof=1) over the 2 model seeds of the per-seed mean gap."""
    if method in SEEDED:
        per = [np.mean(g) for s in ("0", "1") if (g := gaps(val, opt, method, s, sel))]
        if not per:
            return None, None
        return float(np.mean(per)), (float(np.std(per, ddof=1)) if len(per) > 1 else None)
    g = gaps(val, opt, method, "", sel)
    return (float(np.mean(g)), None) if g else (None, None)


def fmt(m, s=None):
    if m is None:
        return "—"
    return f"{100 * m:.2f} ± {100 * s:.2f}" if s is not None else f"{100 * m:.2f}"


def runtime(val, method, sel):
    seeds = ("0", "1") if method in SEEDED else ("",)
    xs = [rt for s in seeds for k, (_, rt) in val.get((method, s), {}).items() if sel(k)]
    return float(np.mean(xs)) if xs else None


def mean_w(val, method, sel):
    seeds = ("0", "1") if method in SEEDED else ("",)
    per = [np.mean([W for k, (W, _) in val.get((method, s), {}).items() if sel(k)]) for s in seeds]
    per = [p for p in per if p == p]
    return (float(np.mean(per)), float(np.std(per, ddof=1)) if len(per) > 1 else None) if per else (None, None)


def latex(rows, path, caption):
    cols = list(rows[0].keys())
    esc = lambda x: str(x).replace("%", r"\%").replace("±", r"$\pm$").replace("_", r"\_").replace("—", "--")
    lines = [r"\begin{table}[t]", r"\centering", r"\small", rf"\caption{{{esc(caption)}}}",
             r"\begin{tabular}{" + "l" * 2 + "r" * (len(cols) - 2) + "}", r"\toprule",
             " & ".join(esc(c) for c in cols) + r" \\", r"\midrule"]
    lines += [" & ".join(esc(r[c]) for c in cols) + r" \\" for r in rows]
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_csv(rows, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


# ------------------------------------------------------------------ figures
def style(plt):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": LINE,
                         "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
                         "grid.color": LINE, "grid.linewidth": 0.6, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.facecolor": "white", "axes.facecolor": "white",
                         "legend.frameon": False, "axes.titlecolor": INK, "axes.titleweight": "bold"})


def heatmap(M, title, path, vmax):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    style(plt)
    cmap = LinearSegmentedColormap.from_list("teal", ["#F2F8F7", "#7FBDB6", TEAL, "#0A4F4A"])
    fig, ax = plt.subplots(figsize=(5.6, 3.6), dpi=150)
    A = np.array([[np.nan if x is None else 100 * x for x in row] for row in M])
    im = ax.imshow(A, cmap=cmap, vmin=0, vmax=100 * vmax, aspect="auto")
    ax.grid(False)
    ax.set_xticks(range(len(SCHEMES)), [SCH_LABEL[s] for s in SCHEMES])
    ax.set_yticks(range(len(CLASSES)), [CLS_LABEL[c] for c in CLASSES])
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            v = A[i, j]
            ax.text(j, i, "—" if np.isnan(v) else f"{v:.2f}%", ha="center", va="center", fontsize=9,
                    color="white" if (not np.isnan(v) and v > 0.55 * 100 * vmax) else INK)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("mean gap to ILP optimum (%)", color=MUTED)
    cb.outline.set_visible(False)
    ax.set_title(title, loc="left", fontsize=10.5)
    fig.tight_layout()
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def line_panels(series_by_scheme, ns, ylabel, title, path, logy=False, xlabel="n (vertices)"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    style(plt)
    fig, axes = plt.subplots(1, len(series_by_scheme), figsize=(max(3.1 * len(series_by_scheme), 6.4), 3.4), dpi=150,
                             sharey=True, squeeze=False)
    sty = {"ours+RR": (TEAL, "o", "-"), "ours+RR+LS": (TEAL, "s", "--"), "weighted_greedy_rr": (AMBER, "^", "-"),
           "wtds_zero_shot+RR": (SLATE, "D", ":"), "ILP (30 s)": (SLATE, "v", "-.")}
    for ax, (panel, series) in zip(axes[0], series_by_scheme.items()):
        for name, ys in series.items():
            c, mk, ls = sty[name]
            xs = [n for n, y in zip(ns, ys) if y is not None]
            yv = [y for y in ys if y is not None]
            ax.plot(xs, yv, color=c, marker=mk, ls=ls, lw=2, ms=6, label=name, mec="white", mew=1.2)
        if len(series_by_scheme) > 1:
            ax.set_title(panel, loc="left")
        ax.set_xscale("log")
        ax.set_xticks(ns, [str(n) for n in ns])
        ax.minorticks_off()
        if logy:
            ax.set_yscale("log")
        ax.set_xlabel(xlabel)
    axes[0][0].set_ylabel(ylabel)
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=len(l), bbox_to_anchor=(0.5, 1.02), fontsize=9)
    fig.suptitle(title, x=0.01, y=1.10, ha="left", fontsize=11, color=INK, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def weight_hist(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    style(plt)
    ws = defaultdict(list)
    seen = set()
    for r in rows:
        k = (r["spec"], r["scheme"], r["weight_seed"])
        if k in seen or int(r["n"]) > 200:
            continue
        seen.add(k)
        g = weighted_graph(make_graph(json.loads(r["spec"])), r["scheme"], int(r["weight_seed"]))
        ws[r["scheme"]].extend(g.w.tolist())
    fig, axes = plt.subplots(1, 4, figsize=(12, 2.8), dpi=150, sharey=False)
    for ax, s in zip(axes, SCHEMES):
        ax.hist(ws[s], bins=np.arange(0.5, 101.5, 5), color=TEAL, edgecolor="white", linewidth=1)
        ax.set_title(f"{SCH_LABEL[s]}  (n_vertices = {len(ws[s]):,})", loc="left", fontsize=10)
        ax.set_xlabel("weight w(v)")
        ax.set_xlim(0, 101)
    axes[0].set_ylabel("vertices")
    fig.suptitle("Weight distribution per scheme (all test graphs with n ≤ 200, all classes)", x=0.01, ha="left",
                 fontsize=11, color=INK, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, facecolor="white")
    plt.close(fig)


# -------------------------------------------------------------------- main
def main():
    grid, sub = load("grid"), load("subclass")
    if not grid:
        print("results/raw_grid.csv missing")
        return
    FIG.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)
    print("re-verifying every stored solution ...", flush=True)
    checked = reverify(grid) + reverify(sub)
    print(f"  {checked} solutions re-verified: all valid, all weights recomputed", flush=True)
    opt, ilp_rt, val = build(grid)
    NS_ILP = sorted({k[2] for k in opt})
    NS_ALL = sorted({k[2] for d in val.values() for k in d})
    small = lambda k: k[2] in NS_ILP

    # ---- headline: class x scheme, n pooled over the ILP sizes
    head, H_ours, H_grr = [], [], []
    for c in CLASSES:
        ro, rg = [], []
        for s in SCHEMES:
            sel = lambda k, c=c, s=s: k[0] == c and k[4] == s and small(k)
            o, os_ = stat(val, opt, "ours_rr", sel)
            gr, _ = stat(val, opt, "greedy_rr", sel)
            ol, ols = stat(val, opt, "ours_rr_ls", sel)
            gl, _ = stat(val, opt, "greedy_rr_ls", sel)
            z, zs = stat(val, opt, "zero_rr", sel)
            ks = [k for k in opt if sel(k)]
            proven = sum(opt[k] is not None for k in ks)
            ro.append(o)
            rg.append(gr)
            head.append({"class": CLS_LABEL[c], "scheme": SCH_LABEL[s], "ours+RR gap %": fmt(o, os_),
                         "greedy+RR gap %": fmt(gr), "ours+RR+LS gap %": fmt(ol, ols), "greedy+RR+LS gap %": fmt(gl),
                         "zero-shot+RR gap %": fmt(z, zs), "proven / ILP runs": f"{proven} / {len(ks)}",
                         "ours+RR time (s)": f"{runtime(val, 'ours_rr', sel):.3f}",
                         "ILP time (s)": f"{np.mean([ilp_rt[k] for k in ks]):.3f}"})
        H_ours.append(ro)
        H_grr.append(rg)
    vmax = max(x for row in H_ours + H_grr for x in row if x is not None)
    heatmap(H_ours, "ours + RR: mean gap to ILP (%)", FIG / "fig_heatmap_ours_rr.png", vmax)
    heatmap(H_grr, "weighted greedy + RR: mean gap to ILP (%)", FIG / "fig_heatmap_greedy_rr.png", vmax)

    # ---- gap vs n per scheme (classes pooled)
    by_n, panels = [], {}
    for s in SCHEMES:
        series = defaultdict(list)
        for n in NS_ILP:
            sel = lambda k, s=s, n=n: k[4] == s and k[2] == n
            row = {"scheme": SCH_LABEL[s], "n": n}
            for m in ("ours_rr", "ours_rr_ls", "greedy_rr", "zero_rr"):
                mm, sd = stat(val, opt, m, sel)
                row[f"{NAMES[m]} gap %"] = fmt(mm, sd)
                if m != "ours_rr_ls":
                    series[NAMES[m]].append(None if mm is None else 100 * mm)
            ks = [k for k in opt if sel(k)]
            row["non-proven"] = sum(opt[k] is None for k in ks)
            by_n.append(row)
        panels[SCH_LABEL[s]] = dict(series)
    line_panels(panels, NS_ILP, "mean gap to ILP (%)", "Gap vs n per weight scheme (classes pooled; proven instances)",
                FIG / "fig_gap_vs_n.png")

    # ---- runtime vs n
    rt_rows, rts = [], {"ours+RR": [], "weighted_greedy_rr": [], "ILP (30 s)": []}
    for n in NS_ALL:
        sel = lambda k, n=n: k[2] == n
        row = {"n": n}
        for m in ("greedy", "greedy_rr", "greedy_rr_ls", "zero_rr", "ours_rr", "ours_rr_ls"):
            t = runtime(val, m, sel)
            row[f"{NAMES[m]} (s)"] = "—" if t is None else f"{t:.4f}"
        ti = [ilp_rt[k] for k in ilp_rt if k[2] == n]
        row["ILP (30 s) (s)"] = f"{np.mean(ti):.4f}" if ti else "not run"
        rt_rows.append(row)
        rts["ours+RR"].append(runtime(val, "ours_rr", sel))
        rts["weighted_greedy_rr"].append(runtime(val, "greedy_rr", sel))
        rts["ILP (30 s)"].append(float(np.mean(ti)) if ti else None)
    line_panels({"All classes and schemes": rts}, NS_ALL, "mean runtime per instance (s, log scale)",
                "Runtime vs n (all classes and schemes; ours on GPU, ILP = CP-SAT 8 workers)",
                FIG / "fig_runtime_vs_n.png", logy=True)

    # ---- n = 500: weights only (no ILP)
    big = [n for n in NS_ALL if n not in NS_ILP]
    large = []
    for n in big:
        for c in CLASSES:
            for s in SCHEMES:
                sel = lambda k, c=c, s=s, n=n: k[0] == c and k[4] == s and k[2] == n
                o, osd = mean_w(val, "ours_rr", sel)
                g_, _ = mean_w(val, "greedy_rr", sel)
                ol, olsd = mean_w(val, "ours_rr_ls", sel)
                gl, _ = mean_w(val, "greedy_rr_ls", sel)
                large.append({"class": CLS_LABEL[c], "scheme": SCH_LABEL[s], "n": n,
                              "ours+RR mean W": f"{o:.1f} ± {osd:.1f}", "greedy+RR mean W": f"{g_:.1f}",
                              "ours+RR vs greedy+RR %": f"{100 * (o - g_) / g_:+.2f}",
                              "ours+RR+LS mean W": f"{ol:.1f} ± {olsd:.1f}", "greedy+RR+LS mean W": f"{gl:.1f}",
                              "ours+RR time (s)": f"{runtime(val, 'ours_rr', sel):.3f}",
                              "greedy+RR time (s)": f"{runtime(val, 'greedy_rr', sel):.4f}"})

    # ---- sub-class sanity check
    subrows = []
    if sub:
        sopt, srt, sval = build(sub)
        for fam, lab in (("unit_interval", "unit interval"), ("convex_bip", "convex bipartite")):
            for n in sorted({k[2] for k in sopt}):
                for s in SCHEMES:
                    sel = lambda k, f=fam, s=s, n=n: k[1] == f and k[4] == s and k[2] == n
                    o, osd = stat(sval, sopt, "ours_rr", sel)
                    gr, _ = stat(sval, sopt, "greedy_rr", sel)
                    ks = [k for k in sopt if sel(k)]
                    subrows.append({"subclass": lab, "scheme": SCH_LABEL[s], "n": n, "ours+RR gap %": fmt(o, osd),
                                    "greedy+RR gap %": fmt(gr),
                                    "proven / ILP runs": f"{sum(sopt[k] is not None for k in ks)} / {len(ks)}"})

    # ---- overall per method (pooled over all classes, schemes, ILP sizes)
    overall = []
    for m in ("greedy", "greedy_rr", "greedy_rr_ls", "zero_rr", "ours_rr", "ours_rr_ls"):
        mm, sd = stat(val, opt, m, small)
        frac = None
        seeds = ("0", "1") if m in SEEDED else ("",)
        hits = [float(abs(W - opt[k]) < 1e-9) for s in seeds for k, (W, _) in val.get((m, s), {}).items()
                if small(k) and opt.get(k) is not None]
        overall.append({"method": NAMES[m], "mean gap %": fmt(mm, sd),
                        "optimal found %": f"{100 * np.mean(hits):.1f}" if hits else "—",
                        "mean runtime (s)": f"{runtime(val, m, small):.4f}"})

    n_ilp = len(opt)
    n_proven = sum(v is not None for v in opt.values())
    tables = {"headline": head, "overall": overall, "gap_vs_n": by_n, "runtime": rt_rows}
    captions = {
        "headline": {"title": "Headline: gap to ILP by class × scheme (n = " + ", ".join(map(str, NS_ILP)) + " pooled)",
                     "text": "Mean (W(S) − W*) / W* over instances whose ILP optimum was proven within 30 s; seeded "
                             "methods show mean ± std (ddof = 1) over the two model seeds. Runtimes are per instance."},
        "overall": {"title": "All classes and schemes pooled",
                    "text": "Same proven instances; 'optimal found' = share of runs whose W(S) equals the proven optimum."},
        "gap_vs_n": {"title": "Gap vs n per scheme (classes pooled)",
                     "text": "'non-proven' = ILP runs that hit the 30 s limit (excluded from the gaps)."},
        "runtime": {"title": "Mean runtime per instance vs n",
                    "text": "Greedy on CPU; GNN rollouts on the GPU (RTX 3050); ILP = CP-SAT, 8 workers, 30 s limit."},
    }
    if large:
        tables["n500"] = large
        captions["n500"] = {"title": "n = 500 (no ILP): mean W(S)",
                            "text": "Negative 'ours vs greedy' = ours is cheaper. Mean ± std over the two model seeds."}
    if subrows:
        tables["subclass"] = subrows
        captions["subclass"] = {"title": "Sub-class sanity check: unit interval and convex bipartite",
                                "text": "These subclasses have exact polynomial algorithms for weighted domination "
                                        "variants (Lee, 2025); reported only as a sanity check."}
    for k, rows in tables.items():
        write_csv(rows, TAB / f"{k}.csv")
        latex(rows, TAB / f"{k}.tex", captions[k]["title"])
    weight_hist(grid, FIG / "fig_weight_hist.png")

    # heatmap data, machine-readable
    hm = {"ours_rr": {}, "greedy_rr": {}}
    for i, c in enumerate(CLASSES):
        hm["ours_rr"][c] = {s: H_ours[i][j] for j, s in enumerate(SCHEMES)}
        hm["greedy_rr"][c] = {s: H_grr[i][j] for j, s in enumerate(SCHEMES)}
    summary = {"validity": {"solutions_checked": checked, "valid_fraction": 1.0,
                            "note": "every stored solution re-verified from the rebuilt graph; W(S) recomputed"},
               "ilp": {"run": n_ilp, "proven": n_proven, "not_proven": n_ilp - n_proven, "time_limit": 30.0},
               "graphs": len({gkey(r) for r in grid}), "grid_note": "class × scheme × n, 10 graphs per cell",
               "n_ilp": NS_ILP, "n_all": NS_ALL, "heatmap": hm, "tables": tables, "captions": captions,
               "findings": json.loads((RES / "findings.json").read_text(encoding="utf-8"))
               if (RES / "findings.json").exists() else []}
    (RES / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote results/summary.json, {len(tables)} tables (CSV + LaTeX), 5 figures; ILP proven {n_proven}/{n_ilp}")


if __name__ == "__main__":
    main()
