"""Base-paper-style comparison: raw policy vs PLAIN greedy, and both + RR (results/raw_vs_greedy/).

    python scripts/rvg_report.py

Data (main test grid, n in {20, 50, 100, 200}, 640 instances, same graphs/seeds as results/raw_grid.csv):
  Greedy, Greedy + RR        results/raw_grid.csv (methods greedy, greedy_rr); the stored sets are re-checked here
  Proposed, Proposed + RR    seeds 0-1: results/ablations/eval_ref.csv; seeds 2-4: results/raw_vs_greedy/
                             eval_proposed_seeds234.csv (scripts/abl_eval.py --variant ref: greedy policy rollout,
                             raw set and RR set both checked with both TDS checkers, W recomputed). Proposed + RR is
                             asserted equal to the stored ours_rr rows (raw_grid.csv seeds 0-1, revision seeds 2-4).
Main set = 624 instances with a proven optimum; secondary = all 640 (gap to the CP-SAT lower bound for 16).
Gap = 100 (W - ref) / ref. Proposed per instance = mean over the 5 model seeds. CI = bootstrap over instances (10,000,
rng 0). Wilcoxon: per instance, Proposed seed-mean W vs Greedy W (two-sided, zero differences dropped). Win/tie/loss and
"Proposed beats Greedy in X%": strict, over (instance, model seed) pairs. Removed by RR = |S| before RR - |S| after RR.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from stats import boot_ci, fmt_p, tex_table, wilcoxon_p  # noqa: E402

OUT = ROOT / "results" / "raw_vs_greedy"
FIG = OUT / "figures"
KEY = ["cls", "scheme", "n", "gidx"]
SCH = ["degree", "inverse", "random", "mixed"]
CLS = ["er", "ba", "chordal", "bipartite"]
CLAB = {"er": "ER", "ba": "BA", "chordal": "Chordal", "bipartite": "Bipartite"}
TEAL, GREY, INK, INK2, GRID = "#0f8a8a", "#7a7a76", "#0b0b0b", "#52514e", "#e4e3df"
TEAL_L, GREY_L = "#8fcaca", "#c4c4c0"
M = ["greedy", "proposed", "greedy_rr", "proposed_rr"]
LAB = {"greedy": "Greedy", "proposed": "Proposed", "greedy_rr": "Greedy + RR", "proposed_rr": "Proposed + RR"}
STY = {"greedy": dict(color=GREY), "proposed": dict(color=TEAL),
       "greedy_rr": dict(color=GREY_L, hatch="///", edgecolor=GREY), "proposed_rr": dict(color=TEAL_L, hatch="///", edgecolor=TEAL)}
plt.rcParams.update({"font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
                     "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb",
                     "hatch.linewidth": 0.6})


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def load():
    import abl_eval
    from run_eval import checked
    grid = pd.read_csv(ROOT / "results" / "raw_grid.csv")
    grid = grid[grid.n <= 200]
    gr = grid[grid.method.isin(["greedy", "greedy_rr"])]
    # re-check the stored greedy sets (both checkers, W recomputed)
    graphs = {(it["cls"], it["scheme"], it["n"], it["gidx"]): it["g"] for it in abl_eval.instances((20, 50, 100, 200))}
    for _, r in gr.iterrows():
        S, W = checked(graphs[(r.cls, r.scheme, r.n, r.gidx)], [int(v) for v in str(r.S).split()], r.method)
        assert abs(W - r.weight) < 1e-9 and len(S) == r["size"]
    g = gr.pivot_table(index=KEY, columns="method", values=["weight", "size"])
    p = pd.concat([pd.read_csv(ROOT / "results" / "ablations" / "eval_ref.csv"),
                   pd.read_csv(OUT / "eval_proposed_seeds234.csv")], ignore_index=True)
    assert p.groupby("model_seed").size().to_dict() == {s: 640 for s in range(5)}
    # Proposed + RR must equal the stored ours_rr rows (seeds 0-1 raw_grid, 2-4 revision)
    st = pd.concat([grid[grid.method == "ours_rr"],
                    pd.read_csv(ROOT / "results" / "revision" / "raw_grid_rev_seeds.csv").query("method == 'ours_rr'")])
    st = st.set_index(KEY + ["model_seed"]).weight
    chk = p.set_index(KEY + ["model_seed"]).w_rr
    match = int((np.abs(chk - st.reindex(chk.index)) < 1e-9).sum())
    assert match == len(chk), f"Proposed + RR differs from stored rows: {len(chk) - match}"
    return g, p, match, len(gr)


def main():
    g, p, match, n_greedy = load()
    ref = p.groupby(KEY).ref_w.first()
    proven = p.groupby(KEY).proven.first().astype(str) == "True"
    W = pd.DataFrame({"greedy": g[("weight", "greedy")], "greedy_rr": g[("weight", "greedy_rr")],
                      "proposed": p.groupby(KEY).w_raw.mean(), "proposed_rr": p.groupby(KEY).w_rr.mean()})
    SZ = pd.DataFrame({"greedy": g[("size", "greedy")], "greedy_rr": g[("size", "greedy_rr")],
                       "proposed": p.groupby(KEY).size_raw.mean(), "proposed_rr": p.groupby(KEY).size_rr.mean()})
    W, SZ = W.reindex(ref.index), SZ.reindex(ref.index)
    GAP = W.apply(lambda c: 100 * (c - ref) / ref)
    pw = {k: p.pivot_table(index=KEY, columns="model_seed", values=v).reindex(ref.index) for k, v in
          (("proposed", "w_raw"), ("proposed_rr", "w_rr"))}
    pg = {k: pw[k].apply(lambda c: 100 * (c - ref) / ref) for k in pw}
    rep = {"reuse_check": dict(proposed_rr_rows_equal_to_stored=match, greedy_sets_rechecked=n_greedy)}

    def block(mask):
        r = {}
        for m in M:
            x = GAP[m][mask]
            r[m] = dict(gap=float(x.mean()), ci=boot_ci(x.values), size=float(SZ[m][mask].mean()))
            if m.startswith("proposed"):
                r[m]["per_seed"] = {int(s): float(pg[m][s][mask].mean()) for s in pg[m].columns}
        for a, b in (("proposed", "greedy"), ("proposed_rr", "greedy_rr")):
            o, bw = pw[a][mask], W[b][mask]
            dd = o.values - bw.values[:, None]
            r[f"{a}_vs_{b}"] = dict(p=wilcoxon_p(W[a][mask].values, bw.values), win=float(100 * (dd < -1e-9).mean()),
                                    tie=float(100 * (np.abs(dd) <= 1e-9).mean()), loss=float(100 * (dd > 1e-9).mean()),
                                    n=int(mask.sum()))
        r["removed_greedy"] = float((SZ.greedy - SZ.greedy_rr)[mask].mean())
        r["removed_proposed"] = float((SZ.proposed - SZ.proposed_rr)[mask].mean())
        return r

    for setname, base in (("main", proven), ("all640", pd.Series(True, index=ref.index))):
        R = {"pooled": block(base.values)}
        for s in SCH:
            R[s] = block((base & (ref.index.get_level_values("scheme") == s)).values)
        for c in CLS:
            R[c] = block((base & (ref.index.get_level_values("cls") == c)).values)
        for n in (20, 50, 100, 200):
            R[f"n{n}"] = block((base & (ref.index.get_level_values("n") == n)).values)
        rep[setname] = R
    rep["main"]["instances"], rep["all640"]["instances"] = int(proven.sum()), int(len(ref))
    lv = ref.index.get_level_values
    rep["main_by_n_scheme"] = {(s or "pooled"): {n: block((proven & (lv("n") == n) & ((lv("scheme") == s) if s else True)).values)
                                                 for n in (20, 50, 100, 200)} for s in [None] + SCH}
    tables(rep)
    figures(rep)
    (OUT / "report.json").write_text(json.dumps(rep, indent=1, default=float))
    summary(rep)
    print("wrote", OUT / "report.json")


def ci(x):
    return f"{x['gap']:.2f} [{x['ci'][0]:.2f}, {x['ci'][1]:.2f}]"


def tables(rep):
    for setname, cap in (("main", "624 instances with a proven optimum"),
                         ("all640", "all 640 instances; gap to the CP-SAT lower bound for the 16 unproven")):
        R, rows = rep[setname], []
        for k in ["pooled"] + SCH + CLS:
            r = R[k]
            a, b = r["proposed_vs_greedy"], r["proposed_rr_vs_greedy_rr"]
            rows.append([CLAB.get(k, k.capitalize()), a["n"], ci(r["greedy"]), ci(r["proposed"]), fmt_p(a["p"]),
                         f"{a['win']:.1f}/{a['tie']:.1f}/{a['loss']:.1f}", ci(r["greedy_rr"]), ci(r["proposed_rr"]),
                         fmt_p(b["p"]), f"{b['win']:.1f}/{b['tie']:.1f}/{b['loss']:.1f}",
                         f"{r['greedy']['size']:.2f} / {r['proposed']['size']:.2f}",
                         f"{r['removed_greedy']:.2f} / {r['removed_proposed']:.2f}"])
        name = "table_raw_vs_greedy" + ("" if setname == "main" else "_all640")
        tex_table(pd.DataFrame(rows, columns=["subset", "inst", "Greedy gap %", "Proposed gap %", "p (raw)",
                                              "W/T/L % (raw)", "Greedy + RR gap %", "Proposed + RR gap %", "p (+RR)",
                                              "W/T/L % (+RR)", "|S| raw G / P", "removed by RR G / P"]),
                  OUT / f"{name}.tex",
                  f"Raw policy vs plain greedy, and both + RR (main test grid, n = 20-200, {cap}). Gap % [95% "
                  "bootstrap CI]; Proposed = mean over 5 model seeds; p = paired Wilcoxon; W/T/L = Proposed "
                  "better/equal/worse over (instance, seed) pairs; G = Greedy, P = Proposed.", f"tab:{name}")


def pfmt(p):
    return "p<1e-4" if p < 1e-4 else f"p={p:.3f}"


def figures(rep):
    R = rep["main"]
    groups = ["pooled"] + SCH
    # (1) main grouped bars
    fig, ax = plt.subplots(figsize=(8, 3.4))
    wd = 0.2
    for j, m in enumerate(M):
        ys = [R[k][m]["gap"] for k in groups]
        err = [[R[k][m]["gap"] - R[k][m]["ci"][0] for k in groups], [R[k][m]["ci"][1] - R[k][m]["gap"] for k in groups]]
        ax.bar(np.arange(5) + (j - 1.5) * wd, ys, wd * 0.92, label=LAB[m], yerr=err, linewidth=0.6,
               error_kw=dict(ecolor=INK2, lw=0.8, capsize=2), **STY[m])
    for i, k in enumerate(groups):
        for off, key, a, b in ((-1, "proposed_vs_greedy", "greedy", "proposed"),
                               (1, "proposed_rr_vs_greedy_rr", "greedy_rr", "proposed_rr")):
            top = max(R[k][a]["ci"][1], R[k][b]["ci"][1])
            ax.text(i + off * wd, top + 0.25, pfmt(R[k][key]["p"]), ha="center", va="bottom", fontsize=6.5, color=INK)
    ax.set_xticks(range(5))
    ax.set_xticklabels(["Pooled"] + [s.capitalize() for s in SCH])
    ax.set_ylabel("mean gap to optimum (%)")
    pb = R["pooled"]
    ax.set_title(f"Proposed beats Greedy in {np.floor(10 * pb['proposed_vs_greedy']['win']) / 10:.1f}% of instances "
                 f"(raw); {np.floor(10 * pb['proposed_rr_vs_greedy_rr']['win']) / 10:.1f}% with RR on both",
                 color=INK, fontsize=9)
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.1), fontsize=7.5)
    ax.set_ylim(top=ax.get_ylim()[1] * 1.08)
    save(fig, "fig1_main_raw_vs_greedy")
    # (2) gap vs n, pooled and 2x2 per scheme
    ns = [20, 50, 100, 200]
    st = {"greedy": (GREY, "-", "^"), "proposed": (TEAL, "-", "s"), "greedy_rr": (GREY, "--", "^"),
          "proposed_rr": (TEAL, "--", "s")}

    def gapn(ax, scheme=None):
        x = rep["main_by_n_scheme"][scheme or "pooled"]
        for m in M:
            ys = [x[n][m]["gap"] for n in ns]
            lo = [x[n][m]["ci"][0] for n in ns]
            hi = [x[n][m]["ci"][1] for n in ns]
            c, ls, mk = st[m]
            ax.plot(ns, ys, color=c, ls=ls, marker=mk, ms=4, lw=1.6, label=LAB[m])
            ax.fill_between(ns, lo, hi, color=c, alpha=0.10, lw=0)
        ax.set_xscale("log")
        ax.set_xticks(ns)
        ax.set_xticklabels([str(n) for n in ns])
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_xlabel("n (vertices)")
        ax.set_ylabel("mean gap (%)")
        ax.set_title(scheme.capitalize() + " weights" if scheme else "All schemes", color=INK, fontsize=9)

    fig, ax = plt.subplots(figsize=(4.5, 3))
    gapn(ax)
    ax.legend(fontsize=7)
    save(fig, "fig2_gap_vs_n_pooled")
    fig, axs = plt.subplots(2, 2, figsize=(8, 5.5))
    for ax, s in zip(axs.flat, SCH):
        gapn(ax, s)
    axs[0, 0].legend(fontsize=7)
    fig.tight_layout()
    save(fig, "fig2_gap_vs_n_by_scheme")
    # (3) effect of RR dumbbells
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    for i, k in enumerate(groups):
        for off, (a, b, c, rm) in ((-0.15, ("greedy", "greedy_rr", GREY, "removed_greedy")),
                                   (0.15, ("proposed", "proposed_rr", TEAL, "removed_proposed"))):
            ya, yb = R[k][a]["gap"], R[k][b]["gap"]
            ax.annotate("", xy=(i + off, yb), xytext=(i + off, ya),
                        arrowprops=dict(arrowstyle="->", color=c, lw=1.6))
            ax.plot([i + off], [ya], "o", color=c, ms=5)
            ax.text(i + off, ya + 0.15, f"-{R[k][rm]:.2f} v", ha="center", va="bottom", fontsize=6.5, color=c)
    ax.set_xticks(range(5))
    ax.set_xticklabels(["Pooled"] + [s.capitalize() for s in SCH])
    ax.set_ylabel("mean gap (%): dot = raw, arrow head = + RR")
    ax.plot([], [], color=GREY, lw=1.6, label="Greedy -> Greedy + RR")
    ax.plot([], [], color=TEAL, lw=1.6, label="Proposed -> Proposed + RR")
    ax.legend(fontsize=7)
    ax.set_title("Effect of RR (label: mean vertices removed by RR per instance)", color=INK, fontsize=9)
    save(fig, "fig3_effect_of_rr")
    # (4) per class
    fig, ax = plt.subplots(figsize=(7, 3))
    for j, m in enumerate(M):
        ys = [R[c][m]["gap"] for c in CLS]
        err = [[R[c][m]["gap"] - R[c][m]["ci"][0] for c in CLS], [R[c][m]["ci"][1] - R[c][m]["gap"] for c in CLS]]
        ax.bar(np.arange(4) + (j - 1.5) * wd, ys, wd * 0.92, label=LAB[m], yerr=err, linewidth=0.6,
               error_kw=dict(ecolor=INK2, lw=0.8, capsize=2), **STY[m])
    for i, c in enumerate(CLS):
        for off, key, a, b in ((-1, "proposed_vs_greedy", "greedy", "proposed"),
                               (1, "proposed_rr_vs_greedy_rr", "greedy_rr", "proposed_rr")):
            top = max(R[c][a]["ci"][1], R[c][b]["ci"][1])
            ax.text(i + off * wd, top + 0.25, pfmt(R[c][key]["p"]), ha="center", va="bottom", fontsize=6.5)
    ax.set_xticks(range(4))
    ax.set_xticklabels([CLAB[c] for c in CLS])
    ax.set_ylabel("mean gap to optimum (%)")
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.1), fontsize=7.5)
    ax.set_title("Raw vs Greedy by graph class (624 proven instances, 95% CI)", color=INK, fontsize=9)
    save(fig, "fig4_by_class")


def summary(rep):
    L = ["# Raw policy vs plain greedy, and both + RR", "",
         "Every number below is generated by `scripts/rvg_report.py` (data: results/raw_grid.csv, "
         "results/ablations/eval_ref.csv, results/raw_vs_greedy/eval_proposed_seeds234.csv from scripts/abl_eval.py). "
         f"Reuse check: Proposed + RR equals the stored ours_rr rows on {rep['reuse_check']['proposed_rr_rows_equal_to_stored']}"
         f" of 3,200 (instance, seed) pairs; {rep['reuse_check']['greedy_sets_rechecked']} stored Greedy / Greedy + RR sets "
         "re-checked with both TDS checkers. Decisions: docs/decisions.md R40.", ""]
    for setname, title in (("main", f"Main set: {rep['main']['instances']} instances with a proven optimum"),
                           ("all640", "Secondary: all 640 instances (gap to the CP-SAT lower bound for the 16 unproven)")):
        R = rep[setname]
        L += [f"## {title}", "", "| subset | Greedy | Proposed (per seed) | Greedy + RR | Proposed + RR (per seed) | "
              "raw: p, W/T/L % | + RR: p, W/T/L % | removed by RR G / P | |S| raw G / P |", "|---|---|---|---|---|---|---|---|---|"]
        for k in ["pooled"] + SCH + CLS + ["n20", "n50", "n100", "n200"]:
            r = R[k]
            a, b = r["proposed_vs_greedy"], r["proposed_rr_vs_greedy_rr"]
            ps = lambda m: ", ".join(f"{v:.2f}" for v in r[m]["per_seed"].values())  # noqa: E731
            L.append(f"| {CLAB.get(k, k)} | {ci(r['greedy'])} | {ci(r['proposed'])} ({ps('proposed')}) | "
                     f"{ci(r['greedy_rr'])} | {ci(r['proposed_rr'])} ({ps('proposed_rr')}) | {fmt_p(a['p'])}, "
                     f"{a['win']:.1f}/{a['tie']:.1f}/{a['loss']:.1f} | {fmt_p(b['p'])}, {b['win']:.1f}/{b['tie']:.1f}/"
                     f"{b['loss']:.1f} | {r['removed_greedy']:.2f} / {r['removed_proposed']:.2f} | "
                     f"{r['greedy']['size']:.2f} / {r['proposed']['size']:.2f} |")
        L.append("")
    L += ["Gap % = mean [95% bootstrap CI]; W/T/L = Proposed better / equal / worse over (instance, model seed) pairs.",
          "", "## Files", "", "- table_raw_vs_greedy.tex|csv (main), table_raw_vs_greedy_all640.tex|csv",
          "- figures/fig1_main_raw_vs_greedy, fig2_gap_vs_n_pooled, fig2_gap_vs_n_by_scheme, fig3_effect_of_rr, "
          "fig4_by_class (.png 300 dpi + .pdf); report.json", ""]
    (OUT / "SUMMARY.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    import matplotlib.ticker  # noqa: F401
    main()
