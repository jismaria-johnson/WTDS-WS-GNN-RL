"""Lattice experiment: tables, figures, report.json and SUMMARY.md from results/lattices/raw.csv (scripts/lat_run.py).

    python scripts/lat_report.py

Reference per instance = CP-SAT W if proven optimal, otherwise CP-SAT's lower bound (column ref_kind = "opt" / "LB";
gaps to an LB are always labelled). Ours: per instance, mean over the 5 model seeds; "± std" = std over model seeds
(of the per-seed mean over weight seeds for random / mixed); 95% CI = t-interval over the 5 seed means.
"Proposed beats Greedy in X%": strict W_ours < W_greedy over (instance, model seed) pairs. Wilcoxon (two-sided, scipy
default) per lattice type on per-instance seed-mean W vs greedy W.
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

from labels import DISP  # noqa: E402
from stats import fmt_p, seed_ci, tex_table, wilcoxon_p  # noqa: E402

LT = ROOT / "results" / "lattices"
FIG = LT / "figures"
KINDS = ["grid", "tri", "hex"]
KLAB = {"grid": "Grid (square)", "tri": "Triangular", "hex": "Hexagonal"}
SCH = ["unit", "degree", "inverse", "random", "mixed"]
INST = ["kind", "rows", "cols", "scheme", "wseed_idx"]
EXACT, OURS, GREEDY = "#0b0b0b", "#0f8a8a", "#7a7a76"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
                     "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb"})


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def load():
    d = pd.read_csv(LT / "raw.csv", dtype={"wseed_idx": str, "model_seed": str})
    d["wseed_idx"] = d.wseed_idx.fillna("")
    d["model_seed"] = d.model_seed.fillna("")
    il = d[d.method == "ilp"].set_index(["part"] + INST)
    il = il.assign(proven=il.proven.astype(str) == "True")
    il["ref"] = np.where(il.proven, il.weight, il.bound)
    il["ref_kind"] = np.where(il.proven, "opt", "LB")
    d = d.join(il[["ref", "ref_kind", "proven", "weight"]].rename(columns={"proven": "ilp_proven", "weight": "ilp_w"}),
               on=["part"] + INST)
    d["gap"] = 100 * (d.weight - d.ref) / d.ref
    bad = d[(d.ref_kind == "opt") & (d.weight < d.ref - 1e-6)]
    assert bad.empty, "a heuristic below a proven optimum"
    return d, il


def wide(x, method):
    """Per instance: seed-mean W (ours) or W (greedy)."""
    return x[x.method == method].groupby(INST).weight.mean()


def beats(x, ours, base):
    o = x[x.method == ours].set_index(INST)
    b = x[x.method == base].set_index(INST).weight
    bw = b.reindex(o.index).values
    ow = o.weight.values
    n = len(ow)
    return (dict(win=100 * float((ow < bw - 1e-9).sum()) / n, tie=100 * float((np.abs(ow - bw) <= 1e-9).sum()) / n,
                 loss=100 * float((ow > bw + 1e-9).sum()) / n, pairs=n) if n else None)


def main():
    d, il = load()
    rep = {}
    lat, gmk = d[d.part == "lat"], d[d.part == "gmk"]
    # ---- closed form
    cf = il[il.closed_form.notna() & (il.closed_form.astype(str) != "")]
    cf = cf.assign(cfv=cf.closed_form.astype(float))
    rep["closed_form"] = dict(checked=int(cf.proven.sum()), unproven=int((~cf.proven).sum()),
                              mismatches=int(((cf["size"] != cf.cfv) & cf.proven).sum()),
                              rows=[dict(n=int(k[2]), part=k[0], cf=int(r.cfv), cpsat=int(r["size"]), proven=bool(r.proven))
                                    for k, r in cf.iterrows()])
    # ---- G(m, k) table
    rows = []
    for (m, k), x in gmk.groupby(["rows", "cols"]):
        ex = x[x.method == "ilp"].iloc[0]
        ours = x[x.method == "ours_rr"]["size"]
        gr = x[x.method == "greedy_rr"]["size"].iloc[0]
        cfv = gt_cf = (ex.closed_form if str(ex.closed_form) not in ("", "nan") else "")
        rows.append(dict(m=m, k=k, n=int(ex.n), exact=int(ex["size"]), proven=bool(str(ex.proven) == "True"),
                         closed_form="" if cfv == "" else int(float(gt_cf)), ours=float(ours.mean()), greedy=int(gr),
                         d_ours=float(ours.mean() - ex["size"]), d_greedy=int(gr - ex["size"]),
                         ours_opt_seeds=int((ours == ex["size"]).sum()), ours_opt_all=bool((ours == ex["size"]).all())))
    g = pd.DataFrame(rows)
    rep["gmk"] = dict(grids=len(g), mean_d_ours=float(g.d_ours.mean()), mean_d_greedy=float(g.d_greedy.mean()),
                      pct_ours_optimal_all_seeds=float(100 * g.ours_opt_all.mean()),
                      pct_ours_optimal_seed_runs=float(100 * g.ours_opt_seeds.sum() / (5 * len(g))),
                      pct_greedy_optimal=float(100 * (g.d_greedy == 0).mean()), unproven=int((~g.proven).sum()),
                      ours_better_than_greedy_pct=float(100 * (g.ours < g.greedy - 1e-9).mean()),
                      ours_worse_than_greedy_pct=float(100 * (g.ours > g.greedy + 1e-9).mean()))
    gt = g.assign(**{"gamma_t (CP-SAT)": g.exact.astype(str) + np.where(g.proven, "", "*"),
                     "closed form": g.closed_form, "Proposed + RR": g.ours.map("{:.1f}".format),
                     "Greedy + RR": g.greedy, "D Proposed": g.d_ours.map("{:+.1f}".format), "D greedy": g.d_greedy.map("{:+d}".format)})
    gt = gt[["m", "k", "n", "gamma_t (CP-SAT)", "closed form", "Proposed + RR", "Greedy + RR", "D Proposed", "D greedy"]]
    s = rep["gmk"]
    gt.loc[len(gt)] = ["mean", "", "", "", "", "", "", f"{s['mean_d_ours']:+.2f}", f"{s['mean_d_greedy']:+.2f}"]
    gt.loc[len(gt)] = ["Proposed optimal", "", "", "", "", f"{s['pct_ours_optimal_all_seeds']:.0f}% (all 5 seeds)",
                       f"greedy {s['pct_greedy_optimal']:.0f}%", "", ""]
    tex_table(gt, LT / "table_grid_mxk.tex", "Rectangular grids G(m, k), 4 <= m <= k <= 10, unit weights (base-paper "
              "style). gamma_t = CP-SAT (60 s; * = not proven); closed form (2a+1)(2a+r), n = 4a+r, for square grids. "
              "Proposed + RR = mean over 5 model seeds. D = |S| - gamma_t.", "tab:grid_mxk")
    # ---- per lattice x n x scheme
    trows, rep["cells"] = [], {}
    for (kind, n, sch), x in lat.groupby(["kind", "n", "scheme"]):
        ilx = x[x.method == "ilp"]
        c = dict(kind=kind, n=int(n), m=int(x.m.iloc[0]), scheme=sch, instances=int(len(ilx)),
                 proven=f"{int((ilx.proven.astype(str) == 'True').sum())}/{len(ilx)}",
                 ref_kind="opt" if (ilx.proven.astype(str) == "True").all() else "LB (some unproven)",
                 W_exact=float(ilx.weight.mean()), S_exact=float(ilx["size"].mean()))
        for meth in ("ours_rr", "ours_rr_ls", "greedy_rr", "greedy_rr_ls"):
            y = x[x.method == meth]
            if y.empty:
                continue
            if meth.startswith("ours"):
                ps = y.groupby("model_seed")
                c[f"W_{meth}"], c[f"W_{meth}_std"] = float(ps.weight.mean().mean()), float(ps.weight.mean().std(ddof=1))
                c[f"S_{meth}"], c[f"S_{meth}_std"] = float(ps["size"].mean().mean()), float(ps["size"].mean().std(ddof=1))
                gseed = ps.gap.mean()
                c[f"gap_{meth}"], c[f"gap_{meth}_ci"] = float(gseed.mean()), seed_ci(gseed.values)
            else:
                c[f"W_{meth}"], c[f"S_{meth}"] = float(y.weight.mean()), float(y["size"].mean())
                c[f"W_{meth}_std"] = float(y.weight.std(ddof=1)) if len(y) > 1 else 0.0
                c[f"S_{meth}_std"] = float(y["size"].std(ddof=1)) if len(y) > 1 else 0.0
                c[f"gap_{meth}"] = float(y.gap.mean())
            c[f"time_{meth}"] = float(y.runtime.mean())
            if meth.startswith("ours"):
                c["device"] = y.device.iloc[0]
        c["time_ilp"] = float(ilx.runtime.mean())
        c["beats_rr"] = beats(x, "ours_rr", "greedy_rr")
        c["beats_rr_ls"] = beats(x, "ours_rr_ls", "greedy_rr_ls")
        rep["cells"][f"{kind}|{n}|{sch}"] = c
        f = lambda k: f"{c[k]:.1f}" if k in c else "--"  # noqa: E731
        trows.append(dict(lattice=KLAB[kind], n=int(n), m=c["m"], scheme=sch, W_exact=f"{c['W_exact']:.1f}",
                          proven=c["proven"], ours_rr=f"{c['W_ours_rr']:.1f} ± {c['W_ours_rr_std']:.1f}",
                          greedy_rr=f("W_greedy_rr"),
                          ours_rr_ls=(f"{c['W_ours_rr_ls']:.1f} ± {c['W_ours_rr_ls_std']:.1f}" if "W_ours_rr_ls" in c else "--"),
                          greedy_rr_ls=f("W_greedy_rr_ls"),
                          gap_ours_rr=f"{c['gap_ours_rr']:.2f}", gap_greedy_rr=f"{c['gap_greedy_rr']:.2f}",
                          gap_ours_rr_ls=(f"{c['gap_ours_rr_ls']:.2f}" if "gap_ours_rr_ls" in c else "--"),
                          gap_greedy_rr_ls=(f"{c['gap_greedy_rr_ls']:.2f}" if "gap_greedy_rr_ls" in c else "--"),
                          gap_ref=c["ref_kind"]))
    tl = pd.DataFrame(trows).rename(columns={
        "ours_rr": "Proposed + RR", "greedy_rr": "Greedy + RR", "ours_rr_ls": "Proposed + RR + LS",
        "greedy_rr_ls": "Greedy + RR + LS", "gap_ours_rr": "gap Proposed + RR", "gap_greedy_rr": "gap Greedy + RR",
        "gap_ours_rr_ls": "gap Proposed + RR + LS", "gap_greedy_rr_ls": "gap Greedy + RR + LS", "W_exact": "W exact"})
    tl.to_csv(LT / "table_lattices.csv", index=False)
    parts = []
    for sch in SCH:
        sub = tl[tl.scheme == sch].drop(columns=["scheme"])
        tex_table(sub, LT / f"table_lattices_{sch}.tex", f"Lattices, {sch} weights: mean W(S) (Proposed = mean ± std over 5 "
                  "model seeds; random/mixed also averaged over 3 weight seeds), gap (%) to the CP-SAT optimum "
                  "(gap_ref = LB when some instance was not proven: gap to the lower bound).", f"tab:lattices_{sch}")
        parts.append((LT / f"table_lattices_{sch}.tex").read_text(encoding="utf-8"))
    (LT / "table_lattices.tex").write_text("\n".join(parts), encoding="utf-8")
    tl.to_csv(LT / "table_lattices.csv", index=False)
    # ---- pooled per lattice x scheme beats-greedy, and Wilcoxon per lattice type
    rep["pooled"] = {}
    for kind in KINDS:
        for sch in SCH + ["weighted (4 schemes)", "all"]:
            x = lat[lat.kind == kind]
            if sch == "weighted (4 schemes)":
                x = x[x.scheme != "unit"]
            elif sch != "all":
                x = x[x.scheme == sch]
            r = {}
            for o, b in (("ours_rr", "greedy_rr"), ("ours_rr_ls", "greedy_rr_ls")):
                bt = beats(x, o, b)
                if bt is None:
                    continue
                ow, bw = wide(x, o), wide(x, b)
                j = ow.index.intersection(bw.index)
                n_lb = int((x[x.method == "ilp"].ref_kind == "LB").sum())
                r[o] = dict(**bt, p=wilcoxon_p(ow[j].values, bw[j].values), n_inst=int(len(j)), n_lb=n_lb,
                            gap_ours=float(x[x.method == o].gap.mean()), gap_greedy=float(x[x.method == b].gap.mean()))
            rep["pooled"][f"{kind}|{sch}"] = r
    unpr = il[~il.proven]
    rep["unproven"] = [dict(part=k[0], kind=k[1], rows=k[2], cols=k[3], scheme=k[4], wseed_idx=k[5], W=float(r.weight),
                            LB=float(r.bound), gap_W_to_LB=float(100 * (r.weight - r.bound) / r.bound))
                       for k, r in unpr.iterrows()]
    figures(lat, rep)
    (LT / "report.json").write_text(json.dumps(rep, indent=1, default=float))
    summary(rep, g)
    print("wrote", LT / "report.json")


def series(x, scheme, metric):
    """Per n: (exact mean, ours mean, ours std, greedy mean, greedy std) of metric (size | weight | ratio)."""
    out = []
    for n, y in sorted(x[x.scheme == scheme].groupby("n"), key=lambda t: t[0]):
        if metric == "ratio":                     # W / W_exact with W_exact = CP-SAT's best value (exact line = 1)
            y = y.assign(val=y.weight / y.ilp_w)
        else:
            y = y.assign(val=y[metric])
        ex = y[y.method == "ilp"].val
        o = y[y.method == "ours_rr"].val
        gr = y[y.method == "greedy_rr"].val
        out.append((n, ex.mean(), o.mean(), o.std(ddof=1), gr.mean(), gr.std(ddof=1) if len(gr) > 1 else 0.0,
                    float((y[y.method == "ilp"].ref_kind == "LB").any())))
    return np.array(out, float)


def panel(ax, x, kind, scheme, metric, rep, ylabel):
    s = series(x[x.kind == kind], scheme, metric)
    ax.plot(s[:, 0], s[:, 1], color=EXACT, marker="o", ms=4, lw=1.6, label="Exact (CP-SAT)")
    un = s[:, 6] > 0                              # CP-SAT not proven optimal at this n: hollow marker
    if un.any():
        ax.plot(s[un, 0], s[un, 1], ls="none", marker="o", ms=6, mfc="white", mec=EXACT,
                label="CP-SAT best, not proven")
    ax.errorbar(s[:, 0], s[:, 2], yerr=s[:, 3], color=OURS, marker="s", ms=4, lw=1.6, capsize=2, label="Proposed + RR")
    ax.errorbar(s[:, 0], s[:, 4], yerr=s[:, 5], color=GREEDY, marker="^", ms=4, lw=1.4, ls="--", capsize=2,
                label="Greedy + RR")
    bt = rep["pooled"][f"{kind}|{scheme}"]["ours_rr"]
    ax.text(0.03, 0.97, f"Proposed beats Greedy in {np.floor(10 * bt['win']) / 10:.1f}% of instances"
            + (f"\n(ties {bt['tie']:.1f}%)" if bt["tie"] > 0 else ""), transform=ax.transAxes, va="top", fontsize=7,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=GRID))
    ax.set_title(KLAB[kind], color=INK, fontsize=9)
    ax.set_xlabel("n (vertices)")
    ax.set_ylabel(ylabel)


def figures(lat, rep):
    def three(scheme, metric, ylabel, name, title):
        fig, axs = plt.subplots(1, 3, figsize=(10, 3))
        for ax, k in zip(axs, KINDS):
            panel(ax, lat, k, scheme, metric, rep, ylabel)
        axs[0].legend(loc="lower right", fontsize=7)
        fig.suptitle(title, color=INK, fontsize=10, y=1.03)
        save(fig, name)

    three("unit", "size", "|S|", "fig1_size_vs_n_unit", "Unit weights: |S| vs n (error bars: std over 5 model seeds; "
          "hollow = CP-SAT not proven optimal in 60 s)")
    for sch in ("degree", "inverse", "random", "mixed"):
        three(sch, "weight", "W(S)", f"fig2_weight_vs_n_{sch}",
              f"{sch.capitalize()} weights: W(S) vs n (error bars: std over model seeds" +
              (" and 3 weight seeds)" if sch in ("random", "mixed") else ")"))
    three("degree", "ratio", "W / W_exact", "fig3_ratio_vs_n_degree",
          "Degree weights: W / W_exact vs n (W_exact = CP-SAT's best value; hollow = not proven optimal in 60 s)")
    fig, axs = plt.subplots(2, 3, figsize=(10, 6))
    for j, k in enumerate(KINDS):
        panel(axs[0, j], lat, k, "unit", "size", rep, "|S| (unit weights)")
        panel(axs[1, j], lat, k, "degree", "ratio", rep, "W / W_exact (degree weights)")
    axs[0, 0].legend(loc="lower right", fontsize=7)
    fig.tight_layout()
    save(fig, "fig4_paper_lattices")


def summary(rep, g):
    L = ["# Lattice experiment (grid, triangular, hexagonal; base paper Chen, Liu, He 2024)", "",
         "Every number below is generated by `scripts/lat_report.py` from `results/lattices/raw.csv` "
         "(`scripts/lat_run.py`; every set checked with both TDS checkers, W recomputed). Decisions: docs/decisions.md "
         "R34 onwards. Gap = 100 (W - ref) / ref with ref = proven CP-SAT optimum; cells containing an unproven instance "
         "are marked 'LB' (gap to the lower bound).", "",
         "## Closed-form check (unit square grids, gamma_t(n x n) = (2a+1)(2a+r), n = 4a+r)", ""]
    c = rep["closed_form"]
    L.append(f"- {c['checked']} square grids proven by CP-SAT, {c['mismatches']} mismatches, {c['unproven']} unproven.")
    L.append("- " + ", ".join(f"{r['n']}x{r['n']}: closed form {r['cf']}, CP-SAT {r['cpsat']}{'' if r['proven'] else ' (unproven)'}"
                              for r in c["rows"]))
    s = rep["gmk"]
    L += ["", "## G(m, k), 4 <= m <= k <= 10, unit weights (table_grid_mxk.tex)", "",
          f"- {s['grids']} grids ({s['unproven']} unproven). Mean Delta Proposed (mean over 5 seeds) = {s['mean_d_ours']:+.3f}, "
          f"mean Delta greedy = {s['mean_d_greedy']:+.3f}.",
          f"- Proposed optimal on all 5 seeds in {s['pct_ours_optimal_all_seeds']:.1f}% of grids; optimal in "
          f"{s['pct_ours_optimal_seed_runs']:.1f}% of (grid, seed) runs; greedy optimal in {s['pct_greedy_optimal']:.1f}%.",
          f"- Seed-mean |S| of Proposed below greedy in {s['ours_better_than_greedy_pct']:.1f}% of grids, above in "
          f"{s['ours_worse_than_greedy_pct']:.1f}%.", "",
          "| m | k | gamma_t | closed form | Proposed + RR (mean) | Greedy + RR | D Proposed | D greedy |", "|---|---|---|---|---|---|---|---|"]
    for _, r in g.iterrows():
        L.append(f"| {r.m} | {r.k} | {r.exact}{'' if r.proven else '*'} | {r.closed_form} | {r.ours:.1f} | {r.greedy} | "
                 f"{r.d_ours:+.1f} | {r.d_greedy:+d} |")
    L += ["", "## Lattices (n ~ 25-900): per lattice x n x scheme", "",
          "| lattice | n | m | scheme | proven | W exact | W Proposed + RR (± std) | W Greedy + RR | W Proposed + RR + LS | W Greedy + RR + LS | "
          "gap Proposed + RR | gap Greedy + RR | gap Proposed+LS | gap greedy+LS | ref | device | time Proposed + RR s | time CP-SAT s |",
          "|" + "---|" * 18]
    for k, c in rep["cells"].items():
        L.append(f"| {c['kind']} | {c['n']} | {c['m']} | {c['scheme']} | {c['proven']} | {c['W_exact']:.1f} | "
                 f"{c['W_ours_rr']:.2f} ± {c['W_ours_rr_std']:.2f} | {c['W_greedy_rr']:.2f} | "
                 f"{c.get('W_ours_rr_ls', float('nan')):.2f} | {c.get('W_greedy_rr_ls', float('nan')):.2f} | "
                 f"{c['gap_ours_rr']:.3f} | {c['gap_greedy_rr']:.3f} | {c.get('gap_ours_rr_ls', float('nan')):.3f} | "
                 f"{c.get('gap_greedy_rr_ls', float('nan')):.3f} | {c['ref_kind']} | {c.get('device', '')} | "
                 f"{c['time_ours_rr']:.3f} | {c['time_ilp']:.2f} |")
    L += ["", "## Proposed vs Greedy per lattice type (beats = strict wins over (instance, model seed) pairs; Wilcoxon on "
          "per-instance seed-mean W)", "", "| lattice | scheme | pair | beats % | ties % | loses % | gap Proposed % | gap greedy % | "
          "Wilcoxon p | instances | of which gap to LB |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for k, r in rep["pooled"].items():
        kind, sch = k.split("|")
        for o, x in r.items():
            L.append(f"| {kind} | {sch} | {DISP[o]} vs {DISP[o.replace('ours', 'greedy')]} | {x['win']:.1f} | {x['tie']:.1f} | {x['loss']:.1f} | "
                     f"{x['gap_ours']:.3f} | {x['gap_greedy']:.3f} | {fmt_p(x['p'])} | {x['n_inst']} | {x['n_lb']} |")
    L += ["", "## Unproven CP-SAT instances (60 s)", ""]
    if rep["unproven"]:
        for u in rep["unproven"]:
            L.append(f"- {u['part']} {u['kind']} {u['rows']}x{u['cols']} {u['scheme']}{u['wseed_idx']}: W {u['W']:.0f}, "
                     f"LB {u['LB']:.1f}, gap {u['gap_W_to_LB']:.3f}%")
    else:
        L.append("- none: every CP-SAT run proved optimality.")
    L += ["", "## Observations (generated from the data above)", "",
          "Gaps of cells with unproven instances are to the CP-SAT lower bound (counted in each sentence); win/tie/loss "
          "and Wilcoxon compare W directly and do not depend on the reference.", ""]
    for kind in KINDS:
        sent = []
        for sch in SCH:
            x = rep["pooled"][f"{kind}|{sch}"]["ours_rr"]
            verdict = ("Proposed better" if x["win"] > x["loss"] else "greedy better" if x["loss"] > x["win"] else "even")
            sent.append(f"{sch}: Proposed + RR wins {x['win']:.0f}% / ties {x['tie']:.0f}% / loses {x['loss']:.0f}% "
                        f"(gap {x['gap_ours']:.2f}% vs {x['gap_greedy']:.2f}%"
                        + (f", {x['n_lb']} of {x['n_inst']} instances vs LB" if x["n_lb"] else "")
                        + f", p {fmt_p(x['p'])}; {verdict})")
        y = rep["pooled"][f"{kind}|all"].get("ours_rr_ls")
        ls = (f" With LS: wins {y['win']:.0f}% / ties {y['tie']:.0f}% / loses {y['loss']:.0f}% "
              f"(gap {y['gap_ours']:.2f}% vs {y['gap_greedy']:.2f}%, p {fmt_p(y['p'])}).") if y else ""
        L.append(f"- **{KLAB[kind]}** - " + "; ".join(sent) + "." + ls)
    L += ["", "## Files", "", "- results/lattices/raw.csv (every run, with weight seeds), table_lattices.tex|csv "
          "(+ per-scheme .tex), table_grid_mxk.tex|csv, report.json",
          "- results/lattices/figures/fig1_size_vs_n_unit, fig2_weight_vs_n_<scheme>, fig3_ratio_vs_n_degree, "
          "fig4_paper_lattices (.png 300 dpi + .pdf)", "- scripts/lat_run.py, scripts/lat_report.py; logs/lat_run.log", ""]
    (LT / "SUMMARY.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
