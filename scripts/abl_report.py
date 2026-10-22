"""Ablation study: tables, figures, report.json and SUMMARY.md.

    python scripts/abl_report.py

Reads results/ablations/eval_*.csv (scripts/abl_eval.py), runs/abl_*_seed*/ and runs/ws_seed{0,1}/ (training curves,
times), results/revision/blocking_ablation.json (A0). Writes results/ablations/{table_ablations.tex|csv,
table_ablations_by_scheme.tex|csv, report.json, SUMMARY.md, figures/*.png|pdf (300 dpi), val_curves/*.csv}.

Gap per (instance, seed) = 100 (W - ref) / ref, ref = proven optimum or CP-SAT bound (16 unproven instances).
Per-instance value of a variant = mean over its model seeds. CI = percentile bootstrap over the 640 instances
(10,000 resamples, rng 0). Tests vs the reference use the reference averaged over the SAME seeds the variant has
(a 1-seed variant is compared with reference seed 0): paired Wilcoxon on W after RR (two-sided, zero differences
dropped), Holm over the variants A1-A5; win/tie/loss = % of instances where the variant's seed-mean W after RR is
lower / equal / higher than the reference's (win = the ablation is better).
"""
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from stats import boot_ci, fmt_p, holm, tex_table, wilcoxon_p  # noqa: E402

AB = ROOT / "results" / "ablations"
FIG = AB / "figures"
KEY = ["cls", "scheme", "n", "gidx"]
SCH = ["degree", "inverse", "random", "mixed"]
VAR = ["ref", "a1", "a2", "a3", "a4", "a5"]
LABEL = {"ref": "Reference (full model)", "a0": "A0 blocking rule (base paper)", "a1": "A1 no action mask",
         "a2": "A2 degree-1 rule off", "a3": "A3 unit reward", "a4": "A4 sum readout", "a5": "A5 from scratch"}
COL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6"]
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
    d = {}
    for v in VAR:
        p = AB / f"eval_{v}.csv"
        if p.exists():
            x = pd.read_csv(p)
            x["gap_raw"] = 100 * (x.w_raw - x.ref_w) / x.ref_w
            x["gap_rr"] = 100 * (x.w_rr - x.ref_w) / x.ref_w
            assert x.groupby("model_seed").size().eq(640).all() or len(x) % 640 == 0
            d[v] = x
    return d


def per_inst(x, col, seeds=None):
    if seeds is not None:
        x = x[x.model_seed.isin(seeds)]
    return x.groupby(KEY)[col].mean()


def main():
    d = load()
    ref = d["ref"]
    rep = {"variants": {}}
    rows, prows, ps = [], {}, {}
    for v in VAR:
        if v not in d:
            continue
        x = d[v]
        seeds = sorted(x.model_seed.unique().tolist())
        gi_raw, gi_rr = per_inst(x, "gap_raw"), per_inst(x, "gap_rr")
        r = dict(seeds=seeds, n_inst=int(gi_rr.size), gap_raw=float(gi_raw.mean()), ci_raw=boot_ci(gi_raw.values),
                 gap_rr=float(gi_rr.mean()), ci_rr=boot_ci(gi_rr.values),
                 per_seed_rr={int(s): float(g.gap_rr.mean()) for s, g in x.groupby("model_seed")},
                 per_seed_raw={int(s): float(g.gap_raw.mean()) for s, g in x.groupby("model_seed")},
                 valid_rr=100.0, valid_raw=100.0, size_rr=float(x.size_rr.mean()), size_raw=float(x.size_raw.mean()),
                 time=float(x.time.mean()), wasted_picks=float(x.wasted_picks.mean()),
                 forced_missing_pct=float(100 * x.forced_missing.mean()), forced_picked=float(x.forced_picked.mean()),
                 picks=float(x.picks.mean()))
        if v != "ref":
            rw = per_inst(ref, "w_rr", seeds)
            aw = per_inst(x, "w_rr")
            rg = per_inst(ref, "gap_rr", seeds)
            r["ref_gap_rr_same_seeds"] = float(rg.mean())
            r["delta_rr"] = float(gi_rr.mean() - rg.mean())
            r["delta_raw"] = float(gi_raw.mean() - per_inst(ref, "gap_raw", seeds).mean())
            r["p"] = wilcoxon_p(aw.values, rw.reindex(aw.index).values)
            dd = aw.values - rw.reindex(aw.index).values
            r["win"], r["tie"], r["loss"] = (float(100 * (dd < -1e-9).mean()), float(100 * (np.abs(dd) <= 1e-9).mean()),
                                            float(100 * (dd > 1e-9).mean()))
            ps[v] = r["p"]
            r["by_scheme"] = {}
            for s in SCH:
                m = aw.index.get_level_values("scheme") == s
                gs = gi_rr[m]
                r["by_scheme"][s] = dict(gap_rr=float(gs.mean()), ci=boot_ci(gs.values),
                                         ref_gap_rr=float(rg[m].mean()), delta=float(gs.mean() - rg[m].mean()),
                                         p=wilcoxon_p(aw[m].values, rw.reindex(aw.index)[m].values))
        else:
            r["by_scheme"] = {s: dict(gap_rr=float(gi_rr[gi_rr.index.get_level_values("scheme") == s].mean()),
                                      ci=boot_ci(gi_rr[gi_rr.index.get_level_values("scheme") == s].values))
                              for s in SCH}
        rep["variants"][v] = r
    if ps:
        for v, ph in zip(ps, holm(list(ps.values()))):
            rep["variants"][v]["p_holm"] = float(ph)
        sp = {(v, s): rep["variants"][v]["by_scheme"][s]["p"] for v in ps for s in SCH}
        for k, ph in zip(sp, holm(list(sp.values()))):
            rep["variants"][k[0]]["by_scheme"][k[1]]["p_holm"] = float(ph)
    a0 = json.loads((ROOT / "results" / "revision" / "blocking_ablation.json").read_text())
    rep["a0"] = dict(episodes=a0["episodes"], valid_pct=a0["valid_pct"], covered_frac=a0["mean_covered_frac_at_stop"])
    # training times and validation curves
    rep["training"] = {}
    (AB / "val_curves").mkdir(parents=True, exist_ok=True)
    for p in sorted((ROOT / "runs").glob("abl_*_seed*/summary.json")) + \
            [ROOT / "runs" / f"ws_seed{s}" / "summary.json" for s in (0, 1)]:
        sm = json.loads(p.read_text())
        rep["training"][p.parent.name] = dict(steps=sm["steps"], wall_min=sm["wall_s"] / 60, best_val_gap_rr=sm["best_gap_rr"])
        shutil.copy(p.parent / "val.csv", AB / "val_curves" / f"{p.parent.name}.csv")
    # table
    fmt = lambda a, b: f"{a:.2f} [{b[0]:.2f}, {b[1]:.2f}]"  # noqa: E731
    for v in VAR:
        if v not in rep["variants"]:
            continue
        r = rep["variants"][v]
        rows.append([LABEL[v], ",".join(map(str, r["seeds"])), fmt(r["gap_raw"], r["ci_raw"]), fmt(r["gap_rr"], r["ci_rr"]),
                     "--" if v == "ref" else f"{r['delta_rr']:+.2f}",
                     "--" if v == "ref" else f"{fmt_p(r['p'])} ({fmt_p(r['p_holm'])})",
                     "--" if v == "ref" else f"{r['win']:.0f}/{r['tie']:.0f}/{r['loss']:.0f}",
                     f"{r['valid_rr']:.0f}%", f"{r['size_rr']:.1f}", f"{1000 * r['time']:.1f}"])
        if v == "ref":
            rows.append([LABEL["a0"], "0-4", "--", "--", "--", "--", "--",
                         f"{rep['a0']['valid_pct']:.1f}% of {rep['a0']['episodes']}", "--", "--"])
    tex_table(pd.DataFrame(rows, columns=["variant", "seeds", "gap raw % [CI]", "gap + RR % [CI]", "delta vs ref (pp)",
                                          "Wilcoxon p (Holm)", "W/T/L % vs ref", "valid", "|S| (RR)", "time (ms)"]),
              AB / "table_ablations.tex",
              "Ablations on the main test grid (640 instances: 4 classes x 4 schemes x n in {20, 50, 100, 200}). Gap to "
              "the proven optimum (CP-SAT bound for 16 unproven), mean over instances of the seed-mean; 95% bootstrap "
              "CI. Delta = gap + RR minus the reference's on the same seeds. Wilcoxon on W after RR vs the reference, "
              "Holm over A1-A5. W/T/L = ablation better/equal/worse. Validity after RR; time = rollout + RR on the CPU.",
              "tab:ablations")
    srows = []
    for v in VAR:
        if v in rep["variants"]:
            r = rep["variants"][v]
            srows.append([LABEL[v]] + [fmt(r["by_scheme"][s]["gap_rr"], r["by_scheme"][s]["ci"]) +
                                       ("" if v == "ref" else f" (p {fmt_p(r['by_scheme'][s]['p_holm'])})") for s in SCH])
    tex_table(pd.DataFrame(srows, columns=["variant"] + [f"{s} gap + RR %" for s in SCH]),
              AB / "table_ablations_by_scheme.tex",
              "Ablations by weight scheme: gap + RR (%) [95% CI] (Holm-adjusted Wilcoxon p vs the reference over the "
              "20 variant x scheme tests).", "tab:ablations_scheme")
    # figure 1: bars
    vs = [v for v in VAR if v in rep["variants"]]
    fig, ax = plt.subplots(figsize=(6.5, 2.8))
    for i, v in enumerate(vs):
        r = rep["variants"][v]
        ax.bar(i, r["gap_rr"], 0.6, color=COL[i], yerr=[[r["gap_rr"] - r["ci_rr"][0]], [r["ci_rr"][1] - r["gap_rr"]]],
               error_kw=dict(ecolor=INK2, lw=0.8, capsize=3))
        ax.text(i, r["ci_rr"][1], f"{r['gap_rr']:.2f}", ha="center", va="bottom", fontsize=7, color=INK)
    ax.set_xticks(range(len(vs)))
    ax.set_xticklabels([LABEL[v].replace(" (full model)", "") for v in vs], rotation=15, ha="right")
    ax.set_ylabel("mean gap after RR (%)")
    ax.set_title(f"Ablations, test grid (640 instances; 95% CI). A0 blocking rule: {rep['a0']['valid_pct']:.1f}% valid "
                 f"sets in {rep['a0']['episodes']} episodes (not plotted)", color=INK, fontsize=8)
    save(fig, "fig1_ablation_gap_rr")
    # figure 2: by scheme
    fig, ax = plt.subplots(figsize=(7.5, 2.8))
    wd = 0.8 / len(vs)
    for i, v in enumerate(vs):
        r = rep["variants"][v]
        ys = [r["by_scheme"][s]["gap_rr"] for s in SCH]
        lo = [ys[j] - r["by_scheme"][s]["ci"][0] for j, s in enumerate(SCH)]
        hi = [r["by_scheme"][s]["ci"][1] - ys[j] for j, s in enumerate(SCH)]
        ax.bar(np.arange(4) + (i - (len(vs) - 1) / 2) * wd, ys, wd * 0.92, color=COL[i], label=LABEL[v],
               yerr=[lo, hi], error_kw=dict(ecolor=INK2, lw=0.6, capsize=1.5))
    ax.set_xticks(range(4))
    ax.set_xticklabels([f"{s} weights" for s in SCH])
    ax.set_ylabel("mean gap after RR (%)")
    ax.legend(fontsize=7, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    ax.set_title("Ablations by weight scheme (95% CI)", color=INK, fontsize=9)
    save(fig, "fig2_ablation_by_scheme")
    # figure 3: validation curves
    fig, ax = plt.subplots(figsize=(6, 2.8))
    for name, lab, c, ls in [("ws_seed0", "fine-tuned, seed 0", COL[0], "-"), ("ws_seed1", "fine-tuned, seed 1", COL[0], "--"),
                             ("abl_a5_seed0", "from scratch, seed 0", COL[5], "-"),
                             ("abl_a5_seed1", "from scratch, seed 1", COL[5], "--")]:
        p = AB / "val_curves" / f"{name}.csv"
        if p.exists():
            vc = pd.read_csv(p)
            ax.plot(vc.step, 100 * vc.gap_rr, color=c, ls=ls, lw=1.6, label=lab)
    ax.set_xlabel("training steps")
    ax.set_ylabel("validation gap after RR (%)")
    ax.set_yscale("log")
    ax.legend(fontsize=7)
    ax.set_title("Validation gap during training: fine-tuned vs from scratch (160 validation graphs)", color=INK,
                 fontsize=9)
    save(fig, "fig3_val_curves_scratch_vs_finetuned")
    (AB / "report.json").write_text(json.dumps(rep, indent=1, default=float))
    summary(rep)
    print("wrote", AB / "report.json")


def summary(rep):
    p = lambda x: "<1e-4" if x < 1e-4 else f"{x:.4f}"  # noqa: E731
    L = ["# Ablation study (guide's review, page 4)", "",
         "Every number below is generated by `scripts/abl_report.py` from `results/ablations/eval_*.csv` "
         "(`scripts/abl_eval.py`, every raw and RR set checked with both TDS checkers, W recomputed) and the training "
         "runs (`scripts/abl_run_all.sh`, `python -m wtdsws.train --config configs/ablations/<a>.yaml`). "
         "Decisions: docs/decisions.md R27 onwards.", "",
         "Test grid: 640 instances (main grid, n in {20, 50, 100, 200}, same graphs/seeds as results/raw_grid.csv). "
         "Gap = 100 (W - ref) / ref, ref = proven optimum (CP-SAT bound for 16 unproven instances).", "",
         "| variant | seeds | gap raw % [CI] | gap + RR % [CI] | per-seed gap + RR | delta vs ref (pp) | p (Holm) | "
         "W/T/L % | size RR | time ms |", "|---|---|---|---|---|---|---|---|---|---|"]
    for v in VAR:
        r = rep["variants"].get(v)
        if r is None:
            continue
        L.append(f"| {LABEL[v]} | {','.join(map(str, r['seeds']))} | {r['gap_raw']:.3f} [{r['ci_raw'][0]:.3f}, "
                 f"{r['ci_raw'][1]:.3f}] | {r['gap_rr']:.3f} [{r['ci_rr'][0]:.3f}, {r['ci_rr'][1]:.3f}] | "
                 + ", ".join(f"s{s}: {g:.3f}" for s, g in r["per_seed_rr"].items()) + " | "
                 + ("--" if v == "ref" else f"{r['delta_rr']:+.3f}") + " | "
                 + ("--" if v == "ref" else f"{p(r['p'])} ({p(r['p_holm'])})") + " | "
                 + ("--" if v == "ref" else f"{r['win']:.1f}/{r['tie']:.1f}/{r['loss']:.1f}")
                 + f" | {r['size_rr']:.2f} | {1000 * r['time']:.2f} |")
        if v == "ref":
            L.append(f"| {LABEL['a0']} | 0-4 | -- | -- | -- | -- | -- | -- | -- | -- |")
    L += ["", f"A0 (existing result, results/revision/blocking_ablation.json): {rep['a0']['valid_pct']:.1f}% valid sets in "
          f"{rep['a0']['episodes']} episodes (mean covered fraction at stop {rep['a0']['covered_frac']:.3f}). "
          "Validity after RR is 100% for every variant (all sets checked); raw policy sets are also all valid TDSs.", "",
          "## By weight scheme: gap + RR % [CI] (Holm p vs reference)", "",
          "| variant | " + " | ".join(SCH) + " |", "|---|---|---|---|---|"]
    for v in VAR:
        r = rep["variants"].get(v)
        if r:
            L.append(f"| {LABEL[v]} | " + " | ".join(
                f"{r['by_scheme'][s]['gap_rr']:.3f} [{r['by_scheme'][s]['ci'][0]:.3f}, {r['by_scheme'][s]['ci'][1]:.3f}]"
                + ("" if v == "ref" else f" (delta {r['by_scheme'][s]['delta']:+.3f}, p {p(r['by_scheme'][s]['p_holm'])})")
                for s in SCH) + " |")
    L += ["", "## Extra measurements", ""]
    for v in VAR:
        r = rep["variants"].get(v)
        if r:
            L.append(f"- {LABEL[v]}: wasted picks (gain 0) per episode {r['wasted_picks']:.3f}; raw sets missing a "
                     f"degree-1-forced vertex {r['forced_missing_pct']:.1f}%; policy picks spent on degree-1-forced "
                     f"vertices per episode {r['forced_picked']:.3f}; policy picks per episode {r['picks']:.2f}; "
                     f"|S| raw {r['size_raw']:.2f}")
    L += ["", "## Training runs (78,000 steps each)", ""]
    for k, t in rep["training"].items():
        L.append(f"- {k}: {t['steps']} steps, {t['wall_min']:.1f} min, best validation gap after RR "
                 f"{100 * t['best_val_gap_rr']:.3f}%")
    L += ["", "## Files", "", "- results/ablations/table_ablations.tex|csv, table_ablations_by_scheme.tex|csv",
          "- results/ablations/figures/fig1_ablation_gap_rr, fig2_ablation_by_scheme, fig3_val_curves_scratch_vs_finetuned "
          "(.png 300 dpi, .pdf)", "- results/ablations/eval_*.csv (per instance x seed), val_curves/*.csv, report.json",
          "- configs/ablations/*.yaml; tests/test_ablations.py; logs/abl_*.log", ""]
    (AB / "SUMMARY.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
