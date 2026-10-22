"""Revision step 7: equal-time and alpha-sweep tables + figures, training / inference times, numbers for SUMMARY.md.

    python scripts/rev_report.py

Reads results/revision/{equal_time_*.csv, alpha_sweep_*.csv, inference_*.json} and runs/ws_seed*/; writes
results/revision/tables/*.tex|csv, results/revision/figures/*.png|pdf (300 dpi) and results/revision/report.json.
Proposed = per-instance mean over the model seeds present; CIs = bootstrap over instances (10,000) unless stated.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402,F401
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats as st  # noqa: E402

from stats import boot_ci, fmt_p, seed_ci, tex_table, wilcoxon_p, wtl  # noqa: E402

REV = ROOT / "results" / "revision"
FIG = REV / "figures"
TAB = REV / "tables"
# validated categorical order (dataviz reference palette, light); markers/linestyles = secondary encoding
COL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]
MK = ["o", "s", "^", "D", "v", "P", "X"]
LS = ["-", "-", "--", "--", ":", "-.", "-"]
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


# ------------------------------------------------------------------ equal time
ET_LABEL = {"ours_bok": "Proposed (best-of-k)", "ours_bok_ls": "Proposed (best-of-k) + LS", "grr": "greedy restarts+RR",
            "grr_ls": "greedy restarts+RR+LS", "greedy_rr_ls": "Greedy + RR + LS", "lp_rr_ls": "LP rounding+RR+LS",
            "cpsat": "CP-SAT"}


def equal_time(rep):
    parts = [pd.read_csv(p) for p in REV.glob("equal_time_*.csv")]
    if not parts:
        return
    d = pd.concat(parts, ignore_index=True)
    d = d[d.proven.astype(str) == "True"]
    d["gap"] = 100 * (d.weight - d.opt) / d.opt
    key = ["cls", "scheme", "n", "gidx"]
    rows, res = [], {}
    budgets = sorted(d.budget.unique())
    for m in [x for x in ET_LABEL if x in set(d.method)]:
        for b in budgets:
            x = d[(d.method == m) & (d.budget == b)]
            found = x.found.astype(str) == "True"
            inst = x[found].groupby(key).gap.mean()
            per_seed = x[found].groupby("model_seed").gap.mean() if m.startswith("ours") else []
            r = dict(gap=float(inst.mean()), ci=boot_ci(inst.values), ci_seed=seed_ci(per_seed),
                     found=f"{int(found.sum())}/{len(x)}", runtime=float(x.runtime.mean()),
                     cand=float(x.candidates.mean()), n_inst=int(inst.size))
            for s in ["degree", "inverse", "random", "mixed"]:
                xs = inst[inst.index.get_level_values("scheme") == s]
                r[s] = dict(gap=float(xs.mean()), ci=boot_ci(xs.values))
            res[f"{m}@{b}"] = r
            rows.append([ET_LABEL[m], b, r["n_inst"], f"{r['gap']:.2f}", f"[{r['ci'][0]:.2f}, {r['ci'][1]:.2f}]",
                         f"{r['runtime']:.3f}", f"{r['cand']:.0f}", r["found"]])
    # paired tests: ours_bok_ls (seed mean) vs every other method, per budget
    ptab = []
    for b in budgets:
        o = d[(d.method == "ours_bok_ls") & (d.budget == b)].groupby(key).weight.mean()
        for m in [x for x in ET_LABEL if x in set(d.method) and x != "ours_bok_ls"]:
            x = d[(d.method == m) & (d.budget == b) & (d.found.astype(str) == "True")].groupby(key).weight.mean()
            j = o.index.intersection(x.index)
            if len(j) == 0:
                continue
            ow = d[(d.method == "ours_bok_ls") & (d.budget == b)].pivot_table(index=key, columns="model_seed",
                                                                                values="weight").reindex(j)
            w, t, l = wtl(ow.values, x[j].values)
            p = wilcoxon_p(o[j].values, x[j].values)
            res[f"ours_bok_ls_vs_{m}@{b}"] = dict(p=p, win=w, tie=t, loss=l, n=len(j))
            ptab.append([b, ET_LABEL[m], len(j), fmt_p(p), f"{w:.0f}/{t:.0f}/{l:.0f}"])
    tex_table(pd.DataFrame(rows, columns=["method", "budget (s)", "inst", "gap %", "95% CI", "mean time (s)",
                                          "candidates", "solution found"]), TAB / "equal_time.tex",
              "Equal wall-clock budget per instance (main grid, n = 100 and 200, 2 graphs per class x scheme, proven "
              "optima). Proposed = mean over 5 model seeds, GNN on the GPU. Candidates = rollouts or restarts.",
              "tab:equal_time")
    tex_table(pd.DataFrame(ptab, columns=["budget (s)", "vs method", "inst", "Wilcoxon p", "Proposed W/T/L %"]),
              TAB / "equal_time_tests.tex", "Proposed (best-of-k) + LS vs each method at equal budget: paired "
              "Wilcoxon (seed-mean W) and win/tie/loss over (instance, seed) pairs.", "tab:equal_time_tests")
    rep["equal_time"] = res
    # figure: gap vs budget, pooled + per scheme
    fig, axs = plt.subplots(1, 5, figsize=(13, 2.8), sharey=False)
    for ax, sub in zip(axs, ["pooled", "degree", "inverse", "random", "mixed"]):
        for i, m in enumerate(ET_LABEL):                  # fixed slot per method: colour follows the entity
            if m not in set(d.method):
                continue
            ys, lo, hi = [], [], []
            for b in budgets:
                r = res[f"{m}@{b}"]
                rr = r if sub == "pooled" else r[sub]
                ys.append(rr["gap"]), lo.append(rr["ci"][0]), hi.append(rr["ci"][1])
            ax.plot(budgets, ys, color=COL[i], marker=MK[i], ls=LS[i], lw=2, ms=5, label=ET_LABEL[m])
            ax.fill_between(budgets, lo, hi, color=COL[i], alpha=0.07, lw=0)
        ax.set_xscale("log")
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_xticks(budgets)
        ax.set_xticklabels([f"{b:g}" for b in budgets])
        ax.set_title(sub.capitalize(), color=INK, fontsize=9)
        ax.set_xlabel("time budget per instance (s)")
    axs[0].set_ylabel("mean gap to optimum (%)")
    axs[0].legend(loc="upper center", bbox_to_anchor=(2.9, -0.28), ncol=7, fontsize=7.5)
    fig.suptitle("Solution quality vs time budget (95% CI bands; Proposed: mean of 5 seeds, GPU)", color=INK, fontsize=10, y=1.06)
    save(fig, "equal_time_quality_vs_time")


# ------------------------------------------------------------------ alpha sweep
AL_M = [("ours_rr", "Proposed + RR"), ("greedy_rr", "Greedy + RR"), ("greedy_tiebreak_rr", "greedy tie-break+RR"),
        ("greedy_rr_ls", "Greedy + RR + LS"), ("ours_rr_ls", "Proposed + RR + LS")]


def alpha(rep):
    ps = [pd.read_csv(p) for p in REV.glob("alpha_sweep_*.csv")]
    if len(ps) < 2:
        return
    d = pd.concat(ps, ignore_index=True)
    assert d.valid.astype(str).eq("True").all()
    key = ["cls", "n", "gidx", "alpha"]
    il = d[d.method == "ilp"].set_index(key)
    proven = il.optimal.astype(str) == "True"
    opt = il.weight[proven]
    alphas = sorted(d.alpha.unique())
    W = {}
    for m, _ in AL_M:
        x = d[d.method == m]
        if m.startswith("ours"):
            W[m] = x.pivot_table(index=key, columns="model_seed", values="weight").reindex(opt.index)
        else:
            W[m] = x.set_index(key).weight.reindex(opt.index)
    seeds = sorted(W["ours_rr"].columns)
    gap = {m: (100 * (W[m].mean(axis=1) if isinstance(W[m], pd.DataFrame) else W[m]) - 100 * opt) / opt
           for m, _ in AL_M}
    al_lvl = opt.index.get_level_values("alpha")
    res, rows = {"seeds": list(map(int, seeds)), "unproven": int((~proven).sum()), "per_alpha": {}}, []
    for a in alphas:
        mk = al_lvl == a
        r = {}
        for m, _ in AL_M:
            g = gap[m][mk]
            r[m] = dict(gap=float(g.mean()), ci=boot_ci(g.values))
        for o, b in (("ours_rr", "greedy_rr"), ("ours_rr", "greedy_tiebreak_rr"), ("ours_rr_ls", "greedy_rr_ls")):
            adv = gap[b][mk] - gap[o][mk]
            w, t, l = wtl(W[o][mk].values, W[b][mk].values)
            r[f"{o}_vs_{b}"] = dict(adv=float(adv.mean()), adv_ci=boot_ci(adv.values),
                                    p=wilcoxon_p(W[o][mk].mean(axis=1).values, W[b][mk].values), win=w, tie=t, loss=l)
        res["per_alpha"][str(a)] = r
        x, y, z = r["ours_rr_vs_greedy_rr"], r["ours_rr_vs_greedy_tiebreak_rr"], r["ours_rr_ls_vs_greedy_rr_ls"]
        rows.append([f"{a:+.2f}", int(mk.sum()), f"{r['ours_rr']['gap']:.2f}", f"{r['greedy_rr']['gap']:.2f}",
                     f"{r['greedy_tiebreak_rr']['gap']:.2f}", f"{x['adv']:+.2f}", fmt_p(x["p"]),
                     f"{x['win']:.0f}/{x['tie']:.0f}/{x['loss']:.0f}", f"{y['adv']:+.2f}", fmt_p(y["p"]),
                     f"{r['ours_rr_ls']['gap']:.2f}", f"{r['greedy_rr_ls']['gap']:.2f}", fmt_p(z["p"])])
    tex_table(pd.DataFrame(rows, columns=["alpha", "inst", "Proposed + RR", "Greedy + RR", "tie-break+RR",
                                          "adv. vs greedy (pp)", "p", "W/T/L %", "adv. vs tie-break", "p (tb)",
                                          "Proposed + RR + LS", "Greedy + RR + LS", "p (LS)"]), TAB / "alpha_sweep.tex",
              f"Alpha-sweep: mean gap (\\%) to the proven ILP optimum, w(v) tied to degree with exponent alpha (ER, "
              f"BA, chordal, bipartite; n = 50, 100; 80 fresh graphs per alpha). Proposed = mean of {len(seeds)} seeds; "
              "advantage = greedy gap - Proposed gap; p = paired Wilcoxon.", "tab:alpha_sweep")
    # trend: Spearman correlation of per-instance advantage with alpha, alpha >= 0 and alpha <= 0
    adv_i = gap["greedy_rr"] - gap["ours_rr"]
    for lab, mk in (("pos", al_lvl >= 0), ("neg", al_lvl <= 0), ("all", al_lvl == al_lvl)):
        rho, p = st.spearmanr(al_lvl[mk], adv_i[mk])
        res[f"spearman_adv_alpha_{lab}"] = dict(rho=float(rho), p=float(p))
    rep["alpha"] = res
    # plot 1
    fig, ax = plt.subplots(figsize=(5.2, 3.3))
    for i, (m, lab) in enumerate(AL_M):
        ys = [res["per_alpha"][str(a)][m]["gap"] for a in alphas]
        lo = [res["per_alpha"][str(a)][m]["ci"][0] for a in alphas]
        hi = [res["per_alpha"][str(a)][m]["ci"][1] for a in alphas]
        ax.plot(alphas, ys, color=COL[i], marker=MK[i], ls=LS[i], lw=2, ms=5, label=lab)
        ax.fill_between(alphas, lo, hi, color=COL[i], alpha=0.12, lw=0)
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_xlabel("alpha  (negative: hubs cheap,  0: equal weights,  positive: hubs expensive)")
    ax.set_ylabel("mean gap to optimum (%)")
    ax.set_xticks(alphas)
    ax.legend(fontsize=7, ncol=2)
    ax.set_title(f"Gap vs alpha (95% CI bands; Proposed: mean of {len(seeds)} seeds)", color=INK, fontsize=9)
    save(fig, "alpha_gap")
    # plot 2: advantage
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    for i, (k, lab) in enumerate([("ours_rr_vs_greedy_rr", "Greedy + RR - Proposed + RR"),
                                  ("ours_rr_vs_greedy_tiebreak_rr", "greedy tie-break+RR - Proposed + RR"),
                                  ("ours_rr_ls_vs_greedy_rr_ls", "Greedy + RR + LS - Proposed + RR + LS")]):
        ys = [res["per_alpha"][str(a)][k]["adv"] for a in alphas]
        lo = [res["per_alpha"][str(a)][k]["adv_ci"][0] for a in alphas]
        hi = [res["per_alpha"][str(a)][k]["adv_ci"][1] for a in alphas]
        ax.plot(alphas, ys, color=COL[[0, 2, 4][i]], marker=MK[i], ls=LS[i], lw=2, ms=5, label=lab)
        ax.fill_between(alphas, lo, hi, color=COL[[0, 2, 4][i]], alpha=0.12, lw=0)
    ax.axhline(0, color=INK, lw=0.9)
    ax.set_xticks(alphas)
    ax.set_xlabel("alpha")
    ax.set_ylabel("advantage of Proposed (pp of gap)")
    ax.legend(fontsize=7)
    ax.set_title("Advantage of Proposed over greedy vs alpha (> 0: Proposed better)", color=INK, fontsize=9)
    save(fig, "alpha_advantage")


# ------------------------------------------------------------------ training and inference time
def times(rep):
    info = json.loads((ROOT / "checkpoints" / "model_info.json").read_text())
    tr = {}
    for s in info["seeds"]:
        tr[int(s["seed"])] = dict(wall_s=s["wall_s"], steps=s["finetune_steps"], best_step=s["best_step"],
                                  best_val_gap_rr=s["finetuned_best_val_gap_rr"], source="model_info.json")
    for s in (2, 3, 4):
        p = ROOT / "runs" / f"ws_seed{s}" / "summary.json"
        if p.exists():
            sm = json.loads(p.read_text())
            best = min(sm["history"], key=lambda h: h["gap_rr"])
            tr[s] = dict(wall_s=sm["wall_s"], steps=sm["steps"], best_step=best["step"], best_val_gap_rr=sm["best_gap_rr"],
                         zero_shot_val_gap_rr=sm["history"][0]["gap_rr"], source=f"runs/ws_seed{s}/summary.json")
    rep["training"] = tr
    inf = {}
    for dev in ("cuda", "cpu"):
        p = REV / f"inference_{dev}.json"
        if p.exists():
            j = json.loads(p.read_text())
            inf[dev] = {n: v["mean_s"] for n, v in j["per_n"].items()}
            inf[dev + "_W"] = {n: [r["W"] for r in v["rows"]] for n, v in j["per_n"].items()}
    if "cuda" in inf and "cpu" in inf:
        inf["same_W_cpu_gpu"] = {n: sum(a == b for a, b in zip(inf["cuda_W"][n], inf["cpu_W"][n])) for n in inf["cuda"]}
    rep["inference"] = {k: v for k, v in inf.items() if not k.endswith("_W")}
    rows = [[s, f"{v['wall_s'] / 60:.1f}", v["steps"], v["best_step"], f"{100 * v['best_val_gap_rr']:.2f}"]
            for s, v in sorted(tr.items())]
    tex_table(pd.DataFrame(rows, columns=["seed", "train min", "steps", "best step", "val gap % (+RR)"]),
              TAB / "training_time.tex", "Fine-tuning time per model seed (RTX 3050 GPU).", "tab:training_time")
    if "cuda" in inf and "cpu" in inf:
        rows = [[n, f"{inf['cuda'][n]:.4f}", f"{inf['cpu'][n]:.4f}", f"{inf['cpu'][n] / inf['cuda'][n]:.2f}",
                 f"{inf['same_W_cpu_gpu'][n]}/{len(inf['cpu_W'][n])}"] for n in inf["cuda"]]
        tex_table(pd.DataFrame(rows, columns=["n", "GPU (s)", "CPU only (s)", "CPU/GPU", "same W"]),
                  TAB / "inference_time.tex", "GNN greedy rollout + RR time per instance, GPU vs CUDA disabled "
                  "(12 grid graphs per n, random weights, seed 0).", "tab:inference_time")


def main():
    TAB.mkdir(parents=True, exist_ok=True)
    rep = {}
    for f in (equal_time, alpha, times):
        try:
            f(rep)
        except Exception as e:                       # report what is missing, keep building the rest
            print(f"{f.__name__}: {type(e).__name__}: {e}")
            rep[f"{f.__name__}_error"] = str(e)
    (REV / "report.json").write_text(json.dumps(rep, indent=1, default=float), encoding="utf-8")
    print(json.dumps({k: v for k, v in rep.items() if k in ("training", "inference")}, indent=1, default=float))


if __name__ == "__main__":
    main()
