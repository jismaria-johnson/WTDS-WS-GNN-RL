"""Revision 2: tables, figures and report.json from the anytime traces.

    python scripts/rev2_report.py

Reads results/revision2/step{1,2}_traces.csv (+ device_bench.json, checkpoints/guided_restarts.json); writes
results/revision2/tables/*.tex|csv, results/revision2/figures/*.png|pdf (300 dpi) and results/revision2/report.json.

W of a method at budget b = the best set in its trace found within b seconds (none -> "not found").
Step 1: gap = 100 (W - W*) / W*, proven ILP optima only. Step 2: gap to best-known = 100 (W - BK) / BK, BK = minimum
over every method's final value on the instance; gap to LB = 100 (W - LB) / LB, LB = max(LP bound, CP-SAT bound).
CI = percentile bootstrap over instances (10,000 resamples, rng 0) on instances where the method has a set.
Paired tests (guided vs X): Wilcoxon two-sided on instances where both have a set (zero differences dropped, scipy
default); win/tie/loss over all instances, a method with no set by the budget counts as worse than any set.
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

from stats import boot_ci, fmt_p, tex_table, wilcoxon_p  # noqa: E402

R2 = ROOT / "results" / "revision2"
FIG, TAB = R2 / "figures", R2 / "tables"
COL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#52514e"]
MK = ["o", "s", "^", "D", "v", "P", "X", "*"]
LS = ["-", "-", "--", "--", ":", "-.", "-", ":"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
                     "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb"})
# fixed slot per method (colour follows the entity across every figure)
LABEL = {"guided": "Proposed (guided restarts)", "random_prior": "random-prior restarts", "plain": "plain restarts",
         "ours_bok_ls": "Proposed (best-of-k) + LS", "ours_greedy_ls": "Proposed + RR + LS",
         "greedy_ls": "Greedy + RR + LS", "lp_ls": "LP rounding+RR+LS", "cpsat": "CP-SAT"}
SLOT = {m: i for i, m in enumerate(LABEL)}
KEY = ["cls", "scheme", "n", "gidx"]
SCH1 = ["degree", "inverse", "random", "mixed"]


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def w_at(trace, b):
    best = np.nan
    for t, W, _ in trace:
        if t <= b + 1e-9:
            best = W
    return best


def load(step):
    d = pd.read_csv(R2 / f"step{step}_traces.csv")
    d["tr"] = d.trace.apply(json.loads)
    d["model_seed"] = d.model_seed.fillna("").astype(str).str.replace(".0", "", regex=False)
    return d


def table_at(d, budgets, ref):
    """Wide frame: index KEY, columns (method, budget) -> W; ref: Series of reference values per instance."""
    out = {}
    for m, x in d.groupby("method"):
        x = x.set_index(KEY)
        for b in budgets:
            out[(m, b)] = x.tr.apply(lambda tr: w_at(tr, b))
    w = pd.DataFrame(out)
    return w, w.apply(lambda c: 100 * (c - ref.reindex(w.index)) / ref.reindex(w.index))


def summarize(gap, m, b, mask=None):
    x = gap[(m, b)] if mask is None else gap[(m, b)][mask]
    v = x.dropna().values
    ci = boot_ci(v)
    return dict(gap=float(np.mean(v)) if len(v) else np.nan, lo=float(ci[0]), hi=float(ci[1]), found=int(len(v)),
                total=int(len(x)))


def paired(w, a, bm, b, mask=None):
    x, y = w[(a, b)], w[(bm, b)]
    if mask is not None:
        x, y = x[mask], y[mask]
    both = x.notna() & y.notna()
    xa, ya = x.fillna(np.inf).values, y.fillna(np.inf).values
    win = int(((xa < ya - 1e-9)).sum())
    loss = int(((xa > ya + 1e-9)).sum())
    tie = len(xa) - win - loss
    p = wilcoxon_p(x[both].values, y[both].values) if both.sum() else np.nan
    return dict(p=p, win=win, tie=tie, loss=loss, n=int(len(xa)), n_both=int(both.sum()),
                mean_diff=float((x[both] - y[both]).mean()) if both.sum() else np.nan)


def fmt_ci(s):
    return f"{s['gap']:.2f} [{s['lo']:.2f}, {s['hi']:.2f}]" if s["found"] else "--"


def curve(gap, d, m, ts, mask=None):
    """Mean gap at each time t over the instances of mask; nan until every instance has a set."""
    x = d[d.method == m].set_index(KEY)
    if mask is not None:
        x = x[mask.reindex(x.index).fillna(False).values]
    ref = gap
    ys = []
    for t in ts:
        v = x.tr.apply(lambda tr: w_at(tr, t))
        g = 100 * (v - ref.reindex(v.index)) / ref.reindex(v.index)
        ys.append(g.mean() if g.notna().all() else np.nan)
    return np.array(ys)


# ------------------------------------------------------------------ step 1
def step1(rep):
    d = load(1)
    d0 = d[d.model_seed.isin(["", "0"])]
    inst = d0.drop_duplicates(KEY).set_index(KEY)
    proven = inst.proven.astype(str) == "True"
    opt = inst.opt[proven]
    budgets = [0.16, 1.0, 5.0]
    w, gap = table_at(d0[d0.set_index(KEY).index.isin(opt.index)], budgets, opt)
    methods = [m for m in LABEL if m in set(d0.method)]
    res = {"instances": int(len(opt)), "unproven_excluded": int((~proven).sum())}
    rows, ptab = [], []
    for m in methods:
        for b in budgets:
            s = summarize(gap, m, b)
            r = [LABEL[m], b, fmt_ci(s), f"{s['found']}/{s['total']}"]
            res[f"{m}@{b}"] = {"pooled": s}
            for sc in SCH1:
                mk = gap.index.get_level_values("scheme") == sc
                ss = summarize(gap, m, b, mk)
                res[f"{m}@{b}"][sc] = ss
                r.append(fmt_ci(ss))
            rows.append(r)
    tex_table(pd.DataFrame(rows, columns=["method", "budget (s)", "pooled gap % [95% CI]", "found"] +
                           [f"{s} gap %" for s in SCH1]), TAB / "step1_gaps.tex",
              "Anytime equal-time comparison, n = 100 and 200 (160 main-grid instances, 5 per class x scheme x n; "
              "proven optima). Each method ran once up to 5 s; values read from its trace. GNN on the CPU.",
              "tab:rev2_step1_gaps")
    comps = ["random_prior", "plain", "lp_ls", "cpsat", "ours_bok_ls"]
    for b in budgets:
        for c in [x for x in comps if x in methods]:
            for sc in ["pooled"] + SCH1:
                mk = None if sc == "pooled" else (w.index.get_level_values("scheme") == sc)
                pr = paired(w, "guided", c, b, mk)
                res[f"guided_vs_{c}@{b}"] = res.get(f"guided_vs_{c}@{b}", {})
                res[f"guided_vs_{c}@{b}"][sc] = pr
                ptab.append([b, LABEL[c], sc, pr["n"], fmt_p(pr["p"]) if pr["n_both"] else "--",
                             f"{pr['win']}/{pr['tie']}/{pr['loss']}", f"{pr['mean_diff']:+.2f}"])
    tex_table(pd.DataFrame(ptab, columns=["budget (s)", "guided vs", "scheme", "inst", "Wilcoxon p",
                                          "guided W/T/L", "mean dW"]), TAB / "step1_tests.tex",
              "GNN-guided restarts vs each method at equal budget (n = 100, 200): paired Wilcoxon on W(S) and "
              "win/tie/loss counts (instances). dW = mean W(guided) - W(other); < 0 = guided cheaper.",
              "tab:rev2_step1_tests")
    # seed 1 (half of the instances)
    s1 = d[(d.method == "guided") & (d.model_seed == "1")]
    if len(s1):
        x1 = s1.set_index(KEY)
        x1 = x1[x1.index.isin(opt.index)]
        r1 = {}
        for b in budgets:
            g1 = 100 * (x1.tr.apply(lambda tr: w_at(tr, b)) - opt.reindex(x1.index)) / opt.reindex(x1.index)
            g0 = gap[("guided", b)].reindex(x1.index)
            pr = paired(pd.DataFrame({("s1", b): g1, ("s0", b): g0}), "s1", "s0", b)
            r1[str(b)] = dict(gap_seed1=float(g1.mean()), gap_seed0_same=float(g0.mean()), n=int(len(g1)),
                              p=pr["p"], wtl=[pr["win"], pr["tie"], pr["loss"]])
        res["guided_seed1_half"] = r1
    rep["step1"] = res
    # figure 1: quality vs time, pooled + degree
    ts = np.logspace(np.log10(0.005), np.log10(5.0), 60)
    d0p = d0[d0.set_index(KEY).index.isin(opt.index)]
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.2))
    for ax, sub in zip(axs, ["pooled", "degree"]):
        mk = None if sub == "pooled" else pd.Series(opt.index.get_level_values("scheme") == "degree", index=opt.index)
        for m in methods:
            ys = curve(opt, d0p, m, ts, mk)
            ax.plot(ts, ys, color=COL[SLOT[m]], ls=LS[SLOT[m]], lw=1.8, label=LABEL[m])
        for b in budgets:
            ax.axvline(b, color=INK2, lw=0.6, ls=":")
        ax.set_xscale("log")
        ax.set_xlabel("wall-clock time (s), log scale")
        ax.set_title("all schemes" if sub == "pooled" else "degree weights", color=INK, fontsize=9)
        ax.set_ylim(bottom=0)
    axs[0].set_ylabel("mean gap to proven optimum (%)")
    axs[0].legend(loc="upper center", bbox_to_anchor=(1.1, -0.2), ncol=3, fontsize=7.5)
    fig.suptitle("n = 100 and 200: quality vs time (a curve starts once every instance has a solution)",
                 color=INK, fontsize=10, y=1.02)
    save(fig, "fig1_quality_vs_time_small")
    # figure 3: GNN-knowledge test at 1 s
    fig, ax = plt.subplots(figsize=(6.5, 2.8))
    groups = ["pooled"] + SCH1
    trio = [m for m in ("guided", "random_prior", "plain") if m in methods]
    wd = 0.26
    for j, m in enumerate(trio):
        ys = [res[f"{m}@1.0"][g]["gap"] for g in groups]
        lo = [ys[i] - res[f"{m}@1.0"][g]["lo"] for i, g in enumerate(groups)]
        hi = [res[f"{m}@1.0"][g]["hi"] - ys[i] for i, g in enumerate(groups)]
        ax.bar(np.arange(len(groups)) + (j - 1) * wd, ys, wd * 0.92, color=COL[SLOT[m]], label=LABEL[m],
               yerr=[lo, hi], error_kw=dict(ecolor=INK2, lw=0.8, capsize=2))
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(["all schemes"] + [f"{s} weights" for s in SCH1])
    ax.set_ylabel("mean gap at 1 s (%)")
    ax.set_title("Does the GNN prior help? Same restart code, only the prior differs (n = 100, 200; 95% CI)",
                 color=INK, fontsize=9)
    ax.legend(fontsize=7.5)
    save(fig, "fig3_gnn_knowledge_test_1s")


# ------------------------------------------------------------------ step 2
def step2(rep):
    d = load(2)
    meta = pd.DataFrame(json.loads((ROOT / "data" / "rev2_large.json").read_text())).set_index(KEY)
    budgets = [1.0, 3.0, 10.0]
    allw = d.groupby(KEY).tr.apply(lambda s: min(W for tr in s for _, W, _ in tr))
    cp = d[d.method == "cpsat"].set_index(KEY)
    cpb = cp.extra.apply(lambda e: json.loads(e)["bound"])
    lb = pd.concat([meta.lp_bound.reindex(allw.index), cpb.reindex(allw.index)], axis=1).max(axis=1)
    assert (allw >= lb - 1e-6).all()
    methods = [m for m in LABEL if m in set(d.method)]
    w, gbk = table_at(d, budgets, allw)
    _, glb = table_at(d, budgets, lb)
    ns = sorted(d.n.unique())
    res = {"instances": int(len(allw)), "bk_minus_lb_pct": float((100 * (allw - lb) / lb).mean())}
    rows, ptab, best = [], [], []
    for n in ns:
        for sc in ["pooled", "degree", "random"]:
            mk = (gbk.index.get_level_values("n") == n)
            if sc != "pooled":
                mk = mk & (gbk.index.get_level_values("scheme") == sc)
            for b in budgets:
                cell = {}
                for m in methods:
                    s, s2 = summarize(gbk, m, b, mk), summarize(glb, m, b, mk)
                    cell[m] = dict(bk=s, lb=s2)
                    rows.append([n, sc, b, LABEL[m], fmt_ci(s), fmt_ci(s2), f"{s['found']}/{s['total']}"])
                res[f"n{n}_{sc}@{b}"] = cell
                full = {m: c["bk"]["gap"] for m, c in cell.items() if c["bk"]["found"] == c["bk"]["total"]}
                if full:
                    bm = min(full, key=full.get)
                    best.append([n, sc, b, LABEL[bm], f"{full[bm]:.2f}"])
                    res[f"n{n}_{sc}@{b}"]["_best"] = bm
                for c in ["plain", "lp_ls", "cpsat", "random_prior", "ours_bok_ls"]:
                    if c in methods and "guided" in methods:
                        pr = paired(w, "guided", c, b, mk)
                        res[f"n{n}_{sc}@{b}"][f"guided_vs_{c}"] = pr
                        ptab.append([n, sc, b, LABEL[c], pr["n"], fmt_p(pr["p"]) if pr["n_both"] > 0 else "--",
                                     f"{pr['win']}/{pr['tie']}/{pr['loss']}"])
    tex_table(pd.DataFrame(rows, columns=["n", "scheme", "budget (s)", "method", "gap to best-known % [CI]",
                                          "gap to LB % [CI]", "found"]), TAB / "step2_gaps.tex",
              "Large-n anytime comparison (fresh test graphs, 4 classes x 2 schemes x 3 graphs per n). Best-known = "
              "minimum over all methods' final values; LB = max(LP relaxation, CP-SAT bound).", "tab:rev2_step2_gaps")
    tex_table(pd.DataFrame(ptab, columns=["n", "scheme", "budget (s)", "guided vs", "inst", "Wilcoxon p",
                                          "guided W/T/L"]), TAB / "step2_tests.tex",
              "Large n: GNN-guided restarts vs each method (paired Wilcoxon on W, win/tie/loss; no set by the "
              "budget = loss).", "tab:rev2_step2_tests")
    tex_table(pd.DataFrame(best, columns=["n", "scheme", "budget (s)", "best method (all found)", "gap to BK %"]),
              TAB / "step2_best.tex", "Large n: method with the lowest mean gap to best-known among methods that "
              "had a set on every instance by the budget.", "tab:rev2_step2_best")
    # first-solution times of the GNN methods
    ft = d[d.method.isin(["ours_greedy_ls", "ours_bok_ls", "guided"])].groupby(["method", "n"]).first_t
    res["first_solution_time"] = {f"{m}@{n}": dict(mean=float(v.mean()), max=float(v.max()))
                                  for (m, n), v in ft}
    ex = d[d.method == "ours_greedy_ls"].extra.apply(lambda e: json.loads(e)["rollout_rr_time"])
    res["greedy_rollout_rr_time"] = {str(n): dict(mean=float(v.mean()), max=float(v.max()))
                                     for n, v in ex.groupby(d[d.method == "ours_greedy_ls"].n)}
    res["devices"] = d[d.method.isin(["guided", "ours_bok_ls", "ours_greedy_ls"])].groupby("n").device.first().to_dict()
    rep["step2"] = res
    # figure 2: quality vs time per n (gap to best-known)
    ts = np.logspace(np.log10(0.01), np.log10(10.0), 60)
    fig, axs = plt.subplots(1, len(ns), figsize=(3.2 * len(ns), 3.0))
    for ax, n in zip(np.atleast_1d(axs), ns):
        mk = pd.Series(allw.index.get_level_values("n") == n, index=allw.index)
        for m in methods:
            ax.plot(ts, curve(allw, d, m, ts, mk), color=COL[SLOT[m]], ls=LS[SLOT[m]], lw=1.8, label=LABEL[m])
        for b in budgets:
            ax.axvline(b, color=INK2, lw=0.6, ls=":")
        ax.set_xscale("log")
        ax.set_title(f"n = {n}", color=INK, fontsize=9)
        ax.set_xlabel("wall-clock time (s), log")
        ax.set_ylim(bottom=0)
    np.atleast_1d(axs)[0].set_ylabel("mean gap to best-known (%)")
    np.atleast_1d(axs)[0].legend(loc="upper center", bbox_to_anchor=(2.2, -0.2), ncol=4, fontsize=7.5)
    fig.suptitle("Large n: quality vs time, degree + random weights (a curve starts once every instance has a "
                 "solution)", color=INK, fontsize=10, y=1.03)
    save(fig, "fig2_quality_vs_time_large")
    # figure 4: gap vs n at 10 s
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.0))
    for ax, (gg, lab) in zip(axs, [(gbk, "gap to best-known (%)"), (glb, "gap to best lower bound (%)")]):
        for m in methods:
            ys, lo, hi = [], [], []
            for n in ns:
                s = summarize(gg, m, 10.0, gg.index.get_level_values("n") == n)
                ok = s["found"] == s["total"]
                ys.append(s["gap"] if ok else np.nan)
                lo.append(s["lo"] if ok else np.nan)
                hi.append(s["hi"] if ok else np.nan)
            ax.plot(ns, ys, color=COL[SLOT[m]], marker=MK[SLOT[m]], ls=LS[SLOT[m]], lw=1.8, ms=4, label=LABEL[m])
            ax.fill_between(ns, lo, hi, color=COL[SLOT[m]], alpha=0.08, lw=0)
        ax.set_xscale("log")
        ax.set_xticks(ns)
        ax.set_xticklabels([str(n) for n in ns])
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_xlabel("n")
        ax.set_ylabel(lab)
    axs[0].legend(loc="upper center", bbox_to_anchor=(1.1, -0.2), ncol=4, fontsize=7.5)
    fig.suptitle("Gap vs n at a 10 s budget (95% CI; a point is shown only if the method had a set on every "
                 "instance)", color=INK, fontsize=10, y=1.03)
    save(fig, "fig4_gap_vs_n_10s")


def main():
    import matplotlib.ticker  # noqa: F401
    global R2, FIG, TAB
    if len(sys.argv) > 2 and sys.argv[1] == "--dir":         # pipeline check on another folder (e.g. smoke)
        R2 = Path(sys.argv[2])
        FIG, TAB = R2 / "figures", R2 / "tables"
    TAB.mkdir(parents=True, exist_ok=True)
    rep = {"tuned": json.loads((ROOT / "checkpoints" / "guided_restarts.json").read_text())}
    p = R2 / "device_bench.json"
    if p.exists():
        rep["device_bench"] = json.loads(p.read_text())
    if (R2 / "step1_traces.csv").exists():
        step1(rep)
    if (R2 / "step2_traces.csv").exists():
        step2(rep)
    (R2 / "report.json").write_text(json.dumps(rep, indent=1, default=float))
    print("wrote", R2 / "report.json")


if __name__ == "__main__":
    main()
