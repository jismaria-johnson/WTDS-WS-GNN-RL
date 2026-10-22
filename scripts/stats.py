"""Revision step 2: statistics for every main result -> results/revision/tables/*.tex|csv and stats_main.json.

    python scripts/stats.py

Gap per instance = 100 (W(S) - W*) / W*, proven ILP optima only (main grid n = 20-200). For "ours" the per-instance
value is the mean over the model seeds that are available (seeds 0-4 when step 0 finished).
* 95% CI over instances: percentile bootstrap, 10,000 resamples of instances (seed-averaged values), rng seed 0.
* 95% CI across seeds: t-interval on the per-seed mean gaps (df = #seeds - 1).
* Paired Wilcoxon signed-rank (two-sided, zero differences dropped = scipy default), ours (seed-mean W) vs baseline,
  per class x scheme, per scheme and pooled; Holm-adjusted p over the 16 cells is also given.
* Win / tie / loss %: over (instance, model seed) pairs, ours vs the baseline on W(S) (tie = equal W).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as st

ROOT = Path(__file__).resolve().parents[1]
REV = ROOT / "results" / "revision"
TAB = REV / "tables"
B = 10_000
CLS = ["er", "ba", "chordal", "bipartite"]
CLS_LABEL = {"er": "ER", "ba": "BA", "chordal": "Chordal", "bipartite": "Bipartite"}
SCH = ["degree", "inverse", "random", "mixed"]
KEY = ["cls", "scheme", "n", "gidx"]


# ------------------------------------------------------------------ generic helpers (also used by other rev scripts)
def boot_ci(x, b=B, seed=0):
    x = np.asarray(x, float)
    if len(x) == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    m = x[rng.integers(0, len(x), size=(b, len(x)))].mean(axis=1)
    return tuple(np.percentile(m, [2.5, 97.5]))


def seed_ci(per_seed):
    v = np.asarray(per_seed, float)
    if len(v) < 2:
        return (np.nan, np.nan)
    h = st.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
    return (v.mean() - h, v.mean() + h)


def wilcoxon_p(a, b):
    d = np.asarray(a, float) - np.asarray(b, float)
    if np.all(np.abs(d) < 1e-12):
        return 1.0
    return float(st.wilcoxon(a, b).pvalue)


def holm(ps):
    ps = np.asarray(ps, float)
    order = np.argsort(ps)
    adj = np.empty_like(ps)
    run = 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - r) * ps[i]))
        adj[i] = run
    return adj


def wtl(ours_w, base_w):
    """ours_w: (instances, seeds) W; base_w: (instances,) W. Percent of (instance, seed) pairs."""
    o = np.asarray(ours_w, float)
    b = np.asarray(base_w, float)[:, None]
    ok = ~np.isnan(o)
    win = ((o < b - 1e-9) & ok).sum()
    loss = ((o > b + 1e-9) & ok).sum()
    tot = ok.sum()
    return 100 * win / tot, 100 * (tot - win - loss) / tot, 100 * loss / tot


def fmt_p(p):
    return "<1e-4" if p < 1e-4 else f"{p:.4f}"


def tex_table(df, path, caption, label):
    cols = list(df.columns)
    lines = [r"\begin{table}[t]", r"\centering", r"\caption{" + caption + "}", r"\label{" + label + "}",
             r"\scriptsize", r"\begin{tabular}{" + "l" * 2 + "r" * (len(cols) - 2) + "}", r"\toprule",
             " & ".join(c.replace("%", r"\%").replace("_", r"\_") for c in cols) + r" \\", r"\midrule"]
    for _, r in df.iterrows():
        lines.append(" & ".join(str(v).replace("%", r"\%").replace("_", r"\_").replace("<", r"$<$")
                                .replace("±", r"$\pm$") for v in r.values) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    df.to_csv(str(path).replace(".tex", ".csv"), index=False)


# ------------------------------------------------------------------ main grid
def load_grid():
    raw = pd.read_csv(ROOT / "results" / "raw_grid.csv")
    raw = raw[raw.n <= 200]
    parts = [raw]
    for f in ("raw_grid_rev_seeds.csv", "raw_grid_rev_baselines.csv"):
        p = REV / f
        if p.exists():
            parts.append(pd.read_csv(p))
    d = pd.concat(parts, ignore_index=True)
    assert d["valid"].astype(str).eq("True").all(), "an invalid row"
    il = raw[raw.method == "ilp"].copy()
    il["proven"] = il["optimal"].astype(str) == "True"
    opt = il.set_index(KEY)[["weight", "proven", "bound"]].rename(columns={"weight": "ilp_w", "bound": "lb"})
    return d, opt


def wide(d, opt):
    """{method: Series W by instance} for unseeded; {method: DataFrame (inst x seed)} for seeded."""
    d = d[d.method != "ilp"]
    un = d[d.model_seed.isna() | (d.model_seed.astype(str) == "")]
    se = d[~(d.model_seed.isna() | (d.model_seed.astype(str) == ""))].copy()
    se["model_seed"] = se["model_seed"].astype(float).astype(int)
    U = un.pivot_table(index=KEY, columns="method", values="weight", aggfunc="first")
    S = {m: g.pivot_table(index=KEY, columns="model_seed", values="weight", aggfunc="first")
         for m, g in se.groupby("method")}
    idx = opt.index
    return U.reindex(idx), {m: s.reindex(idx) for m, s in S.items()}


def compare(ours_W, base_W, ref, mask):
    """ours_W: DataFrame (inst x seed), base_W: Series, ref: Series optimum; mask: boolean Series (instances)."""
    o = ours_W[mask]
    b = base_W[mask]
    r = ref[mask]
    og = 100 * (o.values - r.values[:, None]) / r.values[:, None]
    bg = 100 * (b.values - r.values) / r.values
    om = np.nanmean(og, axis=1)
    per_seed = np.nanmean(og, axis=0)
    diff = om - bg
    w, t, l = wtl(o.values, b.values)
    return dict(n=int(mask.sum()), seeds=list(map(int, o.columns)), ours=float(om.mean()),
                ours_ci_inst=boot_ci(om), ours_ci_seed=seed_ci(per_seed), per_seed=per_seed.tolist(),
                base=float(bg.mean()), base_ci_inst=boot_ci(bg), diff=float(diff.mean()), diff_ci=boot_ci(diff),
                p=wilcoxon_p(np.nanmean(o.values, axis=1), b.values), win=w, tie=t, loss=l)


def ci(c):
    return f"[{c[0]:.2f}, {c[1]:.2f}]"


def main():
    TAB.mkdir(parents=True, exist_ok=True)
    d, opt = load_grid()
    U, S = wide(d, opt)
    proven = opt["proven"]
    ref = opt["ilp_w"]
    seeds = sorted(S["ours_rr"].columns)
    full = S["ours_rr"].notna().all(axis=1)
    print(f"model seeds available for ours: {seeds}; instances with all seeds: {int(full.sum())}/{len(full)}")
    out = {"seeds": [int(s) for s in seeds], "n_proven": int(proven.sum())}

    # ---- pooled main table: every method, gap with CIs
    rows = []
    methods = ["greedy_rr", "greedy_rr_ls", "greedy_tiebreak_rr", "greedy_tiebreak_rr_ls", "greedy_lookahead_rr",
               "greedy_lookahead_rr_ls", "lp_rounding_rr", "lp_rounding_rr_ls", "greedy_random_restart_rr"]
    pooled = {}
    for m in ["ours_rr", "ours_rr_ls", "zero_rr"]:
        W = S[m][proven]
        g = 100 * (W.values - ref[proven].values[:, None]) / ref[proven].values[:, None]
        gm = np.nanmean(g, axis=1)
        per = np.nanmean(g, axis=0)
        opt_pct = 100 * np.nanmean(np.abs(W.values - ref[proven].values[:, None]) < 1e-9)
        pooled[m] = dict(gap=gm.mean(), ci_inst=boot_ci(gm), ci_seed=seed_ci(per), per_seed=per.tolist(),
                         opt=opt_pct, seeds=list(map(int, W.columns)))
        rows.append([m.replace("_rr", "+RR").replace("+RR_ls", "+RR+LS"), len(W.columns), f"{gm.mean():.2f}",
                     ci(boot_ci(gm)), ci(seed_ci(per)), f"{opt_pct:.1f}"])
    for m in methods:
        if m not in U:
            continue
        g = 100 * (U[m][proven] - ref[proven]) / ref[proven]
        opt_pct = 100 * np.mean(np.abs(U[m][proven] - ref[proven]) < 1e-9)
        pooled[m] = dict(gap=float(g.mean()), ci_inst=boot_ci(g.values), opt=float(opt_pct))
        rows.append([m, "-", f"{g.mean():.2f}", ci(boot_ci(g.values)), "-", f"{opt_pct:.1f}"])
    df = pd.DataFrame(rows, columns=["method", "seeds", "gap %", "95% CI (instances)", "95% CI (seeds)", "optimal %"])
    tex_table(df, TAB / "main_pooled.tex",
              f"Mean gap to the proven ILP optimum, all classes, schemes and n = 20--200 ({int(proven.sum())} "
              f"instances). CIs: bootstrap over instances (10,000 resamples) and t-interval over model seeds.",
              "tab:main_pooled")
    out["pooled"] = pooled

    # ---- ours vs every baseline: pooled, per scheme, per class x scheme
    pairs = [("ours_rr", b) for b in ["greedy_rr", "greedy_tiebreak_rr", "greedy_lookahead_rr", "lp_rounding_rr",
                                      "greedy_random_restart_rr", "greedy_rr_ls"]] + \
            [("ours_rr_ls", b) for b in ["greedy_rr_ls", "greedy_tiebreak_rr_ls", "greedy_lookahead_rr_ls",
                                         "lp_rounding_rr_ls"]]
    allcmp, rows = {}, []
    for o, b in pairs:
        if b not in U:
            continue
        res = {"pooled": compare(S[o], U[b], ref, proven)}
        for s in SCH:
            res[s] = compare(S[o], U[b], ref, proven & (opt.index.get_level_values("scheme") == s))
        cells = {}
        for c in CLS:
            for s in SCH:
                mk = proven & (opt.index.get_level_values("cls") == c) & (opt.index.get_level_values("scheme") == s)
                cells[f"{c}/{s}"] = compare(S[o], U[b], ref, mk)
        hp = holm([v["p"] for v in cells.values()])
        for (k, v), h in zip(cells.items(), hp):
            v["p_holm"] = float(h)
        res["cells"] = cells
        allcmp[f"{o} vs {b}"] = res
        for k in ["pooled"] + SCH:
            r = res[k]
            rows.append([o, b, k, r["n"], f"{r['ours']:.2f}", f"{r['base']:.2f}", f"{r['diff']:+.2f}",
                         ci(r["diff_ci"]), fmt_p(r["p"]), f"{r['win']:.1f}/{r['tie']:.1f}/{r['loss']:.1f}"])
    df = pd.DataFrame(rows, columns=["ours", "baseline", "subset", "inst", "ours gap %", "base gap %",
                                     "diff (pp)", "95% CI diff", "Wilcoxon p", "W/T/L %"])
    tex_table(df, TAB / "ours_vs_baselines.tex",
              "Ours (mean over model seeds) vs each baseline on the main grid: mean gap, paired difference "
              "(negative = ours better) with bootstrap 95% CI, two-sided Wilcoxon signed-rank p, and win/tie/loss "
              "over (instance, seed) pairs.", "tab:ours_vs_baselines")
    out["compare"] = allcmp

    # ---- fair Table III: class x scheme, both pairs, CIs, p, W/T/L
    rows = []
    a, bb = allcmp["ours_rr vs greedy_rr"]["cells"], allcmp["ours_rr_ls vs greedy_rr_ls"]["cells"]
    for c in CLS:
        for s in SCH:
            x, y = a[f"{c}/{s}"], bb[f"{c}/{s}"]
            rows.append([CLS_LABEL[c], s.capitalize(),
                         f"{x['ours']:.2f} {ci(x['ours_ci_inst'])}", f"{x['base']:.2f}", fmt_p(x["p"]),
                         f"{x['win']:.0f}/{x['tie']:.0f}/{x['loss']:.0f}",
                         f"{y['ours']:.2f} {ci(y['ours_ci_inst'])}", f"{y['base']:.2f}", fmt_p(y["p"]),
                         f"{y['win']:.0f}/{y['tie']:.0f}/{y['loss']:.0f}"])
    for k, lab in (("pooled", "All"),):
        x, y = allcmp["ours_rr vs greedy_rr"][k], allcmp["ours_rr_ls vs greedy_rr_ls"][k]
        rows.append([lab, "all", f"{x['ours']:.2f} {ci(x['ours_ci_inst'])}", f"{x['base']:.2f}", fmt_p(x["p"]),
                     f"{x['win']:.0f}/{x['tie']:.0f}/{x['loss']:.0f}", f"{y['ours']:.2f} {ci(y['ours_ci_inst'])}",
                     f"{y['base']:.2f}", fmt_p(y["p"]), f"{y['win']:.0f}/{y['tie']:.0f}/{y['loss']:.0f}"])
    df = pd.DataFrame(rows, columns=["class", "scheme", "ours+RR [95% CI]", "greedy+RR", "p", "W/T/L %",
                                     "ours+RR+LS [95% CI]", "greedy+RR+LS", "p (LS)", "W/T/L % (LS)"])
    tex_table(df, TAB / "table3_fair.tex",
              f"Mean gap (\\%) to the proven ILP optimum by class and scheme, n = 20--200, ours = mean over "
              f"{len(seeds)} model seeds. Both columns pairs are comparable: ours+RR vs greedy+RR and ours+RR+LS vs "
              f"greedy+RR+LS (same 1 s LS). p = paired Wilcoxon signed-rank; W/T/L over (instance, seed) pairs.",
              "tab:table3_fair")

    # ---- counts of cells won (by mean) and significant
    for key in ["ours_rr vs greedy_rr", "ours_rr_ls vs greedy_rr_ls", "ours_rr vs greedy_tiebreak_rr",
                "ours_rr vs greedy_lookahead_rr", "ours_rr vs lp_rounding_rr", "ours_rr vs greedy_random_restart_rr"]:
        if key not in allcmp:
            continue
        cells = allcmp[key]["cells"]
        out.setdefault("cell_counts", {})[key] = dict(
            ours_lower=sum(v["ours"] < v["base"] for v in cells.values()),
            sig_ours=sum(v["p"] < 0.05 and v["diff"] < 0 for v in cells.values()),
            sig_base=sum(v["p"] < 0.05 and v["diff"] > 0 for v in cells.values()),
            sig_ours_holm=sum(v["p_holm"] < 0.05 and v["diff"] < 0 for v in cells.values()),
            sig_base_holm=sum(v["p_holm"] < 0.05 and v["diff"] > 0 for v in cells.values()))

    (REV / "stats_main.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("seeds", "n_proven", "cell_counts")}, indent=1, default=float))
    for k in ("ours_rr", "ours_rr_ls", "greedy_rr", "greedy_rr_ls"):
        print(k, {a: (round(b, 3) if isinstance(b, float) else b) for a, b in out["pooled"][k].items()})


if __name__ == "__main__":
    sys.exit(main())
