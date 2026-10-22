"""Revision step 1/2: fair exact cross-check tables (paper Tables V/VI) with greedy + RR + LS added.

    python scripts/rev_table_exact.py

Inputs: results/exact_vs_ours_final.csv (400 fresh graphs, 5 weightings; ours = model seeds 0 and 1) and
results/revision/greedy_rr_ls_final.csv (step 1, same instances); same for the first study (exact_vs_ours.csv /
greedy_rr_ls_first.csv). Gap = 100 (W - W_exact) / W_exact over instances with a proven exact optimum.
Outputs: results/revision/tables/exact_fair_<study>.tex|csv, exact_fair_n1000_degree.tex, exact_fair.json
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from stats import boot_ci, fmt_p, tex_table, wilcoxon_p, wtl  # noqa: E402

REV = ROOT / "results" / "revision"
LAB = {"tree": "Trees", "unit_interval": "Unit interval", "convex_bip": "Convex bip.", "two_tree": "2-trees"}
K = ["cls", "n", "gidx", "scheme"]


def load(study):
    main = pd.read_csv(ROOT / "results" / ("exact_vs_ours_final.csv" if study == "final" else "exact_vs_ours.csv"))
    ls = pd.read_csv(REV / f"greedy_rr_ls_{study}.csv")
    d = pd.concat([main, ls], ignore_index=True)
    d = d[d.method.isin(["exact", "ours_rr", "ours_rr_ls", "greedy_rr", "greedy_rr_ls"])]
    assert d.valid.astype(str).eq("True").all()
    ex = d[d.method == "exact"].set_index(K)
    ex = ex[ex.exact_proven.astype(str) == "True"]["weight"]
    un = d[d.method.isin(["greedy_rr", "greedy_rr_ls"])].pivot_table(index=K, columns="method", values="weight")
    se = {m: d[d.method == m].pivot_table(index=K, columns="model_seed", values="weight")
          for m in ("ours_rr", "ours_rr_ls")}
    return ex, un.reindex(ex.index), {m: s.reindex(ex.index) for m, s in se.items()}


def block(ex, un, se, mask):
    r = ex[mask]
    out = {}
    for o, b in (("ours_rr", "greedy_rr"), ("ours_rr_ls", "greedy_rr_ls")):
        ow, bw = se[o][mask], un[b][mask]
        og = 100 * (ow.values - r.values[:, None]) / r.values[:, None]
        bg = 100 * (bw.values - r.values) / r.values
        w, t, l = wtl(ow.values, bw.values)
        out[o] = dict(ours=float(np.nanmean(og)), ours_ci=boot_ci(np.nanmean(og, axis=1)), base=float(bg.mean()),
                      p=wilcoxon_p(ow.mean(axis=1).values, bw.values), win=w, tie=t, loss=l, n=int(mask.sum()))
    return out


def main():
    allres = {}
    for study in ("final", "first"):
        ex, un, se = load(study)
        rows, res = [], {}
        cls_l = ex.index.get_level_values("cls")
        sch_l = ex.index.get_level_values("scheme")
        for c in LAB:
            for s in ("unit", "degree", "inverse", "random", "mixed"):
                mk = (cls_l == c) & (sch_l == s)
                if not mk.any():
                    continue
                b = block(ex, un, se, mk)
                res[f"{c}/{s}"] = b
                x, y = b["ours_rr"], b["ours_rr_ls"]
                rows.append([LAB[c], s, x["n"], f"{x['ours']:.2f}", f"{x['base']:.2f}", fmt_p(x["p"]),
                             f"{x['win']:.0f}/{x['tie']:.0f}/{x['loss']:.0f}", f"{y['ours']:.2f}", f"{y['base']:.2f}",
                             fmt_p(y["p"]), f"{y['win']:.0f}/{y['tie']:.0f}/{y['loss']:.0f}"])
        cols = ["class", "weights", "inst", "ours+RR", "greedy+RR", "p", "W/T/L %", "ours+RR+LS", "greedy+RR+LS",
                "p (LS)", "W/T/L % (LS)"]
        tex_table(pd.DataFrame(rows, columns=cols), REV / "tables" / f"exact_fair_{study}.tex",
                  f"Exact cross-check ({study} study): mean gap (\\%) to the exact optimum, all n. Comparable pairs: "
                  "ours+RR vs greedy+RR and ours+RR+LS vs greedy+RR+LS (same 1 s LS). Ours = model seeds 0 and 1; "
                  "p = paired Wilcoxon; W/T/L over (graph, seed) pairs.", f"tab:exact_fair_{study}")
        allres[study] = res
        if study == "final":                       # paper Table V format: degree weights, gap at n = 1000
            rows = []
            for c in LAB:
                mk = (cls_l == c) & (sch_l == "degree") & (ex.index.get_level_values("n") == 1000)
                mall = (cls_l == c) & (sch_l == "degree")
                b, ball = block(ex, un, se, mk), block(ex, un, se, mall)
                rows.append([LAB[c], f"{b['ours_rr']['ours']:.2f}", f"{b['ours_rr']['base']:.2f}",
                             f"{b['ours_rr_ls']['ours']:.2f}", f"{b['ours_rr_ls']['base']:.2f}",
                             f"{ball['ours_rr']['win']:.0f}%", f"{ball['ours_rr_ls']['win']:.0f}%",
                             fmt_p(ball["ours_rr_ls"]["p"])])
                allres.setdefault("table5", {})[c] = dict(n1000=b, all_n=ball)
            tex_table(pd.DataFrame(rows, columns=["class", "ours+RR", "greedy+RR", "ours+RR+LS", "greedy+RR+LS",
                                                  "ours+RR better", "ours+RR+LS better", "p (LS, all n)"]),
                      REV / "tables" / "exact_fair_n1000_degree.tex",
                      "Fair Table V: exact cross-check under degree weights; gap (\\%) at n = 1000 and share of "
                      "(graph, seed) pairs over all n where ours is strictly cheaper than greedy with the same "
                      "post-processing.", "tab:exact_fair_t5")
    (REV / "exact_fair.json").write_text(json.dumps(allres, indent=1, default=float), encoding="utf-8")
    t5 = allres["table5"]
    for c, v in t5.items():
        print(c, {k: round(v["n1000"][k]["ours"], 2) for k in v["n1000"]},
              {k: round(v["n1000"][k]["base"], 2) for k in v["n1000"]},
              "win% all n:", {k: round(v["all_n"][k]["win"], 1) for k in v["all_n"]},
              "p LS:", v["all_n"]["ours_rr_ls"]["p"])


if __name__ == "__main__":
    main()
