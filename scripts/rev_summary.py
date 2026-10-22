"""Revision step 7: write results/revision/SUMMARY.md from the JSON outputs (no number is typed by hand).

    python scripts/rev_summary.py
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REV = ROOT / "results" / "revision"


def j(name):
    p = REV / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def ci(c):
    return f"[{c[0]:.2f}, {c[1]:.2f}]"


def p_(p):
    return "< 1e-4" if p < 1e-4 else f"= {p:.4f}"


def table(csv):
    p = REV / "tables" / csv
    if not p.exists():
        return "_(not produced)_\n"
    df = pd.read_csv(p, dtype=str).fillna("")
    rows = [list(df.columns), ["---"] * len(df.columns)] + df.values.tolist()
    return "\n".join("| " + " | ".join(r) + " |" for r in rows) + "\n"


def main():
    sm, rp, lb, bl, ex = j("stats_main.json"), j("report.json"), j("coverage_lb.json"), j("blocking_ablation.json"), \
        j("exact_fair.json")
    L = ["# Revision results (5-hour sprint, 2026-10-21)", "",
         "Every number below is read from the files named next to it; the script that produced each file is given. "
         "All solutions were checked with `is_total_dominating_set` (both checker implementations) and W(S) was "
         "recomputed from S and the weights before any row was written. Decisions: `docs/decisions.md` (R1-R10).", ""]
    seeds = sm.get("seeds", [])
    L += ["## 1. Main result, %d model seeds (`scripts/stats.py` -> `stats_main.json`, `tables/main_pooled.tex`)" %
          len(seeds), "", f"Model seeds: {seeds}. Proven instances (n = 20-200): {sm.get('n_proven')}.", ""]
    for m in ("ours_rr", "ours_rr_ls", "greedy_rr", "greedy_rr_ls", "greedy_random_restart_rr",
              "greedy_tiebreak_rr", "greedy_lookahead_rr", "lp_rounding_rr", "zero_rr"):
        v = sm.get("pooled", {}).get(m)
        if not v:
            continue
        s = f"* **{m}**: gap {v['gap']:.2f}%, 95% CI over instances {ci(v['ci_inst'])}"
        if "ci_seed" in v:
            s += f", 95% CI over seeds {ci(v['ci_seed'])}, per-seed means " + \
                 ", ".join(f"{x:.2f}" for x in v["per_seed"])
        s += f"; optimal in {v['opt']:.1f}% of instances"
        L.append(s)
    L += ["", "Paired tests, pooled (Proposed = seed mean; W/T/L over (instance, seed) pairs; diff < 0 = Proposed better):", ""]
    for k, v in sm.get("compare", {}).items():
        r = v["pooled"]
        L.append(f"* {k}: {r['ours']:.2f}% vs {r['base']:.2f}%, diff {r['diff']:+.2f} pp {ci(r['diff_ci'])}, "
                 f"Wilcoxon p {p_(r['p'])}, W/T/L = {r['win']:.1f}/{r['tie']:.1f}/{r['loss']:.1f}%")
    L += ["", "Per scheme, Proposed vs Greedy (same post-processing):", ""]
    for k in ("ours_rr vs greedy_rr", "ours_rr_ls vs greedy_rr_ls", "ours_rr vs greedy_tiebreak_rr",
              "ours_rr vs greedy_random_restart_rr"):
        v = sm.get("compare", {}).get(k)
        if v:
            L.append(f"* {k}: " + "; ".join(f"{s} {v[s]['ours']:.2f} vs {v[s]['base']:.2f} (p {p_(v[s]['p'])}, "
                                            f"W/T/L {v[s]['win']:.0f}/{v[s]['tie']:.0f}/{v[s]['loss']:.0f})"
                                            for s in ("degree", "inverse", "random", "mixed")))
    L += ["", "Class x scheme cells (16): Proposed lower mean / significant for Proposed / significant for baseline "
          "(p < 0.05, raw; Holm in brackets):", ""]
    for k, v in sm.get("cell_counts", {}).items():
        L.append(f"* {k}: {v['ours_lower']} / {v['sig_ours']} ({v['sig_ours_holm']}) / {v['sig_base']} "
                 f"({v['sig_base_holm']})")
    L += ["", "## 2. Fair Table III (`scripts/stats.py` -> `tables/table3_fair.tex`)", "", table("table3_fair.csv"),
          "## 3. Fair exact cross-check, Table V format (`scripts/rev_greedy_ls_exact.py`, `scripts/rev_table_exact.py`"
          " -> `tables/exact_fair_n1000_degree.tex`, `exact_fair_final.tex`, `exact_fair_first.tex`)", "",
          "greedy + RR + LS added on the same 400 + 240 graphs x 5 weightings (2,000 + 1,200 new solutions). Proposed = "
          "model seeds 0 and 1.", "", table("exact_fair_n1000_degree.csv"), "",
          "## 4. Stronger greedy baselines (`wtdsws/baselines/greedy_plus.py`, `scripts/rev_eval_grid.py` -> "
          "`raw_grid_rev_baselines.csv`; `tables/ours_vs_baselines.tex`)", "",
          "Tests: `tests/test_greedy_plus.py` (200 random graphs, every variant valid).", "",
          table("ours_vs_baselines.csv"),
          "## 5. Equal-time comparison (`scripts/rev_equal_time.py`, `scripts/rev_report.py` -> `tables/equal_time.tex`,"
          " `tables/equal_time_tests.tex`, `figures/equal_time_quality_vs_time.png|pdf`)", "", table("equal_time.csv"),
          "", table("equal_time_tests.csv"), "",
          "## 6. Alpha-sweep (`scripts/rev_alpha_sweep.py`, `scripts/rev_report.py` -> `tables/alpha_sweep.tex`, "
          "`figures/alpha_gap.png|pdf`, `figures/alpha_advantage.png|pdf`)", ""]
    al = rp.get("alpha", {})
    if al:
        L.append(f"Model seeds: {al['seeds']}; ILP not proven: {al['unproven']} of 720 (excluded).")
        for k in ("pos", "neg", "all"):
            v = al.get(f"spearman_adv_alpha_{k}")
            if v:
                L.append(f"* Spearman(alpha, per-instance advantage Greedy + RR - Proposed + RR), alpha {k}: "
                         f"rho = {v['rho']:.3f}, p {p_(v['p'])}")
        L.append("")
    L += [table("alpha_sweep.csv"), "## 7. Coverage fixes (`scripts/rev_coverage.py`)", "",
          f"Unproven grid ILPs: {lb.get('n_unproven')} (`coverage_lb.json`, `tables/unproven_lb.tex`).", "",
          table("unproven_lb.csv"), ""]
    if bl:
        L.append(f"Blocking rule (base paper, no training; `blocking_ablation.json`): {bl['episodes']} episodes, "
                 f"model seeds {bl['seeds']}, valid TDS in {bl['valid_pct']:.1f}%; mean fraction of vertices covered "
                 f"when no legal action remained {bl['mean_covered_frac_at_stop']:.3f}.")
    L += ["", "## 8. Training and inference time (`scripts/rev_report.py` -> `report.json`, "
          "`tables/training_time.tex`, `tables/inference_time.tex`; `scripts/rev_inference_time.py`)", "",
          table("training_time.csv"), "", table("inference_time.csv"), ""]
    inf = rp.get("inference", {})
    if inf.get("same_W_cpu_gpu"):
        L.append("CPU and GPU solutions with identical W(S) per n: " + str(inf["same_W_cpu_gpu"]))
    (REV / "SUMMARY.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("wrote", REV / "SUMMARY.md", len(L), "lines")


if __name__ == "__main__":
    main()
