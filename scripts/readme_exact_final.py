"""Fill the README section "Exact vs ours: pilot -> choose -> fresh test" from results/chosen_settings.json,
results/exact_vs_ours_final_summary.json, results/tables/exact_vs_ours_final.csv and results/stress_test_summary.json
(no hand-typed numbers). Idempotent: the section lives between <!-- EXACT-FINAL:begin/end --> markers, inserted
after <!-- EXACT:end --> the first time.

    python scripts/readme_exact_final.py
"""
import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
CH = json.loads((RES / "chosen_settings.json").read_text(encoding="utf-8"))
S = json.loads((RES / "exact_vs_ours_final_summary.json").read_text(encoding="utf-8"))
T = pd.read_csv(RES / "tables/exact_vs_ours_final.csv")
ST = RES / "stress_test_summary.json"
NAME = {"tree": "Trees", "unit_interval": "Unit-interval", "convex_bip": "Convex bipartite", "two_tree": "2-trees"}
PARAM = {"random_tree": "random labelled tree", "ba_m1": "BA m = 1", "deg3": "average degree 3",
         "deg6": "average degree 6", "deg10": "average degree 10", "k2": "as generated (k = 2)"}
FIG = "results/figures/exact_vs_ours"


def md_table(rows):
    cols = list(rows[0])
    esc = lambda x: str(x).replace("|", "\\|")
    out = ["| " + " | ".join(map(esc, cols)) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    return "\n".join(out + ["| " + " | ".join(esc(r[c]) for c in cols) + " |" for r in rows])


def chosen_rows():
    rows = []
    for c, v in CH["chosen"].items():
        ru = v["runner_up"]
        rows.append({"class": NAME[c], "parameter": PARAM[v["param"]],
                     "n grid": f"{v['grid']} ({v['ns'][0]}–{v['ns'][-1]})",
                     "separation, all weightings (pp)": f"{v['separation']:+.2f}",
                     "weighting for Plots 2–5": v["weighting"],
                     "separation, that weighting (pp)": f"{v['separation_by_weighting'][v['weighting']]:+.2f}",
                     "exact proven (pilot)": f"{v['proven_pct']:.0f}%",
                     "runner-up": "—" if ru is None else f"{PARAM[ru['param']]}, grid {ru['grid']} "
                                                          f"({ru['separation']:+.2f})"})
    return md_table(rows)


def largest_rows():
    rows = []
    for c, v in S["per_class"].items():
        L = v["largest_n"]
        rows.append({"class": NAME[c], "weighting": v["weighting"], "n": int(L["n"]),
                     "W exact": f"{L['W_exact']:.1f}", "W ours": f"{L['W_ours']:.1f}",
                     "W greedy": f"{L['W_greedy']:.1f}", "gap ours %": f"{L['gap_ours_pct']:.2f}",
                     "gap greedy %": f"{L['gap_greedy_pct']:.2f}", "ours optimal %": f"{L['opt_ours_pct']:.0f}",
                     "greedy optimal %": f"{L['opt_greedy_pct']:.0f}",
                     "ours + LS gap %": f"{L['gap_ours_ls_pct']:.2f}"})
    return md_table(rows)


def all_n_rows():
    rows = []
    for c, v in S["per_class"].items():
        a, u = v["chosen_all_n"], v["unit_all_n"]
        rows.append({"class": NAME[c], "weighting": v["weighting"],
                     "gap ours / greedy % (chosen)": f"{a['gap_ours_pct']:.2f} / {a['gap_greedy_pct']:.2f}",
                     "separation (pp)": f"{a['separation_pp']:+.2f}",
                     "optimal % ours / greedy (chosen)": f"{a['opt_ours_pct']:.0f} / {a['opt_greedy_pct']:.0f}",
                     "graphs ours better / greedy better": f"{a['ours_better_graphs']} / {a['greedy_better_graphs']}"
                                                           f" (of {a['graphs']})",
                     "gap ours / greedy % (unit)": f"{u['gap_ours_pct']:.2f} / {u['gap_greedy_pct']:.2f}",
                     "optimal % ours / greedy (unit)": f"{u['opt_ours_pct']:.0f} / {u['opt_greedy_pct']:.0f}"})
    return md_table(rows)


def observations():
    obs = []
    pc = S["per_class"]
    better = [c for c in pc if pc[c]["chosen_all_n"]["separation_pp"] > 0]
    worse = [c for c in pc if pc[c]["chosen_all_n"]["separation_pp"] <= 0]
    obs.append("With the chosen weighting, pooled over all n of the fresh test graphs, ours + RR has a smaller mean "
               "gap than greedy + RR on " + (", ".join(NAME[c] for c in better) or "no class") + ".")
    if worse:
        obs.append("**Ours does NOT beat greedy + RR** on " + ", ".join(
            f"{NAME[c]} (gap ours {pc[c]['chosen_all_n']['gap_ours_pct']:.2f}% vs greedy "
            f"{pc[c]['chosen_all_n']['gap_greedy_pct']:.2f}%)" for c in worse) + " with the chosen weighting.")
    uw = [c for c in pc if pc[c]["unit_all_n"]["gap_ours_pct"] > pc[c]["unit_all_n"]["gap_greedy_pct"]]
    obs.append("With unit weights (Plot 1), greedy + RR has the smaller mean gap on "
               + (", ".join(NAME[c] for c in uw) or "no class") + "; ours has the smaller or equal gap on "
               + (", ".join(NAME[c] for c in pc if c not in uw) or "no class") + ".")
    for c in pc:
        p_sep = CH["chosen"][c]["separation_by_weighting"][pc[c]["weighting"]]
        obs.append(f"{NAME[c]}: the pilot separation for the chosen weighting was {p_sep:+.2f} pp; on the fresh "
                   f"test graphs it is {pc[c]['chosen_all_n']['separation_pp']:+.2f} pp "
                   f"({'same sign' if p_sep * pc[c]['chosen_all_n']['separation_pp'] > 0 else 'different sign'}).")
    ls = [c for c in pc if pc[c]["chosen_all_n"]["gap_ours_ls_pct"] < pc[c]["chosen_all_n"]["gap_greedy_pct"]]
    obs.append("Ours + RR + 1 s local search (table only) has a smaller mean gap than greedy + RR on "
               + (", ".join(NAME[c] for c in ls) or "no class") + ".")
    return "\n".join(f"* {o}" for o in obs)


def stress():
    if not ST.exists():
        return "Not run."
    s = json.loads(ST.read_text(encoding="utf-8"))
    rows = [{"n": int(r["n"]), "W exact": f"{r['W_exact']:.1f}", "gap ours %": f"{r['gap_ours_rr_pct']:.2f}",
             "gap ours + LS %": f"{r['gap_ours_rr_ls_pct']:.2f}", "gap greedy %": f"{r['gap_greedy_rr_pct']:.2f}",
             "ours optimal %": f"{r['opt_ours_rr_pct']:.0f}", "greedy optimal %": f"{r['opt_greedy_rr_pct']:.0f}"}
            for r in s["per_n"]]
    return (f"**Stress test (adversarial for greedy). Separate from the results above and never mixed into them.** "
            f"{s['family']} Degree weights, 10 test-range graphs per n, {s['solutions_checked']:,} checked solutions; "
            f"DP vs ILP cross-check {s['ilp_crosscheck_agree']}/{s['ilp_crosscheck_runs']}.\n\n"
            f"![Stress test]({FIG}/stress_test.png)\n\n" + md_table(rows))


x = S["ilp_crosscheck"]
section = f"""## Exact vs ours: pilot → choose → fresh test (updated plots)

The first exact-vs-ours run (previous section) mixed tree families and averaged densities, so the three methods
often overlapped on the plots. To show where they **do** differ, without selection bias, we used three steps:

1. **Pilot sweep** (`scripts/pilot_sweep.py`). The graphs come from a separate **pilot seed range**
   `SEED_RANGES["pilot"] = [1.4e9, 1.5e9)`, which is disjoint from train / val / test
   (`tests/test_ws_schemes.py::test_pilot_seed_range_disjoint`). We tried 9 class parameters: trees (random labelled
   vs BA m = 1), unit-interval and convex bipartite (average degree 3 / 6 / 10), and 2-trees. Each was run on two
   n grids, A = 100…1000 and B = 200…2000, with 5 graphs per n and 5 weightings. That gives
   {CH['pilot_graphs']} graphs and {CH['pilot_instances']:,} instances; {CH['pilot_unproven']} exact optima were not
   proven. Methods: exact, ours + RR (best.pt) and greedy + RR. All results:
   `results/pilot_sweep.csv` (every instance), `pilot_sweep_per_graph.csv`, `pilot_sweep_settings.csv`,
   `pilot_sweep_choice.csv`.
2. **Choose by rule** (`scripts/pilot_choose.py`, `results/chosen_settings.json`). SEPARATION = greedy gap % −
   ours gap % (vs the exact optimum). For each class we chose the (parameter, grid) with the largest mean separation,
   among the candidates where ours was always valid and the exact optimum was proven on ≥ 95% of instances. Plots 2–5
   use the non-unit weighting with the largest separation for that class.
3. **Fresh test** (`scripts/exact_vs_ours_final.py`). New graphs drawn from the **test** seed range, 10 per n (pilot
   graphs are never reused; this is asserted). {S['graphs']} graphs × 5 weightings, {S['solutions_checked']:,}
   solutions, all checked with `is_total_dominating_set` and W(S) recomputed. Exact optima not proven:
   **{len(S['exact_not_proven'])}**. DP vs ILP cross-check: {x['agree']}/{x['runs']} agree.

The choice in step 2 was made to show differences, so the pilot separations are optimistic. Only the fresh-test
numbers below are results.

![Pilot heatmap]({FIG}/pilot_separation_heatmap.png)

### Chosen settings (from the pilot)
{chosen_rows()}

### Plots (fresh test graphs; ours = mean of 2 checkpoints, error bars = std over graphs)
Plot 1 · |S|, unit weights
![Plot 1]({FIG}/final_plot1_cardinality_unit.png)

Plot 2 · |S|, chosen weighting. *With weighted graphs, several optimal sets can have the same weight but different
sizes, so |S| can differ even when ours is optimal (see Plots 3–5).*
![Plot 2]({FIG}/final_plot2_cardinality_chosen.png)

Plot 3 · W(S), chosen weighting
![Plot 3]({FIG}/final_plot3_weight_chosen.png)

Plot 4 (extra cost W(S) − W_exact) is archived: `results/figures/archive/final_plot4_extra_cost_chosen.png/.pdf`.

Plot 5 · Approximation ratio W(S) / W_exact
![Plot 5]({FIG}/final_plot5_ratio_chosen.png)

**Paper figures** (`scripts/plot_paper.py`, redrawn from the same CSV): `results/figures/paper/` with
plot1_cardinality_unit, plot2_cardinality_weighted, plot3_total_weight, plot5_ratio and **paper_main** (rows Plot 1 /
Plot 3 / Plot 5, columns the classes). Each panel states "Ours beats Greedy in X% of graphs": the share of
(graph, checkpoint) pairs, over all n, where ours + RR is strictly better than greedy + RR (|S| for unit weights,
W(S) otherwise), with ties reported separately and X rounded down. Paper table: `results/tables/paper_table.tex`.

![Paper figure](results/figures/paper/paper_main.png)

### Largest n per class (chosen weighting; mean over 10 fresh graphs)
{largest_rows()}

### All n pooled
{all_n_rows()}

Full table (every class × weighting × n, with ratios, ours + RR + LS and runtimes):
`results/tables/exact_vs_ours_final.csv` / `.tex`.

### Observations (generated from the data by `scripts/readme_exact_final.py`)
{observations()}

### Stress test
{stress()}

### Reproduce
```bash
python scripts/pilot_sweep.py           # pilot seeds only -> results/pilot_sweep.csv
python scripts/pilot_choose.py          # rule-based choice -> results/chosen_settings.json + heatmap
python scripts/exact_vs_ours_final.py   # fresh test graphs -> results/exact_vs_ours_final.csv
python scripts/plot_exact_vs_ours_final.py
python scripts/stress_test.py           # optional, separate figure
python scripts/readme_exact_final.py    # this section
```
"""

readme = (ROOT / "README.md").read_text(encoding="utf-8")
block = f"<!-- EXACT-FINAL:begin -->\n{section}<!-- EXACT-FINAL:end -->"
if "<!-- EXACT-FINAL:begin -->" in readme:
    readme = re.sub(r"<!-- EXACT-FINAL:begin -->.*?<!-- EXACT-FINAL:end -->", lambda m: block, readme, flags=re.S)
else:
    readme = readme.replace("<!-- EXACT:end -->", "<!-- EXACT:end -->\n\n" + block, 1)
(ROOT / "README.md").write_text(readme, encoding="utf-8")
print("README section written:", len(section), "chars")
