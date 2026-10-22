"""Fill the README section "Cross-check against exact algorithms" from results/exact_vs_ours_summary.json,
results/tables/exact_vs_ours.csv and results/exact_validation.json (no hand-typed numbers). Idempotent: the
section lives between <!-- EXACT:begin/end --> markers, inserted after <!-- RESULTS:end --> the first time.

    python scripts/readme_exact.py
"""
import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
S = json.loads((ROOT / "results/exact_vs_ours_summary.json").read_text(encoding="utf-8"))
T = pd.read_csv(ROOT / "results/tables/exact_vs_ours.csv")
V = S["validation"]
NAME = {"tree": "Trees", "unit_interval": "Unit-interval", "convex_bip": "Convex bipartite", "two_tree": "2-trees"}
P = S["optimal_pct"]
f0 = lambda x: f"{x:.0f}"
f2 = lambda x: f"{x:.2f}"


def md_table(rows, cols):
    esc = lambda x: str(x).replace("|", "\\|")              # "|S|" must not split a Markdown cell
    out = ["| " + " | ".join(map(esc, cols)) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    return "\n".join(out + ["| " + " | ".join(esc(r[c]) for c in cols) + " |" for r in rows])


def headline(n):
    rows = []
    for c in NAME:
        u = T[(T.cls == c) & (T.scheme == "unit") & (T.n == n)].iloc[0]
        r = T[(T.cls == c) & (T.scheme == "random") & (T.n == n)].iloc[0]
        rows.append({"class": NAME[c], "|S| exact (unit)": f"{u.S_exact:.1f}", "|S| ours (unit)": f"{u.S_ours:.2f}",
                     "W exact (random)": f"{r.W_exact:.1f}", "W ours (random)": f"{r.W_ours:.2f}",
                     "gap W % (random)": f2(r.gap_W_pct), "ours optimal % (random)": f0(r.opt_W_pct)})
    return md_table(rows, list(rows[0]))


def opt_rows():
    rows = []
    for c in NAME:
        p = P[c]
        rows.append({"class": NAME[c], "ours + RR, unit": f0(p["unit_opt_pct"]),
                     "ours + RR, weighted (W)": f0(p["weighted_opt_W_pct"]),
                     "same |S| as exact, weighted": f0(p["weighted_same_S_pct"]),
                     "ours + RR + LS, unit": f0(p["unit_opt_pct_ours_ls"]),
                     "ours + RR + LS, weighted": f0(p["weighted_opt_W_pct_ours_ls"]),
                     "greedy + RR, unit": f0(p["unit_opt_pct_greedy"]),
                     "greedy + RR, weighted": f0(p["weighted_opt_W_pct_greedy"]),
                     "mean gap % ours (unit / weighted)": f"{p['mean_gap_unit_pct']:.2f} / {p['mean_gap_weighted_pct']:.2f}",
                     "mean gap % greedy (unit / weighted)":
                         f"{p['mean_gap_unit_greedy_pct']:.2f} / {p['mean_gap_weighted_greedy_pct']:.2f}"})
    return md_table(rows, list(rows[0]))


def observations():
    obs = []
    best = max(NAME, key=lambda c: P[c]["unit_opt_pct"])
    worst = min(NAME, key=lambda c: P[c]["unit_opt_pct"])
    obs.append(f"With unit weights, ours + RR reaches the exact optimum most often on {NAME[best].lower()} "
               f"({f0(P[best]['unit_opt_pct'])}% of graph × seed runs) and least often on {NAME[worst].lower()} "
               f"({f0(P[worst]['unit_opt_pct'])}%).")
    w = T[T.scheme != "unit"]
    small, large = w[w.n <= 100].opt_W_pct.mean(), w[w.n >= 500].opt_W_pct.mean()
    gs, gl = w[w.n <= 100].gap_W_pct.mean(), w[w.n >= 500].gap_W_pct.mean()
    obs.append(f"On weighted graphs (all classes and schemes pooled), ours + RR is weight-optimal in {f0(small)}% "
               f"of runs for n ≤ 100 and in {f0(large)}% for n ≥ 500. The mean gap in W is {f2(gs)}% for n ≤ 100 "
               f"and {f2(gl)}% for n ≥ 500. Exact optima become rarer for ours as n grows, while the relative gap "
               f"stays {'below' if max(gs, gl) < 5 else 'around'} 5% on average.")
    for kind in ("unit", "weighted"):
        lower = [NAME[c] for c in NAME if P[c][f"mean_gap_{kind}_pct"] < P[c][f"mean_gap_{kind}_greedy_pct"]]
        higher = [NAME[c] for c in NAME if P[c][f"mean_gap_{kind}_pct"] > P[c][f"mean_gap_{kind}_greedy_pct"]]
        obs.append(f"Mean gap vs weighted greedy + RR "
                   f"({'unit weights' if kind == 'unit' else 'the four weight schemes'}): ours is lower on "
                   f"{', '.join(lower) or 'no class'} and higher on {', '.join(higher) or 'no class'}.")
    ls_gain = [P[c]["weighted_opt_W_pct_ours_ls"] - P[c]["weighted_opt_W_pct"] for c in NAME]
    obs.append(f"Adding 1 s of local search raises the share of weight-optimal runs on weighted graphs by "
               f"{f0(min(ls_gain))}–{f0(max(ls_gain))} percentage points, depending on the class (table only, not "
               f"plotted).")
    diff = [NAME[c] for c in NAME if P[c]["weighted_same_S_pct"] > P[c]["weighted_opt_W_pct"]]
    obs.append(f"Classes where, on weighted graphs, ours has the same |S| as exact more often than the same W: "
               f"{', '.join(diff) or 'none'}. Equal size does not mean optimal, so Plot 3 (W), not Plot 2 (|S|), "
               f"measures the objective.")
    return "\n".join(f"* {o}" for o in obs)


tb, ti = V["tree"], V["interval"]
x = S["ilp_crosscheck"]
section = f"""## Cross-check against exact algorithms

Four graph classes on which the minimum-weight TDS can be computed exactly. Our method is compared with the exact
optimum on {S['graphs']} new test graphs: 4 classes × n ∈ {{20, 50, 100, 200, 500, 1000}} × 10 graphs. The graph and
weight seeds come from the test range and none of them was used by an earlier run. Each graph is solved under 5
weightings: unit, degree, inverse, random and mixed. That gives {S['solutions_checked']:,} solutions, all of them
checked with `is_total_dominating_set` and with W(S) recomputed (`results/exact_vs_ours.csv`).

**Exact methods** (`wtdsws/baselines/exact_special.py`). All of them return a minimum-weight set and, among the
minimum-weight sets, a smallest one.

| class | exact method | reference |
|---|---|---|
| Trees (5 random labelled trees + 5 BA with m = 1 per n) | linear-time rooted DP, 4 states per vertex | Laskar, Pfaff, Hedetniemi & Hedetniemi (1984), SIAM J. Algebraic Discrete Methods 5(3); weighted version = standard DP |
| Unit-interval | **our exact DP for unit-interval graphs** over the sorted interval model ([docs/exact_interval.md](docs/exact_interval.md)) | exact polynomial algorithms for interval graphs: Bertossi & Gori (1988), SIAM J. Discrete Math. 1(3); Chang (1998), SIAM J. Computing 27(6). These were not reproduced |
| Convex bipartite | Exact (ILP): CP-SAT, 60 s, proven-optimal only | polynomial algorithms exist (Lee 2025, Mathematics 13(3):403) |
| 2-trees | Exact (ILP): CP-SAT, 60 s, proven-optimal only | polynomial via bounded treewidth |

**Validation** (`tests/test_exact_special.py`, `results/exact_validation.json`; this passed before any plot was
made):
* tree DP vs brute force: {tb['bruteforce']['instances']} instances (n ≤ 12, all 5 weightings),
  {tb['bruteforce']['weight_mismatches']} weight / {tb['bruteforce']['size_mismatches']} size mismatches;
  unit-interval DP vs brute force: {ti['bruteforce']['instances']} instances,
  {ti['bruteforce']['weight_mismatches']} / {ti['bruteforce']['size_mismatches']} mismatches.
* tree DP vs proven-optimal CP-SAT: {tb['ilp']['instances']} instances (n = 50–200),
  {tb['ilp']['weight_mismatches']} / {tb['ilp']['size_mismatches']} mismatches; unit-interval DP:
  {ti['ilp']['instances']} instances, {ti['ilp']['weight_mismatches']} / {ti['ilp']['size_mismatches']} mismatches
  ({V['ilp_not_proven']} unproven).
* In the main run, the same ILP also runs on every tree and unit-interval graph: {x['proven']}/{x['runs']} proven,
  and {x['agree']}/{x['runs']} agree with the DP on (W, |S|).
* Exact ILP optima not proven within 60 s (convex bipartite, 2-trees): **{len(S['exact_not_proven'])}**. An unproven
  instance would be drawn as a hollow marker.

**Methods:** Exact; ours = GNN+RL + redundancy removal, averaged over both checkpoints (`checkpoints/best.pt`,
`checkpoints/seed1_best.pt`); weighted greedy + RR. Ours + RR + 1 s local search appears only in the table.

### Plot 1 · Cardinality |S|, unit weights (the fair cardinality comparison)
![Plot 1](results/figures/exact_vs_ours/plot1_cardinality_unit.png)

### Plot 2 · Cardinality |S|, weighted graphs (random scheme shown; degree / inverse / mixed in the same folder)
![Plot 2](results/figures/exact_vs_ours/plot2_cardinality_random.png)

*With weighted graphs, several optimal sets can have the same minimum weight but different sizes, so |S| can differ
from the exact method even when ours is optimal. See Plot 3.* Example (`tests/test_exact_special.py`, verified by
brute force): the path 1–2–3–4–5 with weights {{1:3, 2:4, 3:4, 4:1, 5:1}} has minimum weight 9, attained by two
optimal sets, **{{2,3,4}}** (3 vertices) and **{{1,2,4,5}}** (4 vertices).

### Plot 3 · Total weight W(S), weighted graphs (the objective our system minimises)
![Plot 3](results/figures/exact_vs_ours/plot3_weight_random.png)

Paper figure (3 rows: Plot 1 / Plot 2 / Plot 3 with the random scheme × 4 classes):
`results/figures/exact_vs_ours/combined_paper_figure.png/.pdf`. All figures are 300-dpi PNG plus vector PDF.

### n = 100 and n = 1000 (mean over 10 graphs; ours = mean of 2 seeds)
**n = 100**

{headline(100)}

**n = 1000**

{headline(1000)}

### Share of runs where ours reaches the exact optimum (%; all n, graph × model seed)
{opt_rows()}

Full table per class × scheme × n: `results/tables/exact_vs_ours.csv` / `.tex`. App page: **Exact vs ours**.

### Observations (generated from the data by `scripts/readme_exact.py`)
{observations()}

### Reproduce
```bash
python scripts/validate_exact.py        # exact methods vs brute force and proven ILP -> results/exact_validation.json
python scripts/exact_vs_ours.py         # 240 graphs x 5 weightings, all methods   -> results/exact_vs_ours.csv
python scripts/plot_exact_vs_ours.py    # figures, tables, summary (refuses to run if validation failed)
python scripts/readme_exact.py          # this section
```
"""

readme = (ROOT / "README.md").read_text(encoding="utf-8")
block = f"<!-- EXACT:begin -->\n{section}<!-- EXACT:end -->"
if "<!-- EXACT:begin -->" in readme:
    readme = re.sub(r"<!-- EXACT:begin -->.*?<!-- EXACT:end -->", lambda m: block, readme, flags=re.S)
else:
    readme = readme.replace("<!-- RESULTS:end -->", "<!-- RESULTS:end -->\n\n" + block, 1)
(ROOT / "README.md").write_text(readme, encoding="utf-8")
print("README section written:", len(section), "chars")
