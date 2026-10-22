# Revision results (5-hour sprint, 2026-10-21)

Every number below is read from the files named next to it; the script that produced each file is given. All solutions were checked with `is_total_dominating_set` (both checker implementations) and W(S) was recomputed from S and the weights before any row was written. Decisions: `docs/decisions.md` (R1-R10).

## 1. Main result, 5 model seeds (`scripts/stats.py` -> `stats_main.json`, `tables/main_pooled.tex`)

Model seeds: [0, 1, 2, 3, 4]. Proven instances (n = 20-200): 624.

* **ours_rr**: gap 4.62%, 95% CI over instances [4.25, 4.99], 95% CI over seeds [3.95, 5.29], per-seed means 3.88, 5.15, 4.47, 5.15, 4.44; optimal in 32.0% of instances
* **ours_rr_ls**: gap 2.29%, 95% CI over instances [2.04, 2.56], 95% CI over seeds [1.89, 2.70], per-seed means 1.98, 2.49, 2.10, 2.77, 2.12; optimal in 54.9% of instances
* **greedy_rr**: gap 5.51%, 95% CI over instances [4.96, 6.08]; optimal in 34.8% of instances
* **greedy_rr_ls**: gap 2.76%, 95% CI over instances [2.41, 3.13]; optimal in 54.8% of instances
* **greedy_random_restart_rr**: gap 1.95%, 95% CI over instances [1.64, 2.28]; optimal in 53.5% of instances
* **greedy_tiebreak_rr**: gap 5.57%, 95% CI over instances [5.02, 6.15]; optimal in 34.9% of instances
* **greedy_lookahead_rr**: gap 9.98%, 95% CI over instances [8.96, 11.05]; optimal in 29.0% of instances
* **lp_rounding_rr**: gap 5.89%, 95% CI over instances [5.18, 6.63]; optimal in 51.8% of instances
* **zero_rr**: gap 6.02%, 95% CI over instances [5.48, 6.58], 95% CI over seeds [-2.79, 14.83], per-seed means 5.33, 6.72; optimal in 29.0% of instances

Paired tests, pooled (ours = seed mean; W/T/L over (instance, seed) pairs; diff < 0 = ours better):

* ours_rr vs greedy_rr: 4.62% vs 5.51%, diff -0.89 pp [-1.36, -0.44], Wilcoxon p = 0.0092, W/T/L = 35.9/32.2/31.9%
* ours_rr vs greedy_tiebreak_rr: 4.62% vs 5.57%, diff -0.96 pp [-1.43, -0.51], Wilcoxon p = 0.0033, W/T/L = 36.1/32.2/31.7%
* ours_rr vs greedy_lookahead_rr: 4.62% vs 9.98%, diff -5.36 pp [-6.32, -4.43], Wilcoxon p < 1e-4, W/T/L = 49.6/26.5/23.8%
* ours_rr vs lp_rounding_rr: 4.62% vs 5.89%, diff -1.27 pp [-1.94, -0.63], Wilcoxon p = 0.3389, W/T/L = 33.9/28.1/38.0%
* ours_rr vs greedy_random_restart_rr: 4.62% vs 1.95%, diff +2.67 pp [2.31, 3.03], Wilcoxon p < 1e-4, W/T/L = 10.8/34.4/54.7%
* ours_rr vs greedy_rr_ls: 4.62% vs 2.76%, diff +1.86 pp [1.49, 2.23], Wilcoxon p < 1e-4, W/T/L = 17.4/31.7/50.9%
* ours_rr_ls vs greedy_rr_ls: 2.29% vs 2.76%, diff -0.47 pp [-0.75, -0.20], Wilcoxon p = 0.0174, W/T/L = 25.4/54.0/20.7%
* ours_rr_ls vs greedy_tiebreak_rr_ls: 2.29% vs 2.81%, diff -0.52 pp [-0.81, -0.25], Wilcoxon p = 0.0070, W/T/L = 25.5/53.9/20.6%
* ours_rr_ls vs greedy_lookahead_rr_ls: 2.29% vs 3.05%, diff -0.75 pp [-1.08, -0.44], Wilcoxon p < 1e-4, W/T/L = 28.9/51.0/20.1%
* ours_rr_ls vs lp_rounding_rr_ls: 2.29% vs 1.93%, diff +0.36 pp [0.10, 0.63], Wilcoxon p < 1e-4, W/T/L = 18.9/52.9/28.2%

Per scheme, ours vs greedy (same post-processing):

* ours_rr vs greedy_rr: degree 6.85 vs 9.94 (p < 1e-4, W/T/L 58/16/26); inverse 3.52 vs 2.79 (p = 0.0020, W/T/L 20/46/34); random 3.31 vs 3.26 (p = 0.0273, W/T/L 26/41/32); mixed 4.97 vs 6.36 (p = 0.0215, W/T/L 41/24/36)
* ours_rr_ls vs greedy_rr_ls: degree 4.18 vs 5.84 (p < 1e-4, W/T/L 46/28/26); inverse 1.41 vs 1.35 (p = 0.4398, W/T/L 14/71/15); random 1.56 vs 1.36 (p = 0.0510, W/T/L 15/63/22); mixed 2.16 vs 2.73 (p = 0.0520, W/T/L 28/52/20)
* ours_rr vs greedy_tiebreak_rr: degree 6.85 vs 9.94 (p < 1e-4, W/T/L 58/16/26); inverse 3.52 vs 2.89 (p = 0.0039, W/T/L 20/46/34); random 3.31 vs 3.42 (p = 0.1033, W/T/L 27/41/32); mixed 4.97 vs 6.38 (p = 0.0188, W/T/L 41/24/35)
* ours_rr vs greedy_random_restart_rr: degree 6.85 vs 1.62 (p < 1e-4, W/T/L 9/22/69); inverse 3.52 vs 1.22 (p < 1e-4, W/T/L 6/47/47); random 3.31 vs 2.06 (p < 1e-4, W/T/L 12/40/47); mixed 4.97 vs 2.88 (p < 1e-4, W/T/L 16/27/57)

Class x scheme cells (16): ours lower mean / significant for ours / significant for baseline (p < 0.05, raw; Holm in brackets):

* ours_rr vs greedy_rr: 9 / 4 (2) / 2 (0)
* ours_rr_ls vs greedy_rr_ls: 11 / 2 (0) / 0 (0)
* ours_rr vs greedy_tiebreak_rr: 10 / 4 (2) / 2 (0)
* ours_rr vs greedy_lookahead_rr: 13 / 9 (7) / 3 (0)
* ours_rr vs lp_rounding_rr: 7 / 5 (4) / 6 (5)
* ours_rr vs greedy_random_restart_rr: 0 / 0 (0) / 16 (16)

## 2. Fair Table III (`scripts/stats.py` -> `tables/table3_fair.tex`)

| class | scheme | ours+RR [95% CI] | greedy+RR | p | W/T/L % | ours+RR+LS [95% CI] | greedy+RR+LS | p (LS) | W/T/L % (LS) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ER | Degree | 7.64 [5.89, 9.55] | 11.55 | 0.0002 | 70/10/19 | 5.24 [3.93, 6.64] | 8.10 | 0.0045 | 56/19/24 |
| ER | Inverse | 5.85 [4.64, 7.12] | 5.25 | 0.6016 | 32/27/41 | 3.01 [2.05, 4.07] | 3.46 | 0.6918 | 29/46/25 |
| ER | Random | 4.77 [3.49, 6.21] | 4.54 | 0.2383 | 30/33/37 | 2.74 [1.86, 3.67] | 2.64 | 0.2848 | 22/47/30 |
| ER | Mixed | 6.93 [5.36, 8.64] | 8.70 | 0.0267 | 55/13/32 | 4.28 [3.18, 5.47] | 4.40 | 0.2918 | 43/28/29 |
| BA | Degree | 6.58 [4.84, 8.43] | 12.56 | <1e-4 | 67/12/21 | 5.61 [4.00, 7.37] | 7.71 | 0.0673 | 54/17/29 |
| BA | Inverse | 3.04 [2.04, 4.14] | 2.15 | 0.0211 | 19/53/28 | 1.53 [0.86, 2.31] | 1.38 | 0.9169 | 15/74/11 |
| BA | Random | 4.25 [2.98, 5.58] | 4.17 | 0.1240 | 30/34/36 | 2.31 [1.45, 3.21] | 1.53 | 0.0952 | 16/51/32 |
| BA | Mixed | 4.82 [3.64, 6.07] | 5.60 | 0.4773 | 40/24/37 | 2.30 [1.50, 3.22] | 3.37 | 0.1064 | 36/46/18 |
| Chordal | Degree | 4.26 [2.96, 5.63] | 7.30 | 0.0182 | 46/33/20 | 2.78 [1.77, 3.88] | 4.51 | 0.0488 | 40/42/18 |
| Chordal | Inverse | 1.05 [0.58, 1.57] | 1.07 | 0.2485 | 12/70/18 | 0.22 [0.05, 0.45] | 0.14 | 0.2367 | 2/88/10 |
| Chordal | Random | 1.41 [0.86, 2.05] | 2.01 | 0.9655 | 22/55/22 | 0.38 [0.14, 0.69] | 0.41 | 0.9063 | 6/84/10 |
| Chordal | Mixed | 2.13 [1.35, 3.03] | 4.36 | 0.7208 | 30/33/38 | 0.61 [0.23, 1.08] | 1.24 | 0.5937 | 11/82/8 |
| Bipartite | Degree | 9.15 [7.71, 10.77] | 8.77 | 0.2549 | 51/8/41 | 3.33 [2.31, 4.47] | 3.45 | 0.4889 | 36/33/31 |
| Bipartite | Inverse | 4.19 [3.09, 5.44] | 2.77 | 0.0147 | 17/35/48 | 0.91 [0.43, 1.46] | 0.47 | 0.1728 | 8/76/16 |
| Bipartite | Random | 2.80 [1.89, 3.81] | 2.32 | 0.0772 | 24/42/34 | 0.84 [0.44, 1.30] | 0.87 | 0.5256 | 14/69/16 |
| Bipartite | Mixed | 6.13 [4.47, 8.09] | 6.96 | 0.3379 | 40/26/35 | 1.60 [0.99, 2.29] | 2.03 | 0.7457 | 24/50/26 |
| All | all | 4.62 [4.25, 4.99] | 5.51 | 0.0092 | 36/32/32 | 2.29 [2.04, 2.56] | 2.76 | 0.0174 | 25/54/21 |

## 3. Fair exact cross-check, Table V format (`scripts/rev_greedy_ls_exact.py`, `scripts/rev_table_exact.py` -> `tables/exact_fair_n1000_degree.tex`, `exact_fair_final.tex`, `exact_fair_first.tex`)

greedy + RR + LS added on the same 400 + 240 graphs x 5 weightings (2,000 + 1,200 new solutions). Ours = model seeds 0 and 1.

| class | ours+RR | greedy+RR | ours+RR+LS | greedy+RR+LS | ours+RR better | ours+RR+LS better | p (LS, all n) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Trees | 0.39 | 0.80 | 0.15 | 0.43 | 74% | 56% | <1e-4 |
| Unit interval | 11.92 | 15.63 | 6.98 | 7.75 | 92% | 78% | <1e-4 |
| Convex bip. | 8.96 | 11.35 | 2.66 | 2.67 | 79% | 55% | 0.0054 |
| 2-trees | 4.55 | 6.52 | 3.30 | 4.43 | 86% | 72% | <1e-4 |


## 4. Stronger greedy baselines (`wtdsws/baselines/greedy_plus.py`, `scripts/rev_eval_grid.py` -> `raw_grid_rev_baselines.csv`; `tables/ours_vs_baselines.tex`)

Tests: `tests/test_greedy_plus.py` (200 random graphs, every variant valid).

| ours | baseline | subset | inst | ours gap % | base gap % | diff (pp) | 95% CI diff | Wilcoxon p | W/T/L % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ours_rr | greedy_rr | pooled | 624 | 4.62 | 5.51 | -0.89 | [-1.36, -0.44] | 0.0092 | 35.9/32.2/31.9 |
| ours_rr | greedy_rr | degree | 148 | 6.85 | 9.94 | -3.09 | [-4.41, -1.85] | <1e-4 | 58.1/16.2/25.7 |
| ours_rr | greedy_rr | inverse | 159 | 3.52 | 2.79 | +0.73 | [0.27, 1.18] | 0.0020 | 20.0/46.3/33.7 |
| ours_rr | greedy_rr | random | 160 | 3.31 | 3.26 | +0.04 | [-0.77, 0.77] | 0.0273 | 26.5/41.1/32.4 |
| ours_rr | greedy_rr | mixed | 157 | 4.97 | 6.36 | -1.39 | [-2.30, -0.55] | 0.0215 | 40.5/23.9/35.5 |
| ours_rr | greedy_tiebreak_rr | pooled | 624 | 4.62 | 5.57 | -0.96 | [-1.43, -0.51] | 0.0033 | 36.1/32.2/31.7 |
| ours_rr | greedy_tiebreak_rr | degree | 148 | 6.85 | 9.94 | -3.09 | [-4.41, -1.85] | <1e-4 | 58.1/16.2/25.7 |
| ours_rr | greedy_tiebreak_rr | inverse | 159 | 3.52 | 2.89 | +0.63 | [0.18, 1.09] | 0.0039 | 20.3/45.8/34.0 |
| ours_rr | greedy_tiebreak_rr | random | 160 | 3.31 | 3.42 | -0.11 | [-0.96, 0.64] | 0.1033 | 26.9/41.2/31.9 |
| ours_rr | greedy_tiebreak_rr | mixed | 157 | 4.97 | 6.38 | -1.41 | [-2.32, -0.57] | 0.0188 | 40.9/24.2/34.9 |
| ours_rr | greedy_lookahead_rr | pooled | 624 | 4.62 | 9.98 | -5.36 | [-6.32, -4.43] | <1e-4 | 49.6/26.5/23.8 |
| ours_rr | greedy_lookahead_rr | degree | 148 | 6.85 | 16.21 | -9.36 | [-11.75, -7.24] | <1e-4 | 74.3/12.0/13.6 |
| ours_rr | greedy_lookahead_rr | inverse | 159 | 3.52 | 3.72 | -0.21 | [-0.89, 0.41] | 0.4643 | 25.2/43.6/31.2 |
| ours_rr | greedy_lookahead_rr | random | 160 | 3.31 | 5.76 | -2.46 | [-4.06, -1.10] | 0.0188 | 37.6/34.8/27.6 |
| ours_rr | greedy_lookahead_rr | mixed | 157 | 4.97 | 14.72 | -9.76 | [-12.23, -7.50] | <1e-4 | 63.4/14.4/22.2 |
| ours_rr | lp_rounding_rr | pooled | 624 | 4.62 | 5.89 | -1.27 | [-1.94, -0.63] | 0.3389 | 33.9/28.1/38.0 |
| ours_rr | lp_rounding_rr | degree | 148 | 6.85 | 12.53 | -5.68 | [-7.50, -3.92] | <1e-4 | 58.6/14.6/26.8 |
| ours_rr | lp_rounding_rr | inverse | 159 | 3.52 | 3.00 | +0.52 | [-0.35, 1.31] | 0.0013 | 18.5/44.2/37.4 |
| ours_rr | lp_rounding_rr | random | 160 | 3.31 | 1.77 | +1.54 | [0.90, 2.21] | <1e-4 | 16.0/34.1/49.9 |
| ours_rr | lp_rounding_rr | mixed | 157 | 4.97 | 6.76 | -1.80 | [-3.25, -0.38] | 0.0263 | 44.3/18.5/37.2 |
| ours_rr | greedy_random_restart_rr | pooled | 624 | 4.62 | 1.95 | +2.67 | [2.31, 3.03] | <1e-4 | 10.8/34.4/54.7 |
| ours_rr | greedy_random_restart_rr | degree | 148 | 6.85 | 1.62 | +5.24 | [4.38, 6.13] | <1e-4 | 8.9/21.9/69.2 |
| ours_rr | greedy_random_restart_rr | inverse | 159 | 3.52 | 1.22 | +2.29 | [1.80, 2.79] | <1e-4 | 5.8/47.2/47.0 |
| ours_rr | greedy_random_restart_rr | random | 160 | 3.31 | 2.06 | +1.25 | [0.46, 1.92] | <1e-4 | 12.5/40.4/47.1 |
| ours_rr | greedy_random_restart_rr | mixed | 157 | 4.97 | 2.88 | +2.08 | [1.44, 2.73] | <1e-4 | 16.1/27.3/56.7 |
| ours_rr | greedy_rr_ls | pooled | 624 | 4.62 | 2.76 | +1.86 | [1.49, 2.23] | <1e-4 | 17.4/31.7/50.9 |
| ours_rr | greedy_rr_ls | degree | 148 | 6.85 | 5.84 | +1.01 | [-0.04, 2.11] | 0.0510 | 35.1/18.1/46.8 |
| ours_rr | greedy_rr_ls | inverse | 159 | 3.52 | 1.35 | +2.17 | [1.68, 2.67] | <1e-4 | 7.5/45.9/46.5 |
| ours_rr | greedy_rr_ls | random | 160 | 3.31 | 1.36 | +1.94 | [1.40, 2.51] | <1e-4 | 10.6/36.5/52.9 |
| ours_rr | greedy_rr_ls | mixed | 157 | 4.97 | 2.73 | +2.24 | [1.48, 3.05] | <1e-4 | 17.6/25.1/57.3 |
| ours_rr_ls | greedy_rr_ls | pooled | 624 | 2.29 | 2.76 | -0.47 | [-0.75, -0.20] | 0.0174 | 25.4/54.0/20.7 |
| ours_rr_ls | greedy_rr_ls | degree | 148 | 4.18 | 5.84 | -1.66 | [-2.43, -0.92] | <1e-4 | 46.2/28.2/25.5 |
| ours_rr_ls | greedy_rr_ls | inverse | 159 | 1.41 | 1.35 | +0.06 | [-0.25, 0.35] | 0.4398 | 13.6/71.1/15.3 |
| ours_rr_ls | greedy_rr_ls | random | 160 | 1.56 | 1.36 | +0.20 | [-0.35, 0.67] | 0.0510 | 14.9/62.8/22.4 |
| ours_rr_ls | greedy_rr_ls | mixed | 157 | 2.16 | 2.73 | -0.57 | [-1.07, -0.08] | 0.0520 | 28.3/52.0/19.7 |
| ours_rr_ls | greedy_tiebreak_rr_ls | pooled | 624 | 2.29 | 2.81 | -0.52 | [-0.81, -0.25] | 0.0070 | 25.5/53.9/20.6 |
| ours_rr_ls | greedy_tiebreak_rr_ls | degree | 148 | 4.18 | 5.84 | -1.66 | [-2.43, -0.92] | <1e-4 | 46.2/28.2/25.5 |
| ours_rr_ls | greedy_tiebreak_rr_ls | inverse | 159 | 1.41 | 1.43 | -0.02 | [-0.34, 0.28] | 0.8081 | 14.3/70.1/15.6 |
| ours_rr_ls | greedy_tiebreak_rr_ls | random | 160 | 1.56 | 1.52 | +0.05 | [-0.58, 0.59] | 0.0898 | 14.6/63.9/21.5 |
| ours_rr_ls | greedy_tiebreak_rr_ls | mixed | 157 | 2.16 | 2.69 | -0.53 | [-1.02, -0.05] | 0.0437 | 28.4/51.6/20.0 |
| ours_rr_ls | greedy_lookahead_rr_ls | pooled | 624 | 2.29 | 3.05 | -0.75 | [-1.08, -0.44] | <1e-4 | 28.9/51.0/20.1 |
| ours_rr_ls | greedy_lookahead_rr_ls | degree | 148 | 4.18 | 5.75 | -1.57 | [-2.60, -0.64] | 0.0002 | 46.5/25.3/28.2 |
| ours_rr_ls | greedy_lookahead_rr_ls | inverse | 159 | 1.41 | 1.66 | -0.25 | [-0.73, 0.19] | 0.7498 | 17.1/66.7/16.2 |
| ours_rr_ls | greedy_lookahead_rr_ls | random | 160 | 1.56 | 1.91 | -0.35 | [-0.86, 0.08] | 0.5380 | 21.4/60.5/18.1 |
| ours_rr_ls | greedy_lookahead_rr_ls | mixed | 157 | 2.16 | 3.06 | -0.90 | [-1.53, -0.30] | 0.0066 | 32.1/49.6/18.3 |
| ours_rr_ls | lp_rounding_rr_ls | pooled | 624 | 2.29 | 1.93 | +0.36 | [0.10, 0.63] | <1e-4 | 18.9/52.9/28.2 |
| ours_rr_ls | lp_rounding_rr_ls | degree | 148 | 4.18 | 4.37 | -0.19 | [-1.03, 0.62] | 0.8614 | 34.9/27.8/37.3 |
| ours_rr_ls | lp_rounding_rr_ls | inverse | 159 | 1.41 | 0.83 | +0.58 | [0.20, 0.95] | 0.0002 | 10.9/67.0/22.0 |
| ours_rr_ls | lp_rounding_rr_ls | random | 160 | 1.56 | 0.89 | +0.67 | [0.33, 1.03] | <1e-4 | 10.0/63.5/26.5 |
| ours_rr_ls | lp_rounding_rr_ls | mixed | 157 | 2.16 | 1.83 | +0.33 | [-0.12, 0.77] | 0.0440 | 20.9/51.5/27.6 |

## 5. Equal-time comparison (`scripts/rev_equal_time.py`, `scripts/rev_report.py` -> `tables/equal_time.tex`, `tables/equal_time_tests.tex`, `figures/equal_time_quality_vs_time.png|pdf`)

| method | budget (s) | inst | gap % | 95% CI | mean time (s) | candidates | solution found |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ours: rollouts+RR | 0.16 | 62 | 2.26 | [1.66, 2.89] | 0.236 | 10 | 310/310 |
| ours: rollouts+RR | 1.0 | 62 | 1.08 | [0.74, 1.44] | 1.079 | 66 | 310/310 |
| ours: rollouts+RR | 5.0 | 62 | 0.72 | [0.45, 1.01] | 5.077 | 336 | 310/310 |
| ours: rollouts+RR+LS | 0.16 | 62 | 1.45 | [0.94, 2.00] | 0.238 | 9 | 310/310 |
| ours: rollouts+RR+LS | 1.0 | 62 | 0.40 | [0.20, 0.64] | 1.074 | 55 | 310/310 |
| ours: rollouts+RR+LS | 5.0 | 62 | 0.26 | [0.12, 0.42] | 5.068 | 270 | 310/310 |
| greedy restarts+RR | 0.16 | 62 | 1.34 | [0.90, 1.82] | 0.160 | 259 | 62/62 |
| greedy restarts+RR | 1.0 | 62 | 1.22 | [0.81, 1.68] | 1.001 | 1592 | 62/62 |
| greedy restarts+RR | 5.0 | 62 | 1.12 | [0.74, 1.54] | 5.000 | 7886 | 62/62 |
| greedy restarts+RR+LS | 0.16 | 62 | 0.65 | [0.33, 1.02] | 0.160 | 56 | 62/62 |
| greedy restarts+RR+LS | 1.0 | 62 | 0.35 | [0.15, 0.59] | 1.000 | 350 | 62/62 |
| greedy restarts+RR+LS | 5.0 | 62 | 0.28 | [0.09, 0.51] | 5.001 | 1753 | 62/62 |
| greedy+RR+LS | 0.16 | 62 | 1.78 | [1.08, 2.56] | 0.007 | 1 | 62/62 |
| greedy+RR+LS | 1.0 | 62 | 1.78 | [1.08, 2.56] | 0.007 | 1 | 62/62 |
| greedy+RR+LS | 5.0 | 62 | 1.78 | [1.08, 2.56] | 0.007 | 1 | 62/62 |
| LP rounding+RR+LS | 0.16 | 62 | 1.79 | [1.00, 2.67] | 0.039 | 1 | 62/62 |
| LP rounding+RR+LS | 1.0 | 62 | 1.79 | [1.00, 2.67] | 0.015 | 1 | 62/62 |
| LP rounding+RR+LS | 5.0 | 62 | 1.79 | [1.00, 2.67] | 0.013 | 1 | 62/62 |
| CP-SAT (8 workers) | 0.16 | 62 | 0.60 | [0.12, 1.25] | 0.118 | 1 | 62/62 |
| CP-SAT (8 workers) | 1.0 | 62 | 0.11 | [0.00, 0.27] | 0.127 | 1 | 62/62 |
| CP-SAT (8 workers) | 5.0 | 62 | 0.00 | [0.00, 0.00] | 0.234 | 1 | 62/62 |


| budget (s) | vs method | inst | Wilcoxon p | ours W/T/L % |
| --- | --- | --- | --- | --- |
| 0.16 | ours: rollouts+RR | 62 | <1e-4 | 66/15/19 |
| 0.16 | greedy restarts+RR | 62 | 0.3535 | 31/44/25 |
| 0.16 | greedy restarts+RR+LS | 62 | 0.0004 | 12/50/37 |
| 0.16 | greedy+RR+LS | 62 | 0.8780 | 29/48/22 |
| 0.16 | LP rounding+RR+LS | 62 | 0.4638 | 25/49/26 |
| 0.16 | CP-SAT (8 workers) | 62 | <1e-4 | 5/55/41 |
| 1.0 | ours: rollouts+RR | 62 | <1e-4 | 69/26/5 |
| 1.0 | greedy restarts+RR | 62 | <1e-4 | 48/47/5 |
| 1.0 | greedy restarts+RR+LS | 62 | 0.3913 | 14/70/16 |
| 1.0 | greedy+RR+LS | 62 | <1e-4 | 44/52/4 |
| 1.0 | LP rounding+RR+LS | 62 | 0.0036 | 31/61/8 |
| 1.0 | CP-SAT (8 workers) | 62 | 0.0002 | 3/75/23 |
| 5.0 | ours: rollouts+RR | 62 | <1e-4 | 60/33/6 |
| 5.0 | greedy restarts+RR | 62 | <1e-4 | 47/47/6 |
| 5.0 | greedy restarts+RR+LS | 62 | 0.7088 | 14/73/13 |
| 5.0 | greedy+RR+LS | 62 | <1e-4 | 45/52/3 |
| 5.0 | LP rounding+RR+LS | 62 | 0.0008 | 31/63/6 |
| 5.0 | CP-SAT (8 workers) | 62 | 0.0001 | 0/81/19 |


## 6. Alpha-sweep (`scripts/rev_alpha_sweep.py`, `scripts/rev_report.py` -> `tables/alpha_sweep.tex`, `figures/alpha_gap.png|pdf`, `figures/alpha_advantage.png|pdf`)

Model seeds: [0, 1, 2, 3, 4]; ILP not proven: 11 of 720 (excluded).
* Spearman(alpha, per-instance advantage greedy+RR - ours+RR), alpha pos: rho = 0.190, p = 0.0002
* Spearman(alpha, per-instance advantage greedy+RR - ours+RR), alpha neg: rho = 0.056, p = 0.2666
* Spearman(alpha, per-instance advantage greedy+RR - ours+RR), alpha all: rho = 0.234, p < 1e-4

| alpha | inst | ours+RR | greedy+RR | tie-break+RR | adv. vs greedy (pp) | p | W/T/L % | adv. vs tie-break | p (tb) | ours+RR+LS | greedy+RR+LS | p (LS) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| -1.00 | 80 | 4.20 | 3.27 | 3.05 | -0.93 | 0.0035 | 22/42/36 | -1.15 | 0.0006 | 1.57 | 1.80 | 0.8815 |
| -0.75 | 80 | 4.01 | 3.28 | 3.49 | -0.73 | 0.0374 | 20/45/35 | -0.52 | 0.0510 | 1.86 | 1.78 | 0.0463 |
| -0.50 | 80 | 4.43 | 3.26 | 3.48 | -1.17 | 0.0208 | 22/35/42 | -0.95 | 0.0342 | 2.14 | 1.83 | 0.0849 |
| -0.25 | 80 | 4.72 | 3.92 | 4.14 | -0.80 | 0.0278 | 20/28/51 | -0.58 | 0.0517 | 2.25 | 2.34 | 0.6248 |
| +0.00 | 80 | 4.80 | 4.50 | 6.33 | -0.30 | 0.7387 | 15/68/17 | +1.53 | 0.0292 | 3.17 | 2.69 | 0.0460 |
| +0.25 | 78 | 6.02 | 6.09 | 6.09 | +0.07 | 0.8647 | 37/16/46 | +0.07 | 0.8647 | 3.35 | 5.02 | 0.0054 |
| +0.50 | 77 | 5.62 | 7.08 | 7.08 | +1.46 | 0.0635 | 48/19/33 | +1.46 | 0.0635 | 4.05 | 4.94 | 0.1069 |
| +0.75 | 77 | 6.04 | 7.63 | 7.63 | +1.58 | 0.0061 | 50/19/31 | +1.58 | 0.0061 | 4.12 | 5.28 | 0.0051 |
| +1.00 | 77 | 6.83 | 9.79 | 9.79 | +2.96 | 0.0003 | 55/20/25 | +2.96 | 0.0003 | 4.89 | 6.13 | 0.0272 |

## 7. Coverage fixes (`scripts/rev_coverage.py`)

Unproven grid ILPs: 16 (`coverage_lb.json`, `tables/unproven_lb.tex`).

| method | unproven inst | mean gap to LB % | max gap to LB % | all 640: gap to opt or LB % |
| --- | --- | --- | --- | --- |
| ours_rr | 16 | 34.70 | 45.40 | 5.37 |
| ours_rr_ls | 16 | 32.23 | 43.21 | 3.04 |
| greedy_rr | 16 | 37.58 | 51.42 | 6.31 |
| greedy_rr_ls | 16 | 34.54 | 50.92 | 3.56 |
| greedy_tiebreak_rr | 16 | 38.17 | 51.42 | 6.39 |
| greedy_lookahead_rr | 16 | 41.79 | 58.20 | 10.77 |
| lp_rounding_rr | 16 | 49.76 | 74.23 | 6.99 |
| greedy_random_restart_rr | 16 | 28.79 | 39.38 | 2.62 |
| ilp_incumbent_30s | 16 | 22.31 | 35.02 | 0.56 |


Blocking rule (base paper, no training; `blocking_ablation.json`): 3200 episodes, model seeds [0, 1, 2, 3, 4], valid TDS in 0.0%; mean fraction of vertices covered when no legal action remained 0.731.

## 8. Training and inference time (`scripts/rev_report.py` -> `report.json`, `tables/training_time.tex`, `tables/inference_time.tex`; `scripts/rev_inference_time.py`)

| seed | train min | steps | best step | val gap % (+RR) |
| --- | --- | --- | --- | --- |
| 0 | 20.5 | 78000 | 50000 | 4.69 |
| 1 | 25.9 | 78000 | 68000 | 5.18 |
| 2 | 32.0 | 78000 | 28000 | 4.50 |
| 3 | 23.8 | 78000 | 48000 | 5.43 |
| 4 | 23.5 | 78000 | 78000 | 4.38 |


| n | GPU (s) | CPU only (s) | CPU/GPU | same W |
| --- | --- | --- | --- | --- |
| 20 | 0.0215 | 0.0090 | 0.42 | 12/12 |
| 50 | 0.0681 | 0.0288 | 0.42 | 12/12 |
| 100 | 0.0952 | 0.0486 | 0.51 | 12/12 |
| 200 | 0.1691 | 0.1055 | 0.62 | 12/12 |
| 500 | 0.4224 | 0.4209 | 1.00 | 12/12 |


CPU and GPU solutions with identical W(S) per n: {'20': 12, '50': 12, '100': 12, '200': 12, '500': 12}
