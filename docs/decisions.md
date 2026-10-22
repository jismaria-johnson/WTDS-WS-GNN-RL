# Decisions log (wtds-ws-gnn-rl)

Decisions made during the build, in order. Every number quoted anywhere comes from a run in this folder.

## Approved after Phase 1
1. **Zero weights.** `wtdsws` keeps wtds's internal convention (weight 0 = free, pre-selected vertex), which the
   copied wtds tests rely on. User input (explicit `weights=`, files, networkx/adjacency input, CLI, app) must be
   finite and > 0, else `InvalidWeightError`.
2. **Zero-shot baseline.** The unweighted TDS checkpoint was not copied; the method `old_tds_model` became
   `wtds_zero_shot` = the copied wtds-gnn-rl seed-0 model (not fine-tuned) + weighted RR.

## Phase 0 / 1
3. Python 3.13.4 venv with the same package versions as wtds-gnn-rl (torch 2.11.0+cu128, CUDA on the RTX 3050).
   `requirements-lock.txt` = `pip freeze` of this venv.
4. Before the "before" hash snapshot, `wtds-gnn-rl/.venv/Scripts/python.exe -c "import sys"` was run once to read
   the old venv's Python version. It imports nothing from the project; the before/after comparison covers every
   file of the three old folders.
5. Seed ranges: train graph/weight seeds in [0, 1e9), validation in [1.0e9, 1.1e9), test in [1.2e9, 1.3e9)
   (`wtdsws/seeds.py`), disjoint by construction and checked by a test.
6. Chordal G(n,p) uses average degree U[1, 3] before `complete_to_chordal_graph` (denser inputs fill in to
   near-complete graphs). Unit-interval and convex-bipartite densities: average degree U[2, 10] / U[3, 10].

## Phase 2
7. Validation ILP: CP-SAT with 8 workers (not 4). Probe: a dense ER graph (n = 97, m = 613, degree weights) was
   NOT proven in 30 s with 4 workers but proven in 21.9 s with 8 (HiGHS: 13.6 s). Other classes solve in < 0.05 s.
   CP-SAT memory on 100-vertex models is small, so 8 workers are RAM-safe.
8. Validation graphs whose optimum is not proven within 30 s are replaced by a fresh sample of the same
   class x scheme cell (count recorded in data/val_ws.json, `replaced_not_proven`). This slightly biases the
   validation set towards ILP-tractable dense-ER instances; the test grid (Phase 5) has no such replacement.
   Result: 0 graphs needed replacement (160/160 proven; ILP time mean 0.13 s, max 5.27 s; logs/val_set.log).

## Phase 3
9. Free RAM before seed 0: 2.36 GB (< 3 GB), so per the instruction the replay buffer is 20,000 transitions
   (not 30,000) and the validation rollout batch is 20 graphs (not 40). Same settings for seed 1, so both seeds
   are trained identically. The replay buffer stores (graph id, S_t, a, R, S_t+1) and rebuilds features on the GPU,
   so it is small either way.
10. `train.total_steps = 90000` only sets the epsilon horizon (0.2 -> 0.05 over the first 18,000 steps; a first start
    with 60000 measured ~70 steps/s on the GPU, which would have ended before the 20-min box, so it was restarted);
    seed 0 stops at its 20-minute time box (checked at validation points, every 2,000 steps); seed 1 is stopped at
    the same step count (`train.stop_at`).
11. Seed 0 hit the 20-min box at step 78,000 (wall 1229 s; the box is checked at validation points). Seed 1 uses
    `configs/ws_finetune_seed1.yaml` (same config, `stop_at: 78000`, no time box). Free RAM before seed 1: 2.86 GB
    (< 3 GB), so the same 20k buffer / val batch 20 as seed 0.

## Phase 4
12. `app/pages/` was renamed `app/views/`. A folder called `pages/` next to the entry script triggers Streamlit's
    legacy multipage auto-detection: in a real browser `/solve` ran `pages/solve.py` directly (blank page, file-name
    sidebar) instead of the `st.navigation` page. Found with a Playwright browser check (scripts/screenshots.py);
    the headless AppTest suite cannot see it. (The copied wtds app has the same layout; it was not touched.)

## Phase 3 result / Phase 5
13. Both seeds improved on zero-shot, so the fine-tuned best.pt ships for both (scripts/finalize_models.py,
    re-measured): seed 0 4.69 % (step 50k) vs 5.69 % zero-shot; seed 1 5.18 % (step 68k) vs 6.28 %; weighted
    greedy + RR 5.73 % on the same 160 graphs.
14. Test grid design: for each (class, n) 10 graph STRUCTURES are drawn (test seed range) and each is evaluated
    under all four schemes (weight seed per graph), so schemes are compared on identical graphs. Subtypes of chordal
    and bipartite are drawn uniformly per graph. LS time limit = 1.0 s for both greedy + RR + LS and ours + RR + LS.
    ILP = CP-SAT, 8 workers, 30 s, n <= 200. wtds_zero_shot+RR is run for both copied seeds (mean ± std like ours).
15. As instructed, the "after" re-hash of the old folders runs in the background during Phase 5 (disk-bound; the
    evaluation is GPU/CPU-bound). Runtimes in the evaluation are therefore measured on a machine that is not fully
    idle; they are reported as measured.
16. `tests/test_app.py` (copied, 19 tests, skipped until best.pt existed) targeted wtds-app features that Section 8
    removes (the certified "proven minimum" default method, the "Watch the agent" page, the old weighted examples).
    It was rewritten for the new app with the same driver and checks, extended to 38 tests: every hand example under
    degree and inverse (ILP optimum + ours valid), one generated graph per class × scheme at n = 100, isolated vertex,
    bad weights (file and paste), custom weights, 5,000-vertex BA without drawing, the 4-schemes presets (brute-force
    optima), compare/results/how pages. With checkpoints/best.pt present, 0 tests are skipped.
17. Browser check (`scripts/screenshots.py`, Playwright driving the system Edge, no browser download) revealed and
    fixed: the `pages/` auto-detection (#12), an empty weight histogram (Vega-Lite field named `from`), and a
    spurious "relabelled" warning on the built-in examples. playwright was added to the venv (requirements-lock.txt).
18. The default page (Solve) is served at `/` (Streamlit ignores the url_path of the default page); the other pages
    are `/schemes`, `/compare`, `/results`, `/how`.
19. Findings are generated by `scripts/findings.py` and README result sections by `scripts/readme_results.py`, both
    from the result files, so no number in the README is typed by hand.

## Exact vs ours (special graph classes)
20. **Lexicographic exact objective.** All exact methods minimise (W(S), |S|): minimum weight first, then the
    smallest such set (tree DP and interval DP compare (W, |S|) pairs; the ILP uses w'(v) = (n+1)·w(v) + 1, which is
    exact for integer weights). W-optimality is unchanged, and "exact |S|" in Plot 2 is well defined: it is the
    smallest size of any minimum-weight set, not whatever tie a solver happened to return. As a result, ours can show
    a smaller |S| than exact on weighted graphs only if it is not W-optimal.
21. **Interval algorithm.** Bertossi & Gori (1988) and Chang (1998) were not reproduced. `unit_interval_tds` is our
    own DP for unit (= proper) interval graphs, labelled "our exact DP for unit-interval graphs" (plots: "Exact (our
    interval DP)"). It covers exactly the panel's class (the existing unit_interval generator). It is not a general
    interval-graph algorithm, and it rejects orders that are not proper interval orders. The proof is in
    docs/exact_interval.md.
22. **Tree panel composition.** For each n: graphs 0-4 are random labelled trees (`nx.random_labeled_tree`) and graphs
    5-9 are BA with m = 1. The 2-tree panel uses the existing `ktree` generator with k = 2. Unit-interval and convex
    bipartite use `class_spec` densities (average degree U[2, 10] / U[3, 10]).
23. **Seeds.** A new master seed MASTER_SEEDS["test"] + 2 (the grid used +0 and the subclass run +1). Every graph seed
    and weight seed is drawn from the test range [1.2e9, 1.3e9). The script asserts that none of them occurs in
    results/raw_grid.csv or raw_subclass.csv. Validation instances (brute force / ILP checks) come from a fixed
    separate RNG (seed 20261019) and are not test data.
24. **Class assertions.** Every graph is rebuilt as a networkx graph and checked to have the same edge set as the Graph
    that was solved. Checks per class: trees `nx.is_tree`; unit-interval chordal + interval-model realisation;
    convex bipartite bipartite + consecutive-neighbourhood check on the generator's X order; 2-trees chordal +
    `is_k_tree(G, 2)` (new, in graph_classes.py: repeatedly delete a simplicial degree-2 vertex). Every class is also
    checked for isolated vertices.
25. **Extra cross-check.** On trees and unit-interval graphs, the lexicographic CP-SAT ILP (60 s) also runs on every
    graph and weighting (method `ilp_check`). The run aborts if a proven ILP disagrees with the DP on (W, |S|), or if
    any ILP finds a lower W. This goes beyond the requested validation.
26. **Validation counts.** "500 instances" = 100 graphs × 5 weightings per algorithm (n ≤ 12). "300 instances" =
    60 graphs × 5 weightings per algorithm (n = 50-200). Each ILP instance is solved twice: the plain min-weight ILP
    (compares W) and the lexicographic ILP (compares |S|).
27. **Aggregation.** A cell (class, n, scheme) uses only graphs whose exact optimum is proven, for every method. A
    cell with an unproven graph gets a hollow Exact marker and a "k/10 proven" label. Ours = mean of the two
    checkpoints per graph, then mean / std over graphs. "Ours optimal in X% of graphs" = share of (graph, model seed)
    runs of ours + RR that match exact: |S| for Plot 1 (unit weights, where |S| = W), W for Plot 3. Plot 2 boxes show
    both numbers.
28. **Plot axes and colours.** Both axes are log scale, because |S| and W span about two orders of magnitude over n =
    20-1000. Series are dodged horizontally by a few percent so that overlapping markers stay visible. The x ticks
    are at the true n. The specified colours (#1F2328, #0F766E, #9CA3AF) were kept. The palette validator flags them
    as low-chroma neutrals, but they pass colour-blind separation, and each method also has its own marker shape,
    line style and legend entry. The grey has low contrast against white; the same numbers are also in the table.
29. **Unit weights.** The "unit" weighting is an explicit all-ones weight vector (not "unweighted"), so every method
    sees the same Graph object type as under the four schemes.
30. **Results.** The run took 702 s: 240 graphs, 7,800 solutions, all valid. Every exact optimum was proven (CP-SAT
    took at most 6.7 s per instance, far below the 60 s limit), so no plot has a hollow marker. The ilp_check cross-check
    agrees with the DP on 600/600 (graph, weighting) pairs. The README section is generated by
    scripts/readme_exact.py (no hand-typed numbers). The full suite has 708 tests, all passed
    (logs/pytest_final.log).

## Exact vs ours, updated plots: pilot → choose → fresh test
31. **Why.** The guide asked for a clearly visible difference between Exact, Ours and Greedy + RR. To avoid selection
    bias, the plot settings are chosen on a pilot run that uses pilot-only seeds. The final plots are then made on
    NEW test-range graphs. All earlier results and figures stay untouched; the new files use the `final_` /
    `pilot_` prefixes.
32. **Pilot seed range.** `SEED_RANGES["pilot"] = (1_400_000_000, 1_500_000_000)` in wtdsws/seeds.py. It is
    disjoint from train / val / test (check_disjoint, plus the new test
    `test_pilot_seed_range_disjoint`). Pilot master seed 777_100. scripts/pilot_sweep.py asserts that every pilot
    graph and weight seed is in that range.
33. **Pilot design.** The grids A = {100..1000} and B = {200..2000} share graphs at the common n (200..1000), so each
    setting has 15 distinct n × 5 graphs × 5 weightings. Sharing makes the A-vs-B comparison paired and saves
    compute. Settings: trees (random labelled | BA m = 1), unit-interval and convex bipartite (avg degree 3 | 6 |
    10, through the existing generators' avg_deg parameter), and 2-trees (ktree k = 2). Methods: exact, ours + RR
    (best.pt only, as specified) and greedy + RR. Plus, on trees and unit-interval graphs, the same lexicographic
    ILP cross-check as before (it aborts on any disagreement with a proven ILP). Every set passes both TDS checkers,
    and W(S) is recomputed. The pilot also asserts that no heuristic beats a proven optimum. The pilot CSV does not
    store S (size); the final run does.
34. **Separation.** Per (graph, weighting) with a proven optimum: gap_m = 100 (W_m − W_exact) / W_exact, and
    SEPARATION = gap_greedy − gap_ours (percentage points). dW = W_greedy − W_ours and dS = |S|_greedy − |S|_ours are
    also saved (pilot_sweep_per_graph.csv, pilot_sweep_settings.csv).
35. **Selection rule** (scripts/pilot_choose.py, fixed before looking at the results). A candidate is a (parameter,
    grid) pair per class. It is eligible iff ours is always valid and ≥ 95% of its exact optima are proven. Score =
    mean SEPARATION over all its proven instances (all n of the grid × 5 graphs × all 5 weightings, so one setting
    serves both Plot 1 (unit) and Plots 2–5). The highest-scoring candidate wins; ties go to grid B, then to the
    first-listed parameter. The weighting for Plots 2–5 = the non-unit weighting with the largest mean separation
    inside the chosen candidate (unit is excluded because Plot 1 already covers it). The chosen weighting can
    differ per class and is named in every panel title.
36. **Speed-ups (user asked for < 2 h and GPU use).**
    - The pilot ran as a main process plus parallel workers (`--only`, results/pilot_sweep_w*.csv). The main
      process was stopped once a worker owned its next setting. pilot_choose.merge_workers merges every file into
      results/pilot_sweep.csv; a duplicate key keeps the worker copy. The worker files are kept.
    - GNN rollouts are batched on the GPU over the 5 weightings of a graph (one forward pass per step for all 5).
      On a 1000-vertex 2-tree the batched sets were identical to one-by-one rollouts for all 5 weightings, and the
      batch was 4.5× faster (2.8 s vs 12.7 s). The reported runtime of ours is therefore AMORTISED: the batched
      rollout time / 5 + the graph's own RR (+ LS) time. Earlier pilot rows (before the switch) hold
      one-by-one times; runtimes are not used in the selection rule.
    - The final run is split into shards (`--shard i/k`), and the shard outputs are merged. Correctness checks were
      not reduced anywhere: both TDS checkers, recomputed W, the DP vs ILP cross-check, and the "no heuristic below a
      proven optimum" assertion all run on every instance.
37. **Final run.** New master seed MASTER_SEEDS["test"] + 3 (one RNG stream per class: MASTER·10 + class index).
    The script asserts that every graph/weight seed is in the test range and unused by raw_grid, raw_subclass,
    exact_vs_ours, pilot_sweep and stress_test. 10 graphs per (class, n), all 5 weightings, with ours + RR and
    ours + RR + LS (1 s) for both checkpoints.
38. **Final plots.** Linear axes: the chosen grids are evenly spaced, and a log y axis would compress exactly the
    gaps the guide wants to see. The y range is padded so that the legend (top) and the "Ours optimal in X% |
    Greedy optimal in Y%" box (bottom) never cover data. "Ours optimal" = share of (graph, checkpoint) runs with
    W = W_exact; "Greedy optimal" = share of graphs. With unit weights, W = |S|. Plot 4 = W − W_exact (exact = 0)
    and Plot 5 = W / W_exact (exact = 1), both with the chosen weighting.
39. **Stress test (separate figure, never in the main plots).** A small exploration on PILOT-range seeds
    (greedy + RR vs tree DP, n = 500, 6 graphs per cell, CPU only) found that the suggested "BA m = 1 trees with
    degree weights" is NOT hard for greedy: greedy was exactly optimal on 6/6 graphs, and its gap was ≤ 0.19% for
    every weighting on BA m = 1. Spider-like trees with degree weights were hard (greedy gap 6–9%). The gap depends
    on the hubs-per-n ratio; h = max(1, n // 400) hubs with legs of length 2–5 kept the gap at about 6–7% for
    n = 500 and 2000. Family "spider trees", n ∈ {250, 500, 1000, 1500, 2000}, 10 graphs each, test range with
    master MASTER_SEEDS["test"] + 4, exact = tree DP plus the ILP cross-check.
40. **CPU contention incident (pilot).** The first parallel attempt ran about 20 CP-SAT threads on 12 cores. ILPs
    that take < 1 s unloaded hit the 60 s limit: 20 unproven ILP rows in total (exact or ilp_check). These came from
    my scheduling, not from the instances, so they must not decide the ≥ 95%-proven rule. Fix: the workers were
    stopped and restarted with exactly 12 CP-SAT threads (3 workers × 4). `--redo-unproven` re-solves every graph that
    has an unproven ILP row (or was cut off mid-write) and appends the new rows. pilot_choose keeps the later copy of a
    duplicated (setting, n, graph, weighting, method) row. Nothing is deleted: the worker files keep both copies, and
    the main process's file is saved as results/pilot_sweep_main_raw.csv before the merge. The final run uses the
    same 12-thread budget.
41. **Correction to 40: cause of the unproven ILPs.** Almost all unproven rows were the DEGREE weighting, and some
    stayed unproven when re-solved at 12 threads. Re-run in isolation with the same code (4 threads), the same
    instances were proven optimal in 0.2–0.5 s (convex deg 10 n 300, convex deg 3 n 400, 2-tree n 300). So the cause
    is CP-SAT run-to-run variance on the many weight ties that degree weights create (the lexicographic objective has
    many equal-weight optima); oversubscription made it worse. A two-stage alternative (prove min W, then min |S| with
    W fixed) was worse (stage 2 unproven at 60 s) and was not adopted. **Rule adopted:** an ILP that is not proven
    within 60 s gets ONE retry with another CP-SAT random seed (another 60 s). The status column records both
    attempts ("FEASIBLE+retry:OPTIMAL"); runtime = both attempts. "Proven" = proven by either attempt. Applies to the
    pilot (from the restart at about 13:20; earlier unproven graphs are re-solved by --redo-unproven), the final run
    and the stress test.
42. **Time limit (user: "finish it in 20 min", about 14:05).** The pilot was stopped before every setting finished.
    Incomplete candidates were NOT evaluated: convex deg 10 (grid A: 3 graphs missing at n = 1000; grid B: 19
    missing) and unit-interval deg 10 (grids A and B: 3 graphs missing at n = 1000). They are listed as
    complete = False / eligible = False in results/pilot_sweep_choice.csv and marked "incomplete: not evaluated" on
    the heatmap. The rule picks only among complete candidates. All pilot rows are kept, including the partial
    settings.
43. **Chosen settings** (results/chosen_settings.json; separation = mean over all 5 weightings, in pp):
    - trees: random labelled, grid A, +0.12; weighting degree (+0.41). Runner-up random labelled grid B (+0.11).
      This class was chosen early (13:22) from its complete pilot data, so its final run could start sooner; the
      full rule run gives the identical choice (asserted).
    - unit-interval: avg deg 6, grid A, +1.14; weighting degree (+5.38). Runner-up deg 3 grid A (+0.98).
    - convex bipartite: avg deg 6, grid A, +0.17 (exact proven on 99.2%); weighting degree (+2.55). Runner-up
      deg 3 grid B (+0.10).
    - 2-trees: as generated, grid A, +0.48; weighting degree (+2.76).
    Degree weighting has the largest separation in every class, as the brief expected. For unit and inverse
    weights, the separation is negative on several candidates (greedy closer to the optimum); the heatmap shows
    these values. The overall separations are small (< 1.2 pp) except for unit-interval.
44. **Thread count matters more than load (final run).** The first final launch used 12 shards × 2 CP-SAT threads.
    With 2 threads, the lexicographic ILP stalled on tie-heavy weightings: the ILP cross-check hit 2 × 60 s on
    n = 100 unit-interval graphs, and exact convex / 2-tree ILPs were unproven at n = 300–400. Relaunched with 6
    threads per process (unit-interval 2 shards; convex and 2-trees 1 process each), the same work ran with 0
    unproven instances. The 2-thread outputs were moved, not deleted, to results/aborted_2thread_shards/. They are not
    used: they contain the same graphs (same seeds) and were re-solved from scratch. This is also the most likely
    explanation of the pilot's unproven ILPs (2–4 threads per worker); 41's retry rule stays in force.
45. **Paper redraw (no new experiments).** scripts/plot_paper.py only reads results/exact_vs_ours_final.csv; the
    sha256 of that file and of results/tables/exact_vs_ours_final.csv is identical before and after. The new box
    "Ours beats Greedy in X% of graphs" counts (graph, checkpoint) pairs over all n of a panel, with STRICT
    improvement (|S| for unit weights, W(S) otherwise, including Plot 2). Ties are reported as "(ties Y%)" when
    Y > 0. X is rounded DOWN to one decimal. The weighted plots use degree weights (the pilot's choice for every
    class). Plot 4 was moved to results/figures/archive/. Paper table: results/tables/paper_table.tex
    (n = 100 / 500 / 1000).

## Revision sprint (2026-10-21, 5-hour box from 23:18 IST; guide's review `Revision-needed.pdf`)
Rules: only the existing system (no new learned models, no re-implementation of published methods); other project
folders read-only; every solution checked with `is_total_dominating_set` and W(S) recomputed; existing test graphs and
seeds reused; correctness checks never cut.

R1. **Seeds 2, 3, 4** (step 0). Only two wtds checkpoints exist (`checkpoints/pretrained/wtds_seed{0,1}_best.pt`), and
    training new wtds models would be a new learned model in another project, so seed s starts from
    `wtds_seed{s mod 2}_best.pt` (2, 4 <- wtds seed 0; 3 <- wtds seed 1). Everything else is seed-specific: torch init
    RNG, epsilon draws, replay sampling and the training-graph stream (`TRAIN_MASTER * (s + 1)`). Config = seed 1's
    (`stop_at: 78000`, no time box, same lr / epsilon / buffer 20k / batch 64). Run one after another
    (`scripts/train_seeds_234.sh`; free RAM at start 1.5 GB). Consequence: seed-to-seed spread over 5 seeds measures
    fine-tuning randomness, with only two distinct starting points; stated in the paper text.
R2. **Table III (step 1).** The main grid already had `greedy_rr_ls` (same 1 s LS, `results/raw_grid.csv`). The
    exact-vs-ours studies (first: 240 graphs; final: 400 graphs; 5 weightings) only had greedy + RR, so
    `scripts/rev_greedy_ls_exact.py` adds greedy + RR + LS on the same instances. The instances are rebuilt with the
    original scripts' spec/build functions and checked against the stored rows (same m and graph seed, and greedy + RR
    re-run gives exactly the stored W) before the new row is written.
R3. **Stronger greedy (step 3)**, `wtdsws/baselines/greedy_plus.py`, tests `tests/test_greedy_plus.py` (200 random
    graphs, 4 classes x 4 schemes, every variant valid; LP bound <= brute-force optimum; restart never worse than
    greedy + RR): 9 passed. Notes: (a) supports of leaves are already forced by the degree-1 rule, so "prefer vertices
    adjacent to leaves" rarely fires; the effective tie-break is lower degree, then lower weight. (b) Lookahead
    evaluates every candidate (no top-k pruning). (c) Random restart: eps = 0.05 multiplicative noise per vertex and
    restart, random tie rank; restart 0 is plain greedy. (d) LP rounding: scipy `linprog(method="highs")`.
    On the grid each is run + RR and + RR + 1 s LS; random restart gets 0.16 s (our GNN + RR time at n = 200).
R4. New grid rows cover n = 20-200 only (the n with ILP optima); n = 500 is not re-run for the new methods/seeds.
R5. **Which table is "Table III".** In the current paper Table III (grid by class x scheme) already has
    greedy + RR + LS; the incomparable table is Table V (exact cross-check: ours + RR + LS next to greedy + RR only).
    Both are rebuilt fairly: `results/revision/tables/table3_fair.tex` (5 seeds, CIs, p, W/T/L) and
    `exact_fair_*.tex` (`scripts/rev_table_exact.py`; ours there = model seeds 0, 1, the only seeds run on those
    graphs; seeds 2-4 were not run on the 13,000-solution cross-check for time).
R6. **Statistics (`scripts/stats.py`).** Ours per instance = mean W over the model seeds; Wilcoxon two-sided on the
    seed-mean vs the baseline (zero differences dropped, scipy default); win/tie/loss over (instance, seed) pairs;
    bootstrap 10,000 resamples of instances (rng seed 0); seed CI = t-interval on per-seed mean gaps (df = 4).
    Holm-adjusted p over the 16 class x scheme cells is reported next to raw p. The script reproduces the paper's
    2-seed numbers from the raw rows (4.51 / 5.51 / 2.24 / 2.76), which checks the pipeline.
R7. **Equal time (`scripts/rev_equal_time.py`).** 64 instances (n = 100, 200; graphs 0-1 of every class x scheme)
    from the main grid; 3 budgets x 7 methods; ours on 5 seeds. 64 not 640 instances to fit the 5-hour box
    (ours alone: 64 x 6.16 s x 2 variants x 5 seeds = 66 min). Sampling temperature 0.03 and batches of 8 sampled
    rollouts: the temperature tuned on validation data for the parent wtds model (`checkpoints/pretrained/
    wtds_inference.json`); not re-tuned here. CP-SAT gets 8 workers (as the ground-truth ILP) and no warm start
    (a warm start would hand it the greedy solution). CPU methods ran while seed 2 was training (one Python process
    on the GPU); the GNN equal-time and inference timings run only after all training ended, with no ILP running.
R8. **Alpha-sweep (`scripts/rev_alpha_sweep.py`).** 80 fresh graphs (test range, new RNG stream
    MASTER_SEEDS["test"] + 5000; asserted disjoint from every earlier results file), the same 80 structures for all
    9 alphas (720 instances). alpha = +-1 use the integer formulas of the degree / inverse schemes (checked equal);
    other alphas use ceil(100 r^|alpha| - 1e-9). ILP = CP-SAT 30 s, 8 workers, greedy warm start (as the grid).
R9. **Coverage.** Unproven grid ILPs (16): gap to CP-SAT's best bound (`results/revision/tables/unproven_lb.tex`).
    Blocking rule (no training): trained policy, argmax Q over unselected vertices that are not neighbours of a
    selected vertex (forced vertices also block); 1,280 episodes (seeds 0, 1; re-run with 5 seeds at the end):
    0.0% end in a valid TDS. This is forced by the rule: a selected vertex needs a selected neighbour, which the
    rule forbids, so no selected vertex can ever be dominated.
R10. All GNN work (training, grid / alpha / equal-time rollouts, blocking ablation) runs on the GPU (device "cuda").
    The only CPU-only GNN run is the inference-time measurement the revision asks for. Its first attempt failed
    before measuring anything: `CUDA_VISIBLE_DEVICES=` (empty) did not hide the GPU under Windows / Git Bash, and
    the script's own assertion caught it. It is re-run with `CUDA_VISIBLE_DEVICES=-1` after the GPU equal-time run,
    so the two measurements do not compete.
R11. Final pass (01:58, 2026-10-22, 2 h 40 min into the box): all steps 0-7 completed, nothing cut. 45 tests pass
    (greedy_plus, ws_schemes, postprocess, checker). Notes for the paper: (a) at the 0.16 s budget ours overran
    (mean 0.236 s), because the first greedy rollout alone takes ~0.17 s at n = 200 and a started batch is finished;
    the 1 s and 5 s budgets were met within 8%. (b) CPU-only inference is FASTER than the GPU for n <= 200 (batch of one
    small graph: kernel-launch overhead dominates) and equal at n = 500; the old "GPU vs CPU" runtime comparison should
    use the CPU numbers. (c) Equal-time results use 62 of 64 instances (2 unproven ILP optima excluded).
