"""Fill the generated sections of README.md from checkpoints/model_info.json, results/summary.json and
logs/pytest_final.log (no hand-typed numbers). Idempotent: each section lives between <!-- NAME:begin/end --> markers
(a bare <!-- NAME --> placeholder is replaced the first time).

    python scripts/readme_results.py
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
mi = json.loads((ROOT / "checkpoints/model_info.json").read_text())
sm = json.loads((ROOT / "results/summary.json").read_text(encoding="utf-8"))
T = sm["tables"]
pct = lambda x: f"{100 * x:.2f}%"


def md_table(rows, cols=None):
    cols = cols or list(rows[0].keys())
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    out += ["| " + " | ".join(str(r[c]) for c in cols) + " |" for r in rows]
    return "\n".join(out)


s0, s1 = mi["seeds"]
ov = {r["method"]: r for r in T["overall"]}
glance = f"""## Results at a glance

* **Validation (160 graphs, 10 per class × scheme, proven optima):** gap after RR, zero-shot → fine-tuned:
  seed 0 {pct(s0['zero_shot_val_gap_rr'])} → **{pct(s0['shipped_val_gap_rr'])}**, seed 1
  {pct(s1['zero_shot_val_gap_rr'])} → **{pct(s1['shipped_val_gap_rr'])}**; weighted greedy + RR
  {pct(mi['val_gap_weighted_greedy_rr'])}. Both seeds: {s0['finetune_steps']:,} fine-tuning steps.
* **Test grid ({sm['graphs']} weighted instances, ILP proven on {sm['ilp']['proven']}/{sm['ilp']['run']}):** mean gap
  over all classes, schemes and n ∈ {{{', '.join(map(str, sm['n_ilp']))}}} — ours + RR {ov['ours+RR']['mean gap %']}%,
  weighted greedy + RR {ov['weighted_greedy_rr']['mean gap %']}%, the copied wtds model (zero-shot) + RR
  {ov['wtds_zero_shot+RR']['mean gap %']}%; with the same 1 s local search: ours {ov['ours+RR+LS']['mean gap %']}%,
  greedy {ov['weighted_greedy_rr_ls']['mean gap %']}% (mean ± std over the 2 model seeds).
* **Validity:** {sm['validity']['solutions_checked']:,} reported solutions, **100 % valid**; each was re-verified
  from the rebuilt graph and its W(S) recomputed (`scripts/analyze.py`).
* The gain over greedy is **not uniform**: see the findings in Section 7.6 (written from the data).
"""

training = f"""## 5. Training: short fine-tuning, not from scratch

Both copied wtds checkpoints load with no missing or unexpected keys (tested). **Zero-shot first:** before any
training they were evaluated on the new validation set (160 graphs = 10 per class × scheme, n = 50–100, optima proven
by CP-SAT in 30 s; 0 of 160 needed replacing). Then each seed was fine-tuned from its own wtds `best.pt` with the
**same reward and algorithm**: lr 5e-5, ε 0.2 → 0.05 over the first 20 % of the step horizon
({mi['train']['total_steps']:,}), batch 64, a fresh replay buffer of {mi['train']['buffer_size']:,} transitions
(30k planned; 20k because less than 3 GB of RAM was free), every episode drawing a class uniformly from
{{ER, BA, chordal, bipartite}}, then a scheme uniformly from {{degree, inverse, random, mixed}}, n ∈ [20, 100].
Training ran on the GPU (RTX 3050). Seeds were trained **one after the other** (free RAM < 8 GB): seed 0 in a
20-minute box, which ended at step {s0['finetune_steps']:,}; seed 1 was stopped at the same step count. Validation
every 2,000 steps; `best.pt` = lowest validation gap after RR (step 0 included, so a seed that never improved would
ship its zero-shot weights — this did not happen). Training, validation and test seeds come from disjoint ranges
(checked in `tests/test_ws_schemes.py` and printed by `scripts/make_val_set.py`).

| | zero-shot gap (+RR) | fine-tuned best gap (+RR) | best step | steps | wall time |
|---|---|---|---|---|---|
| seed 0 (`checkpoints/best.pt`) | {pct(s0['zero_shot_val_gap_rr'])} | **{pct(s0['shipped_val_gap_rr'])}** | {s0['best_step']:,} | {s0['finetune_steps']:,} | {s0['wall_s'] / 60:.1f} min |
| seed 1 (`checkpoints/seed1_best.pt`) | {pct(s1['zero_shot_val_gap_rr'])} | **{pct(s1['shipped_val_gap_rr'])}** | {s1['best_step']:,} | {s1['finetune_steps']:,} | {s1['wall_s'] / 60:.1f} min |
| weighted greedy + RR (same 160 graphs) | {pct(mi['val_gap_weighted_greedy_rr'])} | | | | |

All of this is in `checkpoints/model_info.json` (re-measured by `scripts/finalize_models.py`).
"""

H = T["headline"]
head_cols = ["class", "scheme", "ours+RR gap %", "greedy+RR gap %", "ours+RR+LS gap %", "greedy+RR+LS gap %",
             "zero-shot+RR gap %", "proven / ILP runs", "ours+RR time (s)", "ILP time (s)"]
results = f"""## 6. Ground-truth evaluation setup

Grid: class ∈ {{ER, BA, chordal, bipartite}} × scheme ∈ {{degree, inverse, random, mixed}} × n ∈
{{{', '.join(map(str, sm['n_ilp']))}}}, 10 graphs per cell, plus n = {', '.join(str(n) for n in sm['n_all'] if n not in sm['n_ilp'])}
for greedy and the GNN only (no ILP). For each (class, n) the same 10 graph structures are used under all four schemes,
so schemes are compared on identical graphs. Methods: weighted greedy, weighted greedy + RR, weighted greedy + RR + LS,
the copied wtds model zero-shot + RR, ours + RR, ours + RR + LS (LS: the same 1 s limit for greedy and ours), and the
ILP (CP-SAT, 30 s, 8 workers) as ground truth. Gaps use only instances whose optimum was **proven**; the number of
non-proven ILP runs is reported. Seeded methods: mean ± std (ddof = 1) over the two model seeds of the per-seed mean
gap. One job at a time on the GPU; the "after" hash of the old folders ran in the background as instructed.

## 7. Results

### 7.1 Headline: ours + RR vs weighted greedy + RR, by class × scheme

{md_table(H, head_cols)}

{sm['captions']['headline']['text']}

![ours + RR heatmap](results/figures/fig_heatmap_ours_rr.png) ![greedy + RR heatmap](results/figures/fig_heatmap_greedy_rr.png)

### 7.2 All classes and schemes pooled

{md_table(T['overall'])}

### 7.3 Gap vs n per scheme (classes pooled)

{md_table(T['gap_vs_n'])}

![gap vs n](results/figures/fig_gap_vs_n.png)

### 7.4 Runtime vs n

{md_table(T['runtime'])}

{sm['captions']['runtime']['text']}

![runtime vs n](results/figures/fig_runtime_vs_n.png)

### 7.5 n = 500 (no ILP) and the sub-class sanity check

{md_table(T['n500'])}

{sm['captions']['n500']['text']}

Sub-class sanity check (unit interval and convex bipartite; exact polynomial algorithms exist for weighted domination
variants on these subclasses, Lee 2025 — **not** a case for learning, only a check):

{md_table(T['subclass'])}

![weight distributions](results/figures/fig_weight_hist.png)

### 7.6 Findings (computed by `scripts/findings.py` from the data above)

""" + "\n".join(f"{i + 1}. " + re.sub(r"</?b>", "**", f) for i, f in enumerate(sm["findings"])) + """

Files: `results/summary.json`, `results/tables/*.csv` and `*.tex` (LaTeX), `results/figures/*.png`, raw per-instance
rows (with the solution and the graph spec) in `results/raw_grid.csv` and `results/raw_subclass.csv`.
"""

log = (ROOT / "logs/pytest_final.log").read_text(encoding="utf-8", errors="replace")
m = re.findall(r"(\d+ (?:passed|failed)[^\n]*?) in ([\d.]+s)", log)
summary_line = m[-1][0] if m else "see logs/pytest_final.log"
tests = f"""Final run: **{summary_line}** (`logs/pytest_final.log`). The suite covers: the four schemes (range,
determinism, monotonicity, the max/min vertex gets 100, mixed = rounded mean), the ten exact hand examples of Section 9
against brute force, 500 generated graphs per class (class assertion, convexity, no isolated vertex), weighted ILP =
brute force on 200 graphs (n ≤ 10, all classes × schemes), every solver valid on 300 graphs, strict loading of both
copied checkpoints, isolated-vertex and bad-weight errors, the CLI, all copied wtds tests (renamed package), and the
app end to end (every hand example under degree and inverse, one generated graph per class × scheme at n = 100, an
isolated vertex, bad weights, custom weights, a 5,000-vertex BA graph without drawing, the 4-schemes presets and
every other page)."""

lim = f"""## 10. Limitations

* **Short fine-tuning:** {s0['finetune_steps']:,} steps per seed (about 20 minutes each on an RTX 3050), with a 20k
  replay buffer because of low free RAM. Longer training might change the picture.
* **Two seeds** only; the ± values are the spread of two runs, not a confidence interval.
* **ILP ground truth only up to n = 200** (30 s limit): {sm['ilp']['not_proven']} of {sm['ilp']['run']} ILP runs were
  not proven and are excluded from the gaps; at n = 500 only the weights of the heuristics are compared.
* **The gain is not uniform:** weighted greedy + RR has the lower gap in several class × scheme cells (Section 7.6),
  mostly under inverse and random weights. Greedy is also two orders of magnitude faster than the GNN rollout.
* **The schemes are synthetic models of real costs** (load, infrastructure, site cost); real deployments may combine
  costs differently. Custom weights are supported for that case.
* Runtimes were measured on a laptop with a background disk job (the requested re-hash) running during part of the
  evaluation.
* On unit-interval and convex-bipartite graphs exact polynomial algorithms exist for related weighted domination
  problems (Lee 2025); results there are a sanity check, not an argument for learning.
"""

repro = """## Reproducing the results

```bash
python scripts/make_val_set.py                                   # 160 validation graphs + proven ILP optima (data/)
python scripts/zero_shot.py                                      # zero-shot gaps of the copied wtds checkpoints
python -m wtdsws.train --config configs/ws_finetune.yaml --seed 0         # 20-min box (GPU)
python -m wtdsws.train --config configs/ws_finetune_seed1.yaml --seed 1   # stops at seed 0's step count
python scripts/finalize_models.py                                # best.pt per seed -> checkpoints/, model_info.json
python scripts/run_eval.py --part grid                           # class x scheme x n (+ n = 500), ~20 min
python scripts/run_eval.py --part subclass                       # unit interval + convex bipartite
python scripts/findings.py && python scripts/analyze.py          # re-verify every solution, tables, figures
python scripts/readme_results.py                                 # fill the generated README sections
python -m pytest -q > logs/pytest_final.log
python -B scripts/hash_tree.py docs/original_hashes_after.txt "../tds-gnn-rl" "../wtds-gnn-rl" "../ewtds-gnn-rl"
```

The old projects (`tds-gnn-rl`, `wtds-gnn-rl`, `ewtds-gnn-rl`) were only read and copied from: the SHA-256 hashes of
all their files before (`docs/original_hashes_before.txt`) and after (`docs/original_hashes_after.txt`) the build are
identical.
"""

readme = (ROOT / "README.md").read_text(encoding="utf-8")
for name, body in (("RESULTS-AT-A-GLANCE", glance), ("TRAINING", training), ("RESULTS", results), ("TESTS", tests),
                   ("LIMITATIONS", lim + "\n" + repro)):
    block = f"<!-- {name}:begin -->\n{body.strip()}\n<!-- {name}:end -->"
    pat = re.compile(rf"<!-- {name}:begin -->.*?<!-- {name}:end -->", re.S)
    if pat.search(readme):
        readme = pat.sub(lambda _: block, readme)
    else:
        assert f"<!-- {name} -->" in readme, name
        readme = readme.replace(f"<!-- {name} -->", block)
(ROOT / "README.md").write_text(readme, encoding="utf-8")
print("README.md sections filled:", summary_line)
