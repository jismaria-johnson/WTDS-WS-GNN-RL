"""Findings computed from the evaluation data (no hand-typed numbers). Writes results/findings.json and prints them.

    python scripts/findings.py && python scripts/analyze.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np

from analyze import CLS_LABEL, SCH_LABEL, build, load, stat
from wtdsws.graph_classes import CLASSES
from wtdsws.weights import SCHEMES

grid = load("grid")
opt, ilp_rt, val = build(grid)
NS = sorted({k[2] for k in opt})
small = lambda k: k[2] in NS
pp = lambda x: 100 * x

# 1. scheme difficulty for greedy (+RR), classes and n pooled
sch = {s: (stat(val, opt, "greedy_rr", lambda k, s=s: k[4] == s and small(k))[0],
           stat(val, opt, "ours_rr", lambda k, s=s: k[4] == s and small(k))[0]) for s in SCHEMES}
order = sorted(SCHEMES, key=lambda s: -sch[s][0])
hard = order[0]
f1 = (f"<b>Hardest scheme for greedy:</b> {SCH_LABEL[hard].lower()}. Weighted greedy + RR has a mean gap of "
      f"{pp(sch[hard][0]):.2f}% under {SCH_LABEL[hard].lower()} weights (classes and n = {', '.join(map(str, NS))} pooled), "
      f"against " + ", ".join(f"{pp(sch[s][0]):.2f}% ({SCH_LABEL[s].lower()})" for s in order[1:]) +
      f". Ours + RR: " + ", ".join(f"{pp(sch[s][1]):.2f}% ({SCH_LABEL[s].lower()})" for s in order) + ".")

# 2. where ours gains most / loses (class x scheme cells)
cells = []
for c in CLASSES:
    for s in SCHEMES:
        sel = lambda k, c=c, s=s: k[0] == c and k[4] == s and small(k)
        o, osd = stat(val, opt, "ours_rr", sel)
        g, _ = stat(val, opt, "greedy_rr", sel)
        cells.append((g - o, c, s, o, osd, g))
cells.sort(reverse=True)
wins = [x for x in cells if x[0] > 0]
loss = [x for x in cells if x[0] < 0]
top = cells[:3]
f2 = (f"<b>Where ours gains most:</b> " + "; ".join(
    f"{CLS_LABEL[c]} / {SCH_LABEL[s].lower()} ({pp(o):.2f} ± {pp(sd):.2f}% vs greedy + RR {pp(g):.2f}%)"
    for _, c, s, o, sd, g in top) + f". Ours + RR has the lower mean gap in {len(wins)} of 16 class × scheme cells.")
f3 = (f"<b>Where greedy is better:</b> in {len(loss)} of 16 cells weighted greedy + RR has the lower mean gap, under "
      + ", ".join(sorted({SCH_LABEL[s].lower() for _, c, s, *_ in loss})) + " weights: " + "; ".join(
          f"{CLS_LABEL[c]} / {SCH_LABEL[s].lower()} ({pp(o):.2f}% vs {pp(g):.2f}%)" for _, c, s, o, sd, g in
          sorted(loss)) + ("." if not (none := [SCH_LABEL[x].lower() for x in SCHEMES
                                                if x not in {s for _, c, s, *_ in loss}])
                           else f". No {', '.join(none)}-scheme cell is among them.")) \
    if loss else "<b>Where greedy is better:</b> nowhere in the 16 cells."

# 3. local search, zero-shot vs fine-tuned, runtime
o_rr, o_rr_sd = stat(val, opt, "ours_rr", small)
o_ls, o_ls_sd = stat(val, opt, "ours_rr_ls", small)
g_rr, _ = stat(val, opt, "greedy_rr", small)
g_ls, _ = stat(val, opt, "greedy_rr_ls", small)
z_rr, z_sd = stat(val, opt, "zero_rr", small)
f4 = (f"<b>Fine-tuning helped on the test grid:</b> mean gap of the copied wtds model + RR {pp(z_rr):.2f} ± {pp(z_sd):.2f}% "
      f"→ fine-tuned ours + RR {pp(o_rr):.2f} ± {pp(o_rr_sd):.2f}% (weighted greedy + RR {pp(g_rr):.2f}%). With the same "
      f"1 s local search, ours + RR + LS reaches {pp(o_ls):.2f} ± {pp(o_ls_sd):.2f}% and greedy + RR + LS {pp(g_ls):.2f}%.")
n_max = max(NS)
rt = lambda m, n: np.mean([rt for s in (("0", "1") if m.startswith(("ours", "zero")) else ("",))
                           for k, (_, rt) in val[(m, s)].items() if k[2] == n])
ilp_n = np.mean([v for k, v in ilp_rt.items() if k[2] == n_max])
np_ = sum(1 for k, v in opt.items() if v is None)
np_er = sum(1 for k, v in opt.items() if v is None and k[0] == "er")
f5 = (f"<b>Runtime:</b> at n = {n_max} ours + RR takes {rt('ours_rr', n_max):.3f} s per instance on the GPU, the ILP "
      f"{ilp_n:.2f} s on average (30 s limit), weighted greedy + RR {rt('greedy_rr', n_max) * 1000:.1f} ms. "
      f"{np_} of {len(opt)} ILP runs were not proven within 30 s ({np_er} of them on ER graphs); they are excluded "
      f"from the gaps.")

# 4. n = 500 (no ILP): ours vs greedy + RR in W(S)
big = sorted({k[2] for d in val.values() for k in d} - set(NS))
f6 = ""
if big:
    n = big[0]
    rel = {}
    for c in CLASSES:
        for s in SCHEMES:
            sel = lambda k, c=c, s=s: k[0] == c and k[4] == s and k[2] == n
            o = np.mean([np.mean([W for k, (W, _) in val[("ours_rr", sd)].items() if sel(k)]) for sd in ("0", "1")])
            g = np.mean([W for k, (W, _) in val[("greedy_rr", "")].items() if sel(k)])
            rel[(c, s)] = 100 * (o - g) / g
    deg = [rel[(c, "degree")] for c in CLASSES]
    cheaper = sum(1 for v in rel.values() if v < 0)
    f6 = (f"<b>n = {n} (no ILP):</b> ours + RR is cheaper than weighted greedy + RR in {cheaper} of 16 class × scheme "
          f"cells. Under degree weights the difference in mean W(S) is " + ", ".join(
              f"{v:+.2f}% ({CLS_LABEL[c]})" for v, c in zip(deg, CLASSES)) + " (negative = ours cheaper).")

out = [f1, f2, f3, f4, f5] + ([f6] if f6 else [])
(ROOT / "results/findings.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
for f in out:
    print("-", f.replace("<b>", "").replace("</b>", ""))
