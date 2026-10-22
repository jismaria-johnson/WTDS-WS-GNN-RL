import csv
import json
from pathlib import Path

import streamlit as st

from components.theme import notice, stat_tiles, table

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results"
FIG = RES / "figures" / "exact_vs_ours"
SCHEMES = {"Degree": "degree", "Inverse degree": "inverse", "Random": "random", "Mixed": "mixed"}
CLASS_LABEL = {"tree": "Trees", "unit_interval": "Unit-interval", "convex_bip": "Convex bipartite", "two_tree": "2-trees"}
CAPTION2 = ("With weighted graphs, several optimal sets can have the same minimum weight but different sizes, so |S| "
            "can differ from the exact method even when ours is optimal. See Plot 3.")
CAPTION2_FINAL = "With weighted graphs, equal-cost sets can differ in size; total weight (Plot 3) is the objective."
PAPER = RES / "figures" / "paper"
BOX = ('Text box in each panel: "Ours beats Greedy in X% of graphs" = share of (graph, checkpoint) pairs, pooled over '
       'all n, where ours + RR is strictly better than greedy + RR (|S| in Plot 1, total weight W(S) in Plots 2, 3, 5); '
       'ties are shown separately; X is rounded down.')
PARAM_LABEL = {"random_tree": "random labelled tree", "ba_m1": "BA m = 1", "deg3": "average degree 3",
               "deg6": "average degree 6", "deg10": "average degree 10", "k2": "2-tree generator (as generated)"}


def pending(what):
    notice(f"<b>Pending.</b> <span class='mono'>{what}</span> has not been produced yet; it appears here "
           f"automatically once <span class='mono'>scripts/exact_vs_ours.py</span> and "
           f"<span class='mono'>scripts/plot_exact_vs_ours.py</span> have run.", "warn")


def figure(name, caption, folder=FIG):
    p = folder / f"{name}.png"
    if p.exists():
        st.image(str(p), caption=caption, width="stretch")
    else:
        pending(str(p.relative_to(ROOT)).replace("\\", "/"))


def render_earlier():
    st.markdown('<div class="caption">Four graph classes on which the optimum can be computed exactly: trees (linear '
                'tree DP), unit-interval graphs (our exact interval DP), convex bipartite graphs and 2-trees '
                '(CP-SAT ILP, 60 s, proven optimal only). n = 20 to 1000, 10 graphs per class and n, 5 weightings. '
                'Every set passed <span class="mono">is_total_dominating_set</span> and its weight was recomputed.'
                '</div>', unsafe_allow_html=True)
    sp = RES / "exact_vs_ours_summary.json"
    if sp.exists():
        s = json.loads(sp.read_text(encoding="utf-8"))
        v = s["validation"]
        stat_tiles([("Exact-method validation", f"{v['total_mismatches']} mismatches",
                     f"{v['total_instances']:,} instances vs brute force and proven ILP", "teal"),
                    ("Solutions checked", f"{s['solutions_checked']:,}", "all valid TDS", ""),
                    ("Exact not proven", str(len(s["exact_not_proven"])), "hollow markers on the plots", ""),
                    ("Graphs", str(s["graphs"]), "x 5 weightings", "")])
    else:
        pending("results/exact_vs_ours_summary.json")
    st.markdown('<div class="section-title">Plot 1 · Cardinality |S| with unit weights</div>', unsafe_allow_html=True)
    st.markdown('<div class="caption">All weights are 1, so |S| is the objective: this is the fair cardinality '
                'comparison.</div>', unsafe_allow_html=True)
    figure("plot1_cardinality_unit", "Exact vs ours vs greedy + RR, unit weights")
    label = st.radio("Weight scheme for Plots 2 and 3", list(SCHEMES), index=2, horizontal=True, key="exact_scheme")
    sch = SCHEMES[label]
    st.markdown('<div class="section-title">Plot 2 · Cardinality |S| on weighted graphs</div>', unsafe_allow_html=True)
    figure(f"plot2_cardinality_{sch}", f"{label} weights")
    notice(CAPTION2)
    st.markdown('<div class="section-title">Plot 3 · Total weight W(S) on weighted graphs</div>', unsafe_allow_html=True)
    st.markdown('<div class="caption">W(S) is the objective our system minimises.</div>', unsafe_allow_html=True)
    figure(f"plot3_weight_{sch}", f"{label} weights")
    st.markdown('<div class="section-title">Table</div>', unsafe_allow_html=True)
    tp = RES / "tables" / "exact_vs_ours.csv"
    if not tp.exists():
        pending("results/tables/exact_vs_ours.csv")
        return
    with open(tp, newline="") as f:
        rows = list(csv.DictReader(f))
    show = st.radio("Scheme", ["Unit"] + list(SCHEMES), index=0, horizontal=True, key="exact_table_scheme")
    key = "unit" if show == "Unit" else SCHEMES[show]
    cols = {"class": "cls", "n": "n", "|S| exact": "S_exact", "|S| ours": "S_ours", "W exact": "W_exact",
            "W ours": "W_ours", "gap W %": "gap_W_pct", "% optimal (W)": "opt_W_pct", "t exact (s)": "t_exact_s",
            "t ours (s)": "t_ours_s"}

    def fmt(c, x):
        if c == "cls":
            return CLASS_LABEL.get(x, x)
        if c == "n":
            return x
        v = float(x)
        return f"{v:.0f}" if c == "opt_W_pct" else f"{v:.3f}" if c.startswith("t_") else f"{v:.2f}" \
            if c == "gap_W_pct" else f"{v:.1f}"

    out = [{k: fmt(c, r[c]) for k, c in cols.items()} for r in rows if r["scheme"] == key]
    table(out, list(cols), numeric=[k for k in cols if k != "class"])
    st.markdown('<div class="caption">Means over the graphs whose exact optimum is proven; ours = mean of the two '
                'model seeds. gap W = (W ours − W exact) / W exact. Full table (with ours + RR + 1 s local search and '
                'greedy + RR): <span class="mono">results/tables/exact_vs_ours.csv</span>.</div>',
                unsafe_allow_html=True)


def section(title, caption=None):
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)
    if caption:
        st.markdown(f'<div class="caption">{caption}</div>', unsafe_allow_html=True)


def chosen_table():
    p = RES / "chosen_settings.json"
    if not p.exists():
        pending("results/chosen_settings.json")
        return None
    ch = json.loads(p.read_text(encoding="utf-8"))["chosen"]
    rows = [{"class": CLASS_LABEL[c], "parameter": PARAM_LABEL.get(v["param"], v["param"]),
             "n grid": f"{v['grid']}: {v['ns'][0]}–{v['ns'][-1]}", "weighting (Plots 2–5)": v["weighting"],
             "separation, all weightings (pp)": f"{v['separation']:+.2f}",
             "separation, chosen weighting (pp)": f"{v['separation_by_weighting'][v['weighting']]:+.2f}",
             "exact proven %": f"{v['proven_pct']:.0f}"} for c, v in ch.items()]
    num = ["separation, all weightings (pp)", "separation, chosen weighting (pp)", "exact proven %"]
    table(rows, list(rows[0]), numeric=num)
    return ch


def final_table():
    tp = RES / "tables" / "exact_vs_ours_final.csv"
    if not tp.exists():
        pending("results/tables/exact_vs_ours_final.csv")
        return
    with open(tp, newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["plotted"] == "True"]
    show = st.radio("Weights", ["Unit", "Chosen weighting"], index=1, horizontal=True, key="final_table_scheme")
    rows = [r for r in rows if (r["scheme"] == "unit") == (show == "Unit")]
    cols = {"class": "cls", "weights": "scheme", "n": "n", "W exact": "W_exact", "W ours": "W_ours",
            "W greedy": "W_greedy", "gap ours %": "gap_ours_pct", "gap greedy %": "gap_greedy_pct",
            "ours optimal %": "opt_ours_pct", "greedy optimal %": "opt_greedy_pct", "|S| exact": "S_exact",
            "|S| ours": "S_ours", "|S| greedy": "S_greedy"}

    def fmt(c, x):
        if c == "cls":
            return CLASS_LABEL.get(x, x)
        if c in ("scheme", "n"):
            return x
        v = float(x)
        return f"{v:.0f}" if c.startswith("opt_") else f"{v:.2f}" if "gap" in c else f"{v:.1f}"

    out = [{k: fmt(c, r[c]) for k, c in cols.items()} for r in rows]
    table(out, list(cols), numeric=[k for k in cols if k not in ("class", "weights")])
    st.markdown('<div class="caption">Means over graphs whose exact optimum is proven; ours = mean of the two '
                'checkpoints. Full table with ours + RR + 1 s local search, ratios and runtimes: '
                '<span class="mono">results/tables/exact_vs_ours_final.csv</span>.</div>', unsafe_allow_html=True)


def render():
    st.markdown('<div class="section-title" style="font-size:1.6rem;">Exact vs ours</div>', unsafe_allow_html=True)
    st.markdown('<div class="caption">Four graph classes on which the optimum can be computed exactly: trees (tree '
                'DP), unit-interval graphs (our exact interval DP), convex bipartite graphs and 2-trees (CP-SAT ILP, '
                '60 s, proven optimal only). Procedure: a <b>pilot sweep</b> on separate pilot seeds chose, per class, '
                'the n grid, the class parameter and the weighting where greedy + RR and ours differ most (by a fixed '
                'rule); the <b>final plots</b> use 10 fresh test-range graphs per n, never the pilot graphs. Every '
                'set passed <span class="mono">is_total_dominating_set</span> and its weight was recomputed.</div>',
                unsafe_allow_html=True)
    sp = RES / "exact_vs_ours_final_summary.json"
    if sp.exists():
        s = json.loads(sp.read_text(encoding="utf-8"))
        stat_tiles([("Fresh test graphs", str(s["graphs"]), "x 5 weightings", ""),
                    ("Solutions checked", f"{s['solutions_checked']:,}", "all valid TDS", "teal"),
                    ("Exact not proven", str(len(s["exact_not_proven"])), "hollow markers on the plots", ""),
                    ("DP vs ILP cross-check", f"{s['ilp_crosscheck']['agree']}/{s['ilp_crosscheck']['runs']}",
                     "agree (trees, unit-interval)", "")])
    else:
        pending("results/exact_vs_ours_final_summary.json")
    section("Pilot sweep and chosen settings",
            "SEPARATION = greedy gap % − ours gap % (percentage points, vs the exact optimum), averaged over the "
            "pilot instances of a setting. Positive = ours is closer to the optimum. Full pilot results: "
            "<span class='mono'>results/pilot_sweep*.csv</span>; the rule is in "
            "<span class='mono'>docs/decisions.md</span>.")
    ch = chosen_table()
    figure("pilot_separation_heatmap", "Pilot sweep: separation per class x weighting (pilot seeds only)")
    st.markdown(f'<div class="caption">{BOX} Weighted plots use degree weights, the weighting the pilot chose for '
                'every class.</div>', unsafe_allow_html=True)
    section("Plot 1 · Cardinality |S| with unit weights", "All weights are 1, so |S| is the objective.")
    figure("plot1_cardinality_unit", "Exact vs ours vs greedy + RR, unit weights", PAPER)
    section("Plot 2 · Cardinality |S| on weighted graphs", "Degree weights.")
    figure("plot2_cardinality_weighted", "Degree weights", PAPER)
    notice(CAPTION2_FINAL)
    section("Plot 3 · Total weight W(S)", "W(S) is the objective our system minimises (degree weights).")
    figure("plot3_total_weight", "Degree weights", PAPER)
    section("Plot 5 · Approximation ratio", "W(S) / W_exact; the exact method is the line at 1.0 (degree weights).")
    figure("plot5_ratio", "Approximation ratio", PAPER)
    section("Paper figure", "Rows: Plot 1, Plot 3, Plot 5; columns: the four classes.")
    figure("paper_main", "results/figures/paper/paper_main.pdf (vector)", PAPER)
    section("Table")
    if ch is not None:
        final_table()
    with st.expander("Stress test (adversarial for greedy) — not part of the main results"):
        st.markdown('<div class="caption">Spider trees with degree weights: a few expensive hubs and many cheap '
                    'legs, a structure found (on pilot seeds) to be hard for weighted greedy. Shown separately and '
                    'never mixed into the plots above.</div>', unsafe_allow_html=True)
        figure("stress_test", "Stress test (adversarial for greedy)")
    with st.expander("Earlier study: n = 20–1000, all four weightings (first exact-vs-ours run)"):
        render_earlier()
