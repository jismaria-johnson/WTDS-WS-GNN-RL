import json
from pathlib import Path

import streamlit as st

from components import solver
from components.theme import notice, stat_tiles, table

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results"
FIG = RES / "figures"


def pending(what):
    notice(f"<b>Experiment pending.</b> <span class='mono'>{what}</span> has not been produced yet; it appears here "
           f"automatically once the experiment has run.", "warn")


def pct(x):
    return "—" if x is None else f"{100 * x:.2f}%"


def model_card():
    info = solver.model_info()
    st.markdown('<div class="section-title">Model card (validation, 160 graphs)</div>', unsafe_allow_html=True)
    if not info:
        pending("checkpoints/model_info.json")
        return
    tiles = []
    for s in info["seeds"]:
        tiles.append((f"Seed {s['seed']} · val gap +RR", pct(s["shipped_val_gap_rr"]),
                      f"zero-shot {pct(s['zero_shot_val_gap_rr'])}; fine-tuned best {pct(s['finetuned_best_val_gap_rr'])}"
                      f" after {s['finetune_steps']:,} steps; shipped: {s['shipped']}", "teal"))
    tiles.append(("Weighted greedy + RR", pct(info["val_gap_weighted_greedy_rr"]), "same 160 validation graphs", ""))
    stat_tiles(tiles)
    st.markdown(f'<div class="caption">{info.get("validation_note", "")}</div>', unsafe_allow_html=True)


def render():
    st.markdown('<div class="section-title" style="font-size:1.6rem;">Results</div>', unsafe_allow_html=True)
    st.markdown('<div class="caption">Every number on this page is read from files written by runs in this repository '
                '(<span class="mono">checkpoints/model_info.json</span>, <span class="mono">results/summary.json</span>). '
                'Every reported solution passed <span class="mono">is_total_dominating_set</span> and its weight was '
                'recomputed from the solution.</div>', unsafe_allow_html=True)
    model_card()
    st.markdown('<div class="section-title">Ground-truth evaluation (class × scheme × n)</div>', unsafe_allow_html=True)
    sp = RES / "summary.json"
    if not sp.exists():
        pending("results/summary.json")
        return
    s = json.loads(sp.read_text(encoding="utf-8"))
    v = s["validity"]
    stat_tiles([("Solutions checked", f"{v['solutions_checked']:,}", "all methods, all cells, both seeds", ""),
                ("Valid", f"{100 * v['valid_fraction']:.0f}%", "is_total_dominating_set", "teal"),
                ("ILP proven", f"{s['ilp']['proven']} / {s['ilp']['run']}",
                 f"{s['ilp']['not_proven']} not proven in {s['ilp']['time_limit']:g} s (excluded from gaps)", ""),
                ("Graphs", f"{s['graphs']:,}", s.get("grid_note", ""), "")])
    figs = [("fig_heatmap_ours_rr.png", "Gap of ours + RR, class × scheme"),
            ("fig_heatmap_greedy_rr.png", "Gap of weighted greedy + RR, class × scheme"),
            ("fig_gap_vs_n.png", "Gap vs n per scheme"), ("fig_runtime_vs_n.png", "Runtime vs n"),
            ("fig_weight_hist.png", "Weight distribution per scheme")]
    c = st.columns(2)
    for i, (f, cap) in enumerate(figs):
        with c[i % 2]:
            if (FIG / f).exists():
                st.image(str(FIG / f), caption=cap, width="stretch")
            else:
                pending(f"results/figures/{f}")
    for k in s["tables"]:
        st.markdown(f'<div class="section-title">{s["captions"][k]["title"]}</div>', unsafe_allow_html=True)
        rows = s["tables"][k]
        if rows:
            cols = list(rows[0].keys())
            table(rows, cols, numeric=[c for c in cols if c not in ("class", "scheme", "method", "subclass")])
        st.markdown(f'<div class="caption">{s["captions"][k]["text"]}</div>', unsafe_allow_html=True)
    if s.get("findings"):
        st.markdown('<div class="section-title">Findings (from the data above)</div>', unsafe_allow_html=True)
        st.markdown("".join(f"<p>{f}</p>" for f in s["findings"]), unsafe_allow_html=True)
