"""Same graph, 4 schemes: how the chosen set changes when only the weights change."""
import html

import numpy as np
import streamlit as st

from components import solver
from components.graph_input import example, with_scheme
from components.network import MAX_DRAW, solution_html
from components.state import current, graph_pills, isolated_card, set_current, weight_seed
from components.theme import card_title, notice, table

from wtdsws.weights import FORMULA, MEANING, SCHEMES

PRESETS = {"Two triangles: degree vs inverse": "Two triangles + bridge", "House: degree vs inverse": "House"}
NAMES = {"degree": "Degree", "inverse": "Inverse degree", "random": "Random", "mixed": "Mixed"}


def hubs(g):
    """The highest-degree vertices: degree >= the 90th percentile (at least the max-degree vertices)."""
    d = g.degrees
    return set(np.flatnonzero(d >= max(np.percentile(d, 90), 1)).tolist())


def run_all(lg, model, seed):
    out = {}
    for sch in SCHEMES:
        w = with_scheme(lg, sch, seed)
        g = w.graph
        r = solver.run_rl(g, model, rr=True, ls_time=0.0)
        opt = solver.optimum_for(w) if g.n <= 150 else None
        H = hubs(g)
        out[sch] = dict(res=r, opt=opt, hubs_in_S=len(H & set(r["S"])), hubs=len(H),
                        mean_deg_S=float(g.degrees[r["S"]].mean()), mean_deg=float(g.degrees.mean()))
    return out


def render():
    lg = current()
    left, right = st.columns([1, 3], gap="large")
    with left:
        with st.container(border=True):
            card_title("Same graph, 4 schemes")
            st.markdown('<div class="caption">Runs GNN + RL (+ RR) on the current graph under each weight scheme '
                        '(and the exact ILP when n ≤ 150). Only the weights change; the graph is the same.</div>',
                        unsafe_allow_html=True)
            preset = None
            for label, ex in PRESETS.items():
                if st.button(label, width="stretch", key=f"preset_{ex}"):
                    preset = ex
            models = solver.available_models()
            model = st.radio("Model", list(models) or ["—"], key="s4_model")
            seed = st.number_input("Weight seed (random, mixed)", 0, 10**6, weight_seed(), key="s4_seed")
            go = st.button("Run on all 4 schemes", type="primary", width="stretch",
                           disabled=len(lg.graph.isolated_vertices()) > 0 or not models)
    if preset:
        set_current(example(preset), ("ex", preset))
        lg = current()
        go = True
    with right:
        st.markdown('<div class="section-title">Same graph, 4 schemes</div>', unsafe_allow_html=True)
        graph_pills(lg)
        if lg.custom:
            notice("This graph has custom weights; this page replaces them by each of the four schemes.", "warn")
        if isolated_card(lg):
            return
        key = (st.session_state.get("lg_key"), int(seed), model)
        if go:
            with st.spinner("Solving the graph under each weight scheme…"):
                st.session_state.schemes4 = dict(key=key, data=run_all(lg, model, int(seed)))
        s4 = st.session_state.get("schemes4")
        if not s4 or s4["key"] != key:
            st.markdown('<div class="caption" style="margin-top:12px;">Press <b>Run on all 4 schemes</b>, or one of the '
                        'example buttons.</div>', unsafe_allow_html=True)
            return
        data = s4["data"]
        rows = []
        for sch in SCHEMES:
            d = data[sch]
            r, opt = d["res"], d["opt"]
            gap = f"{100 * (r['weight'] - opt[0]) / opt[0]:.1f}%" if opt else "—"
            rows.append({"Scheme": NAMES[sch], "S (ours)": ", ".join(str(lg.label(v)) for v in r["S"][:12]) +
                         (" …" if len(r["S"]) > 12 else ""), "|S|": r["size"], "W(S)": solver.fmt_w(r["weight"]),
                         "W* (ILP)": solver.fmt_w(opt[0]) if opt else "—", "Gap": gap,
                         "Hubs in S": f"{d['hubs_in_S']} / {d['hubs']}",
                         "Mean degree in S": f"{d['mean_deg_S']:.2f} (graph {d['mean_deg']:.2f})",
                         "Valid": "yes" if r["valid"] else "NO"})
        table(rows, list(rows[0]), numeric=("|S|", "W(S)", "W* (ILP)", "Gap"))
        st.markdown('<div class="caption">Hubs = vertices whose degree is at least the 90th percentile. Under '
                    '<b>degree</b> weights hubs are expensive, under <b>inverse</b> weights they are cheap; compare '
                    'the hub counts and the mean degree of the chosen vertices.</div>', unsafe_allow_html=True)
        if lg.graph.n <= MAX_DRAW:
            cols = st.columns(2)
            for i, sch in enumerate(SCHEMES):
                w = with_scheme(lg, sch, int(seed))
                with cols[i % 2]:
                    with st.container(border=True):
                        r = data[sch]["res"]
                        st.markdown(f'<div class="card-title">{html.escape(NAMES[sch])} · W(S) = '
                                    f'{solver.fmt_w(r["weight"])}, |S| = {r["size"]}</div>'
                                    f'<div class="caption"><span class="mono">{FORMULA[sch]}</span> — '
                                    f'{MEANING[sch]}</div>', unsafe_allow_html=True)
                        st.iframe(solution_html(w, r["S"], height=300), height=340)
