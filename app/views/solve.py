import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from components import solver
from components.graph_input import (CLASS_LABELS, DENSITY, EXAMPLES, SAMPLE_COMBINED, SAMPLE_EDGES, SAMPLE_WEIGHTS,
                                    SCHEME_LABELS, SUBTYPE_LABELS, GraphInputError, example, generate, parse_text,
                                    parse_upload)
from components.network import MAX_DRAW, solution_html
from components.state import (context, current, graph_pills, isolated_card, scheme_key, scheme_label, set_current,
                              weighted_current)
from components.theme import LINE, MUTED, TEAL, card_title, notice, stat_tiles

from wtdsws.weights import FORMULA, MEANING

METHODS = ["GNN + RL (ours)", "Weighted greedy", "Greedy + RR", "Exact ILP"]


def input_card():
    card_title("1 · Graph")
    t_ex, t_gen, t_up, t_paste = st.tabs(["Examples", "Generate", "Upload", "Paste"])
    with t_ex:
        st.caption("Hand-checked graphs (optima verified by brute force). Pick a weight scheme below.")
        for i, name in enumerate(EXAMPLES):
            if st.button(name, key=f"ex_{i}", width="stretch",
                         type="primary" if st.session_state.get("lg_key") == ("ex", name) else "secondary"):
                set_current(example(name), ("ex", name))
                st.rerun()
    with t_gen:
        cls = CLASS_LABELS[st.selectbox("Graph class", list(CLASS_LABELS), index=0, key="gen_cls")]
        subs = SUBTYPE_LABELS[cls]
        sub = subs[st.selectbox("Subtype", list(subs), index=0, key=f"gen_sub_{cls}")]
        n = st.slider("Vertices", 10, 5000, 100, step=10, key="gen_n")
        lab, lo, hi, dflt, step, is_int = DENSITY[sub]
        dens = st.slider(lab, lo, hi, dflt, step, key=f"gen_d_{sub}")
        seed = st.number_input("Graph seed", 0, 10**6, 0, key="gen_seed")
        if st.button("Use this graph", key="gen_go", width="stretch"):
            d = int(dens) if is_int else float(dens)
            set_current(generate(cls, sub, int(n), d, int(seed)), ("gen", cls, sub, n, d, seed))
    with t_up:
        st.caption("Edges: `u v` per line (any labels), the combined format (`v <vertex> <weight>`, `e <u> <v>`), "
                   "DIMACS `.col`/`.clq`, or `.mtx`. Optional weights file: `vertex weight` per line (weights > 0). "
                   "With weights the scheme switches to Custom.")
        up = st.file_uploader("Graph file", type=["txt", "edges", "el", "col", "clq", "mtx"], key="up_g")
        upw = st.file_uploader("Weights file (optional)", type=["txt", "w", "weights"], key="up_w")
        c1, c2, c3 = st.columns(3)
        c1.download_button("Edges", SAMPLE_EDGES, "sample_edges.txt", "text/plain", width="stretch")
        c2.download_button("Weights", SAMPLE_WEIGHTS, "sample_weights.txt", "text/plain", width="stretch")
        c3.download_button("Combined", SAMPLE_COMBINED, "sample_combined.txt", "text/plain", width="stretch")
        if up is not None:
            key = ("up", up.name, up.size, up.file_id, upw.file_id if upw else None)
            if st.session_state.get("last_upload") != key:
                st.session_state.last_upload = key
                try:
                    set_current(parse_upload(up.name, up.getvalue(), upw.getvalue() if upw else None), key)
                except GraphInputError as ex:
                    st.session_state.input_error = str(ex)
    with t_paste:
        et = st.text_area("Edges (u v per line)", "1 2\n2 3\n3 1\n4 5\n5 6\n6 4\n3 4", height=130)
        wt = st.text_area("Custom weights (optional, vertex weight per line)", "", height=90,
                          help="Leave empty to use a weight scheme. Weights must be > 0.")
        if st.button("Use pasted graph", width="stretch"):
            try:
                set_current(parse_text(et, wt, "pasted text"), ("paste", et, wt))
                st.session_state.pop("input_error", None)
            except GraphInputError as ex:
                st.session_state.input_error = str(ex)


def scheme_card(lg):
    card_title("2 · Weight scheme")
    opts = [k for k in SCHEME_LABELS if k != "Custom" or lg.custom]
    if st.session_state.get("scheme_sel") not in opts:
        st.session_state.scheme_sel = scheme_label()
    st.segmented_control("Weight scheme", opts, key="scheme_sel", label_visibility="collapsed")
    sch = scheme_key()
    if sch in ("random", "mixed"):
        st.number_input("Weight seed", 0, 10**6, key="wseed")
    if sch == "custom":
        st.markdown('<div class="caption">Custom weights from your file / pasted text.</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="caption"><span class="mono">{FORMULA[sch]}</span><br>{MEANING[sch]}</div>',
                    unsafe_allow_html=True)


def weight_histogram(g, height=140):
    w = np.asarray(g.w, float)
    bins = np.arange(0, 101, 10) if w.max() <= 100 and w.min() >= 1 else np.histogram_bin_edges(w, 10)
    cnt, edges = np.histogram(w, bins=bins)
    df = pd.DataFrame({"w_lo": edges[:-1], "w_hi": edges[1:], "vertices": cnt})
    ax = dict(labelColor=MUTED, titleColor=MUTED, gridColor=LINE)
    ch = (alt.Chart(df).mark_rect(color=TEAL, stroke="#FFFFFF", strokeWidth=1)
          .encode(x=alt.X("w_lo:Q", title="weight", scale=alt.Scale(domain=[float(edges[0]), float(edges[-1])]),
                          axis=alt.Axis(**ax)), x2="w_hi:Q",
                  y=alt.Y("vertices:Q", title="vertices", axis=alt.Axis(**ax, tickMinStep=1, tickCount=3),
                          scale=alt.Scale(domain=[0, max(1, int(cnt.max()))], nice=False)), y2=alt.datum(0),
                  tooltip=[alt.Tooltip("w_lo:Q", title="from"), alt.Tooltip("w_hi:Q", title="to"), "vertices:Q"])
          .properties(height=height).configure_view(strokeWidth=0).configure(font="Inter", background="#FFFFFF"))
    st.altair_chart(ch, width="stretch")


def method_card(lg):
    card_title("3 · Method")
    n = lg.graph.n
    if st.session_state.get("method") not in METHODS:
        st.session_state.method = METHODS[0]
    method = st.segmented_control("Method", METHODS, label_visibility="collapsed", key="method") or METHODS[0]
    ilp_off = n > solver.ILP_MAX_N
    if method == "Exact ILP" and ilp_off:
        notice(f"Exact ILP is disabled above {solver.ILP_MAX_N} vertices: the problem is NP-hard and the solver "
               f"could run for minutes. Use GNN + RL or greedy for this graph.", "warn")
    opts = {}
    if method == "GNN + RL (ours)":
        models = solver.available_models()
        if not models:
            notice("No checkpoint found in <span class='mono'>checkpoints/</span>.", "warn")
        opts["model"] = st.radio("Model", list(models) or ["—"], horizontal=True, key="model_choice")
        opts["rr"] = st.toggle("Redundancy removal", True, help="Drop vertices (most expensive first) while S stays a TDS.")
    if method in ("GNN + RL (ours)", "Greedy + RR"):
        ls = st.toggle("Local search", False, help="Time-limited 2-for-1 and 1-for-1 swaps that strictly lower W(S).")
        opts["ls_time"] = st.slider("Local search time (s)", 0.2, 5.0, 1.0, 0.2) if ls else 0.0
    go = st.button("Find total dominating set", type="primary", width="stretch",
                   disabled=(method == "Exact ILP" and ilp_off))
    return method, opts, go


def run(lg, method, opts):
    g = lg.graph
    if method == "GNN + RL (ours)":
        return solver.run_rl(g, opts["model"], rr=opts["rr"], ls_time=opts.get("ls_time", 0.0))
    key = {"Weighted greedy": "greedy", "Greedy + RR": "greedy_rr", "Exact ILP": "ilp"}[method]
    return solver.run_baseline(g, key, ls_time=opts.get("ls_time", 0.0))


def result_view(lg, res):
    g = lg.graph
    opt = solver.optimum_for(lg)
    if opt is None and res.get("optimal"):
        opt = (res["weight"], "proven optimal by the exact ILP")
    items = [("Total weight W(S)", solver.fmt_w(res["weight"]), res["method"], "hero"),
             ("|S|", res["size"], "vertices chosen", "")]
    if opt:
        val, src = opt
        gap = 100.0 * (res["weight"] - val) / val if val > 0 else 0.0
        items.append(("Gap", f"{gap:.1f}%", f"vs W* = {solver.fmt_w(val)}, {src}", "amber" if gap > 1e-9 else "teal"))
    elif "bound" in res:
        items.append(("ILP status", "time limit", f"not proven; bound {res['bound']:.1f}", "amber"))
    else:
        items.append(("Gap", "—", "ILP optimum not available (n > 150 or not proven in 10 s)", ""))
    items.append(("Runtime", f"{res['runtime']:.3f}s", "wall clock", ""))
    items.append(("Valid", "✓", "is_total_dominating_set", "teal"))
    stat_tiles(items)
    notice("<b>Valid total dominating set:</b> every vertex, including every vertex of S, has a neighbour in S. "
           "W(S) is recomputed from the chosen vertices and the weights.")
    if "size_raw" in res:
        st.markdown(f'<div class="caption">The policy picked {res["size_raw"]} vertices ({res["forced"]} forced by the '
                    f'degree-1 rule) → {res["size_rr"]} after redundancy removal → {res["size_ls"]} after local '
                    f'search.</div>', unsafe_allow_html=True)
    if g.n <= MAX_DRAW:
        st.iframe(solution_html(lg, res["S"], height=520), height=560)
    else:
        notice(f"The graph has {g.n} vertices; drawing is skipped above {MAX_DRAW} so the page stays responsive. "
               "Download the solution below.")
    c1, c2, _ = st.columns([1, 1, 2])
    c1.download_button("solution.json", solver.solution_json(lg, res), "solution.json", "application/json",
                       width="stretch")
    if g.n <= MAX_DRAW:
        if c2.button("Prepare PNG", width="stretch"):
            st.session_state.png = solver.solution_png(lg, res["S"])
        if st.session_state.get("png"):
            c2.download_button("solution.png", st.session_state.png, "solution.png", "image/png", width="stretch")


def render():
    left, right = st.columns([1, 2.5], gap="large")
    with left:
        with st.container(border=True):
            input_card()
        base = current()
        with st.container(border=True):
            scheme_card(base)
            wlg = weighted_current()
            if not len(wlg.graph.isolated_vertices()):
                weight_histogram(wlg.graph)
        with st.container(border=True):
            method, opts, go = method_card(wlg)
    with right:
        st.markdown('<div class="section-title">Result</div>', unsafe_allow_html=True)
        err = st.session_state.pop("input_error", None)
        if err:
            notice(f"<b>Could not read the input.</b> {err}", "err")
        graph_pills(wlg, scheme_label())
        if wlg.note:
            st.markdown(f'<div class="caption">{wlg.note}</div>', unsafe_allow_html=True)
        for w in wlg.warnings:
            notice(w, "warn")
        if isolated_card(wlg):
            return
        ctx = context()
        if go:
            try:
                with st.spinner("Finding a minimum-weight total dominating set…"):
                    st.session_state.result = dict(ctx=ctx, res=run(wlg, method, opts))
                    st.session_state.pop("png", None)
            except Exception as ex:
                notice(f"<b>Could not solve:</b> {ex}", "err")
        res = st.session_state.get("result")
        if res is None or res["ctx"] != ctx:
            st.markdown('<div class="caption" style="margin-top:14px;">Choose a weight scheme and a method, then press '
                        '<b>Find total dominating set</b>. Numbers on the vertices are their weights.</div>',
                        unsafe_allow_html=True)
            if wlg.graph.n <= MAX_DRAW:
                st.iframe(solution_html(wlg, [], height=500), height=540)
        else:
            result_view(wlg, res["res"])
