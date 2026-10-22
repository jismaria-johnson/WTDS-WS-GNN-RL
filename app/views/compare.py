import altair as alt
import pandas as pd
import streamlit as st

from components import solver
from components.state import context, graph_pills, isolated_card, scheme_label, weighted_current
from components.theme import AMBER, GREY, INK, LINE, MUTED, TEAL, card_title, notice, table


def render():
    lg = weighted_current()
    g = lg.graph
    left, right = st.columns([1, 3], gap="large")
    with left:
        with st.container(border=True):
            card_title("Compare")
            st.markdown('<div class="caption">Runs every method on the current graph with the current weight '
                        'scheme (both set on the <b>Solve</b> page). Local search uses the same time limit for greedy '
                        'and for the GNN.</div>',
                        unsafe_allow_html=True)
            models = solver.available_models()
            model = st.radio("Model", list(models) or ["—"], key="cmp_model")
            ls = st.slider("Local search time (s)", 0.0, 3.0, 1.0, 0.5, key="cmp_ls")
            go = st.button("Run all methods", type="primary", width="stretch", disabled=len(g.isolated_vertices()) > 0)
            if g.n > solver.ILP_MAX_N:
                notice(f"Exact ILP is skipped above {solver.ILP_MAX_N} vertices.", "warn")
    with right:
        st.markdown('<div class="section-title">Compare methods</div>', unsafe_allow_html=True)
        graph_pills(lg, scheme_label())
        if isolated_card(lg):
            return
        if go:
            rows = []
            names = {"greedy": "Weighted greedy", "greedy_rr": "Weighted greedy + RR",
                     "greedy_rr_ls": "Weighted greedy + RR + LS", "zero": "wtds model zero-shot + RR",
                     "rl_rr": "GNN + RL + RR (ours)", "rl_rr_ls": "GNN + RL + RR + LS (ours)", "ilp": "Exact ILP"}
            runs = [("greedy", lambda: solver.run_baseline(g, "greedy")),
                    ("greedy_rr", lambda: solver.run_baseline(g, "greedy_rr")),
                    ("greedy_rr_ls", lambda: solver.run_baseline(g, "greedy_rr", ls_time=ls)) if ls > 0 else None,
                    ("zero", lambda: solver.run_zero_shot(g, rr=True)) if solver.ZERO_SHOT.exists() else None,
                    ("rl_rr", lambda: solver.run_rl(g, model, rr=True, ls_time=0.0)) if models else None,
                    ("rl_rr_ls", lambda: solver.run_rl(g, model, rr=True, ls_time=ls)) if models and ls > 0 else None,
                    ("ilp", lambda: solver.run_baseline(g, "ilp", ilp_time=30)) if g.n <= solver.ILP_MAX_N else None]
            with st.spinner("Running the weighted baselines, the GNN policy and the exact ILP…"):
                for item in runs:
                    if item is None:
                        continue
                    key, fn = item
                    r = fn()
                    rows.append(dict(key=key, method=names[key], weight=r["weight"], size=r["size"],
                                     runtime=r["runtime"], valid=r["valid"], optimal=r.get("optimal")))
            st.session_state.compare = dict(key=context(), rows=rows)
        cmp_ = st.session_state.get("compare")
        if not cmp_ or cmp_.get("key") != context():
            st.markdown('<div class="caption" style="margin-top:12px;">Press <b>Run all methods</b>.</div>',
                        unsafe_allow_html=True)
            return
        rows = cmp_["rows"]
        opt = solver.optimum_for(lg)
        ref = opt[0] if opt else next((r["weight"] for r in rows if r["key"] == "ilp" and r["optimal"]), None)
        out = []
        for r in rows:
            gap = f"{100 * (r['weight'] - ref) / ref:.1f}%" if ref else "—"
            note = ""
            if r["key"] == "ilp":
                note = "proven optimal" if r["optimal"] else "time limit (not proven)"
            out.append({"Method": r["method"], "W(S)": solver.fmt_w(r["weight"]), "|S|": r["size"], "Gap": gap,
                        "Runtime (s)": f"{r['runtime']:.3f}", "Valid": "yes" if r["valid"] else "NO", "Note": note})
        table(out, ["Method", "W(S)", "|S|", "Gap", "Runtime (s)", "Valid", "Note"],
              numeric=("W(S)", "|S|", "Gap", "Runtime (s)"), highlight=lambda r: "ours" in r["Method"])
        if ref:
            st.markdown(f'<div class="caption">Gap is relative to the minimum weight W* = {solver.fmt_w(ref)} '
                        f'({opt[1] if opt else "proven by the exact ILP"}).</div>', unsafe_allow_html=True)
        cheapest = min(rows, key=lambda r: (r["weight"], r["size"]))
        smallest = min(rows, key=lambda r: (r["size"], r["weight"]))
        if cheapest["size"] > smallest["size"] and cheapest["weight"] < smallest["weight"]:
            notice(f"<b>Cheapest is not smallest.</b> The cheapest set found ({cheapest['method']}: W = "
                   f"{solver.fmt_w(cheapest['weight'])}) has {cheapest['size']} vertices, more than the smallest set "
                   f"found ({smallest['method']}: {smallest['size']} vertices, W = {solver.fmt_w(smallest['weight'])}).")
        df = pd.DataFrame([{"Method": r["method"], "W(S)": r["weight"],
                            "kind": "ours" if "ours" in r["method"] else ("exact" if r["key"] == "ilp" else "baseline")}
                           for r in rows])
        chart = (alt.Chart(df).mark_bar(cornerRadiusEnd=3, height=22)
                 .encode(y=alt.Y("Method:N", sort=None, title=None,
                                 axis=alt.Axis(labelColor=INK, labelFontSize=12, labelLimit=280)),
                         x=alt.X("W(S):Q", title="total weight W(S) (smaller is better)",
                                 axis=alt.Axis(labelColor=MUTED, gridColor=LINE, titleColor=MUTED)),
                         color=alt.Color("kind:N", scale=alt.Scale(domain=["ours", "baseline", "exact"],
                                                                   range=[TEAL, GREY, AMBER]), legend=None),
                         tooltip=["Method", "W(S)"])
                 .properties(height=46 * len(df) + 20)
                 .configure_view(strokeWidth=0).configure(font="Inter", background="#FFFFFF"))
        st.altair_chart(chart, width="stretch")
