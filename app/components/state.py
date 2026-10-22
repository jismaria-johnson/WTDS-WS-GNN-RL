"""The current graph and weight scheme, shared by all pages."""
import html

import streamlit as st

from .graph_input import SCHEME_LABELS, example, with_scheme
from .theme import notice

DEFAULT_EXAMPLE = "Two triangles + bridge"


def current():
    """The current graph (structure + any user-supplied weights), before the weight scheme is applied."""
    if "lg" not in st.session_state:
        st.session_state.lg = example(DEFAULT_EXAMPLE)
        st.session_state.lg_key = ("ex", DEFAULT_EXAMPLE)
    return st.session_state.lg


def set_current(lg, key):
    if st.session_state.get("lg_key") != key:
        st.session_state.lg = lg
        st.session_state.lg_key = key
        if lg.custom:
            st.session_state.scheme_sel = "Custom"
        elif st.session_state.get("scheme_sel") == "Custom":
            st.session_state.scheme_sel = "Degree"
        for k in ("result", "compare", "schemes4", "png"):
            st.session_state.pop(k, None)


def scheme_label():
    lab = st.session_state.get("scheme_sel") or ("Custom" if current().custom else "Degree")
    if lab == "Custom" and not current().custom:
        lab = "Degree"
    return lab


def scheme_key():
    return SCHEME_LABELS[scheme_label()]


def weight_seed():
    return int(st.session_state.get("wseed", 0))


def context():
    """Identifies (graph, scheme, seed): results computed for another context are not shown."""
    sch = scheme_key()
    return (st.session_state.get("lg_key"), sch, weight_seed() if sch in ("random", "mixed") else None)


def weighted_current():
    return with_scheme(current(), scheme_key(), weight_seed())


def isolated_card(lg):
    iso = lg.graph.isolated_vertices().tolist()
    if not iso:
        return False
    shown = ", ".join(html.escape(str(lg.label(v))) for v in iso[:20]) + (" …" if len(iso) > 20 else "")
    notice(f"<b>No total dominating set can exist for this graph.</b><br>"
           f"A total dominating set needs every vertex to have a <i>neighbour</i> in the set, and "
           f"{len(iso)} vertex{'es have' if len(iso) > 1 else ' has'} no neighbours at all: "
           f"<span class='mono'>{shown}</span>.<br>Remove the isolated vertices or connect them to the graph, "
           f"then try again.", "err")
    return True


def graph_pills(lg, scheme=None):
    g = lg.graph
    import networkx as nx
    comps = nx.number_connected_components(g.to_networkx()) if g.n <= 20000 else None
    w = g.w
    pills = [f"n = {g.n}", f"m = {g.num_edges}",
             f"scheme: {scheme}" if scheme else None,
             f"weights {w.min():g}–{w.max():g}, total {w.sum():g}",
             f"{comps} component{'s' if comps != 1 else ''}" if comps is not None else None,
             f"source: {lg.source}"]
    st.markdown("".join(f'<span class="pill">{html.escape(p)}</span>' for p in pills if p), unsafe_allow_html=True)
