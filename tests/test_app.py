"""End-to-end tests of the wtds-ws Streamlit app (headless, streamlit.testing).

Adapted from the wtds-gnn-rl app tests (same driver, same checks) for the new pages: weight-scheme selector,
graph classes, "Same graph, 4 schemes", Results. Hand-example optima are the brute-force values of Section 7.
"""
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
import os

# Runs once checkpoints/best.pt exists; WTDSWS_APP_TESTS=1 runs it earlier against the zero-shot fallback model.
pytestmark = pytest.mark.skipif(not (ROOT / "checkpoints" / "best.pt").exists() and not os.environ.get("WTDSWS_APP_TESTS"),
                                reason="needs checkpoints/best.pt")

DRIVER = """
import sys
sys.path.insert(0, r"{app}"); sys.path.insert(0, r"{root}")
import streamlit as st
from components.theme import inject
from components.graph_input import generate, parse_text, example
from components.state import set_current
inject()
{setup}
from views import {page}
{page}.render()
"""


def app_for(tmp_path, page, setup=""):
    from streamlit.testing.v1 import AppTest
    f = tmp_path / f"drv_{page}.py"
    f.write_text(DRIVER.format(app=ROOT / "app", root=ROOT, page=page, setup=setup), encoding="utf-8")
    return AppTest.from_file(str(f), default_timeout=300)


def text_of(at, cls):
    return " ".join(re.sub(r"<[^>]+>", " ", m.value) for m in at.markdown if f'class="{cls}' in m.value)


def stats(at):
    return " ".join(text_of(at, "stat").split())


def click(at, label):
    [b for b in at.button if b.label == label][0].click().run()
    assert not at.exception, [e.value for e in at.exception]


def setup_for(lg_expr, key, scheme=None, method=None):
    s = f"set_current({lg_expr}, {key!r})"
    if scheme:
        s += f"\nst.session_state.scheme_sel = {scheme!r}"
    if method:
        s += f"\nif 'method_set' not in st.session_state:\n    st.session_state.method = {method!r}\n" \
             f"    st.session_state.method_set = True"
    return s


def test_entry_app_loads():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / "app" / "streamlit_app.py"), default_timeout=300).run()
    assert not at.exception
    assert "Weighted Total Domination under Different Weight Schemes" in " ".join(m.value for m in at.markdown)


HAND = [("Two triangles + bridge", "Degree", 200), ("Two triangles + bridge", "Inverse degree", 134),
        ("House", "Degree", 167), ("House", "Inverse degree", 134), ("Path P4", "Degree", 200),
        ("Path P4", "Inverse degree", 100), ("Star K1,5", "Degree", 120), ("Star K1,5", "Inverse degree", 120),
        ("Kite", "Inverse degree", 109), ("Kite", "Degree", 225)]


@pytest.mark.parametrize("name,scheme,W", HAND, ids=[f"{h[0]}-{h[1]}" for h in HAND])
def test_hand_examples_ilp_optimum_and_ours_valid(tmp_path, name, scheme, W):
    at = app_for(tmp_path, "solve", setup_for(f"example({name!r})", ("ex", name), scheme, "Exact ILP")).run()
    assert not at.exception
    click(at, "Find total dominating set")
    s = stats(at)
    assert "Valid" in s and f"Total weight W(S) {W} " in s and "Gap 0.0%" in s
    at.session_state.method = "GNN + RL (ours)"
    at.run()
    click(at, "Find total dominating set")
    s = stats(at)
    assert "Valid" in s and "GNN + RL (ours)" in s and f"vs W* = {W}" in s


CLASS_SCHEME = [(c, sub, sch) for c, sub in (("er", "er"), ("ba", "ba"), ("chordal", "ktree"), ("bipartite", "bip_random"))
                for sch in ("Degree", "Inverse degree", "Random", "Mixed")]


@pytest.mark.parametrize("cls,sub,scheme", CLASS_SCHEME, ids=[f"{c}-{s}" for c, _, s in CLASS_SCHEME])
def test_generated_class_x_scheme_n100(tmp_path, cls, sub, scheme):
    from components_path import dens
    at = app_for(tmp_path, "solve", setup_for(f"generate({cls!r}, {sub!r}, 100, {dens(sub)!r}, 1)", (cls, sub),
                                              scheme)).run()
    click(at, "Find total dominating set")
    s = stats(at)
    assert "Valid" in s and "Gap" in s
    assert f"scheme: {scheme}" in " ".join(m.value for m in at.markdown)


def test_isolated_vertex_friendly_error(tmp_path):
    at = app_for(tmp_path, "solve", setup_for('parse_text("0 1\\n1 2\\n7", "")', "iso")).run()
    assert not at.exception and "No total dominating set can exist" in text_of(at, "notice")


def test_bad_weights_rejected():
    import sys
    sys.path.insert(0, str(ROOT / "app"))
    from components.graph_input import GraphInputError, parse_text
    for bad in ("-3", "0", "nan"):
        with pytest.raises(GraphInputError, match="vertex 2"):
            parse_text("1 2\n2 3", f"1 4\n2 {bad}")
    with pytest.raises(GraphInputError):
        parse_text("1 2\n2 3", "1 abc")


def test_bad_weight_paste_shows_error(tmp_path):
    at = app_for(tmp_path, "solve").run()
    at.text_area[1].set_value("1 5\n3 -2").run()
    click(at, "Use pasted graph")
    assert "Could not read the input" in text_of(at, "notice")


def test_custom_weights_switch_scheme(tmp_path):
    at = app_for(tmp_path, "solve", setup_for('parse_text("1 2\\n2 3\\n3 1", "1 5\\n2 1\\n3 2")', "cw")).run()
    click(at, "Find total dominating set")
    s = stats(at)
    assert "Valid" in s and "Total weight W(S) 3 " in s
    assert "scheme: Custom" in " ".join(m.value for m in at.markdown)


def test_messy_input_parsing():
    import sys
    sys.path.insert(0, str(ROOT / "app"))
    from components.graph_input import GraphInputError, parse_text
    lg = parse_text("# c\na b\nb b\na b\nb,c\nthis line has too many\n\nc a", "a 2\nb 3")
    assert lg.graph.n == 3 and lg.graph.num_edges == 3 and lg.labels == ["a", "b", "c"]
    assert list(lg.graph.w) == [2, 3, 1]
    w = " ".join(lg.warnings)
    assert "self-loop" in w and "duplicate" in w and "malformed" in w and "relabelled" in w and "defaults to 1" in w
    with pytest.raises(GraphInputError):
        parse_text("   \n# only a comment\n", "")
    lg = parse_text("v x 2\nv y 3\ne x y\n", "")                  # combined format pasted
    assert list(lg.graph.w) == [2, 3]


def test_large_ba_no_drawing_fast(tmp_path):
    import time
    at = app_for(tmp_path, "solve", setup_for('generate("ba", "ba", 5000, 3, 0)', "ba5k", "Degree")).run()
    t0 = time.time()
    click(at, "Find total dominating set")
    assert time.time() - t0 < 120
    assert "Valid" in stats(at) and "drawing is skipped" in text_of(at, "notice")


@pytest.mark.parametrize("preset", ["Two triangles: degree vs inverse", "House: degree vs inverse"])
def test_four_schemes_presets(tmp_path, preset):
    at = app_for(tmp_path, "schemes").run()
    click(at, preset)
    t = " ".join(re.sub(r"<[^>]+>", " ", m.value) for m in at.markdown)
    for name in ("Degree", "Inverse degree", "Random", "Mixed"):
        assert name in t
    assert "NO" not in [c for c in t.split()]                     # every set valid
    if preset.startswith("Two"):                                  # brute-force optima of Section 7
        assert re.search(r"Degree\s+3, 4\s+2\s+200\s+200\s+0\.0%", t), t
        assert re.search(r"Inverse degree\s+3, 4\s+2\s+134\s+134\s+0\.0%", t), t


@pytest.mark.parametrize("page,button", [("compare", "Run all methods"), ("results", None), ("how", None)])
def test_other_pages(tmp_path, page, button):
    at = app_for(tmp_path, page, setup_for('example("Two triangles + bridge")', "p", "Inverse degree")).run()
    assert not at.exception
    if button:
        click(at, button)
        t = " ".join(re.sub(r"<[^>]+>", " ", m.value) for m in at.markdown)
        assert "Exact ILP" in t and "proven optimal" in t


def test_exact_vs_ours_page(tmp_path):
    at = app_for(tmp_path, "exact").run()
    assert not at.exception
    t = " ".join(re.sub(r"<[^>]+>", " ", m.value) for m in at.markdown)
    assert "Exact vs ours" in t and "Plot 1" in t and "Plot 2" in t and "Plot 3" in t and "Plot 5" in t
    assert "Paper figure" in t and "Ours beats Greedy" in t and "Pilot sweep" in t
    assert "equal-cost sets can differ in size; total weight (Plot 3) is the objective" in t
    assert "several optimal sets can have the same minimum weight but different sizes" in t   # earlier study
    at.radio(key="final_table_scheme").set_value("Unit").run()
    assert not at.exception
    for label in ("Degree", "Inverse degree", "Mixed"):
        at.radio(key="exact_scheme").set_value(label).run()
        assert not at.exception
