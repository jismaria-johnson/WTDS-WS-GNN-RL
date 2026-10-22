"""Graph classes (Section 3): 500 samples per class pass the class assertion, no isolated vertex."""
import networkx as nx
import numpy as np
import pytest

from wtdsws import generate_graph
from wtdsws.generators import make_graph
from wtdsws.graph_classes import (CLASSES, SUBTYPES, ClassAssertionError, ClassSchemeSampler, assert_class,
                                  is_convex_bipartite, is_unit_interval_realisation)

N_PER_CLASS = 500


@pytest.mark.parametrize("cls", CLASSES)
def test_500_samples_per_class(cls):
    s = ClassSchemeSampler(100 + CLASSES.index(cls), seed_range=(0, 10**9))
    subs = set()
    for _ in range(N_PER_CLASS):
        spec = s.sample_spec(3, 120, cls=cls)
        g = make_graph(spec)                 # make_graph runs assert_class itself
        G = g.to_networkx()
        subs.add(spec["family"])
        assert G.number_of_nodes() == spec["n"]
        assert min(d for _, d in G.degree()) >= 1, "isolated vertex"
        if cls == "chordal":
            assert nx.is_chordal(G)
        if cls == "bipartite":
            assert nx.is_bipartite(G)
            if spec["family"] == "convex_bip":
                from wtdsws.graph_classes import build_class_family
                H = build_class_family(spec)
                n1 = H.graph["n1"]
                assert is_convex_bipartite(G, list(range(n1)), list(range(n1, g.n)))
        assert g.weights is not None and g.weights.min() >= 1 and g.weights.max() <= 100
    assert subs == set(SUBTYPES[cls]), subs        # every subtype was exercised


def test_unit_interval_matches_its_model_and_is_chordal():
    from wtdsws.graph_classes import unit_interval
    for seed in range(100):
        G = unit_interval(int(5 + seed % 60), 1.0 + seed % 7, seed)
        assert is_unit_interval_realisation(G, G.graph["points"], G.graph["r"])
        assert nx.is_chordal(G) and min(d for _, d in G.degree()) >= 1


def test_ktree_structure():
    from wtdsws.graph_classes import ktree
    for k in (1, 2, 3, 4):
        for seed in range(20):
            n = 10 + seed
            G = ktree(n, k, seed)
            assert G.number_of_edges() == k * (k + 1) // 2 + (n - k - 1) * k   # k-tree edge count
            assert nx.is_chordal(G) and nx.is_connected(G)
            if k == 1:
                assert nx.is_tree(G)


def test_ba_m1_is_tree_with_leaves():
    for seed in range(20):
        G = generate_graph("ba", n=50, seed=seed, density=1)
        assert nx.is_tree(G) and sum(1 for _, d in G.degree() if d == 1) >= 2


def test_convexity_checker_rejects_non_convex():
    G = nx.Graph([(0, 3), (2, 3), (1, 4)])          # y=3 sees x=0 and x=2 but not x=1
    assert not is_convex_bipartite(G, [0, 1, 2], [3, 4])
    assert is_convex_bipartite(G, [0, 2, 1], [3, 4])
    G2 = nx.Graph([(0, 1), (0, 2)])
    assert not is_convex_bipartite(G2, [0, 1], [2])   # edge inside X


def test_assert_class_catches_violations():
    C4 = nx.cycle_graph(4)
    with pytest.raises(ClassAssertionError):
        assert_class(C4, {"cls": "chordal", "family": "chordal_gnp"})
    with pytest.raises(ClassAssertionError):
        assert_class(nx.cycle_graph(3), {"cls": "bipartite", "family": "bip_random"})
    G = nx.path_graph(3)
    G.add_node(3)
    with pytest.raises(ClassAssertionError):
        assert_class(G, {"cls": "er", "family": "er"})


def test_generate_graph_api_deterministic():
    for cls in CLASSES:
        for sub in SUBTYPES[cls]:
            a = generate_graph(cls, n=40, seed=3, subtype=sub)
            b = generate_graph(cls, n=40, seed=3, subtype=sub)
            assert sorted(a.edges()) == sorted(b.edges()) and a.graph["subtype"] == sub
            assert sorted(make_graph(a.graph["spec"]).to_networkx().edges()) == sorted(a.edges())
    with pytest.raises(ValueError):
        generate_graph("planar", 10, 0)
    with pytest.raises(ValueError):
        generate_graph("chordal", 10, 0, subtype="convex_bip")
