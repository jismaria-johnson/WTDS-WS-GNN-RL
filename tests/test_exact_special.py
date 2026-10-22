"""Exact special-class algorithms: tree DP, unit-interval DP, lexicographic ILP (wtdsws/baselines/exact_special.py).

The two validation tests are the gate before any exact-vs-ours plot: 0 mismatches against brute force
(500 instances each, n <= 12, all 5 weightings) and against proven-optimal CP-SAT (300 instances each, n = 50-200).
"""
import networkx as nx
import numpy as np
import pytest

from ws_helpers import graph_from_labelled
from wtdsws.baselines.exact_special import ilp_exact, interval_order, lex_weights, tree_tds, unit_interval_tds
from wtdsws.baselines.ilp import brute_force_weighted
from wtdsws.checker import is_total_dominating_set
from wtdsws.exact_validation import validate
from wtdsws.graph import Graph, NoTotalDominatingSetError
from wtdsws.graph_classes import is_k_tree, ktree


@pytest.fixture(scope="module")
def report():
    return validate(n_bf=500, n_ilp=300)


@pytest.mark.parametrize("kind", ["tree", "interval"])
def test_dp_equals_bruteforce_500(report, kind):
    bf = report[kind]["bruteforce"]
    assert bf["instances"] == 500
    assert bf["weight_mismatches"] == 0 and bf["size_mismatches"] == 0, bf


@pytest.mark.parametrize("kind", ["tree", "interval"])
def test_dp_equals_proven_ilp_300(report, kind):
    il = report[kind]["ilp"]
    assert il["instances"] == 300 and il["not_proven"] == 0, il
    assert il["weight_mismatches"] == 0 and il["size_mismatches"] == 0, il


def test_same_weight_different_size_path_example():
    """Path 1-2-3-4-5, w = {1:3, 2:4, 3:4, 4:1, 5:1}: minimum weight 9, attained by {2,3,4} (3 vertices) and
    {1,2,4,5} (4 vertices). This is why |S| can differ from the exact method on weighted graphs."""
    g, labels = graph_from_labelled([(1, 2), (2, 3), (3, 4), (4, 5)], {1: 3, 2: 4, 3: 4, 4: 1, 5: 1})
    W, _, sets = brute_force_weighted(g)
    assert W == 9
    assert sorted(sorted(labels[v] for v in S) for S in sets) == [[1, 2, 4, 5], [2, 3, 4]]
    for S in sets:
        assert is_total_dominating_set(g.adj, S) and float(g.w[S].sum()) == 9
    r = tree_tds(g)                                   # lexicographic: the smaller of the two optima
    assert r.weight == 9 and sorted(labels[v] for v in r.vertices) == [2, 3, 4]
    assert ilp_exact(g, 10).vertices == r.vertices


def test_tree_dp_forest_and_rejections():
    T1, T2 = nx.random_labeled_tree(9, seed=1), nx.path_graph(4)
    F = nx.disjoint_union(T1, T2)
    rng = np.random.default_rng(0)
    g = Graph.from_networkx(F)
    g = Graph(g.n, g.indptr, g.indices, rng.integers(1, 50, g.n).astype(float))
    r = tree_tds(g)
    assert r.is_valid and r.weight == brute_force_weighted(g)[0]
    with pytest.raises(NoTotalDominatingSetError):
        tree_tds(Graph.from_edges(3, [(0, 1)]))           # vertex 2 is isolated
    with pytest.raises(ValueError, match="not a forest"):
        tree_tds(Graph.from_networkx(nx.cycle_graph(5)))


def test_tree_dp_star_and_path():
    assert tree_tds(Graph.from_networkx(nx.star_graph(6))).size == 2
    for n in range(2, 15):                                 # gamma_t(P_n) = floor(n/2) + ceil(n/4) - floor(n/4)
        assert tree_tds(Graph.from_networkx(nx.path_graph(n))).size == n // 2 + (n + 3) // 4 - n // 4


def test_interval_dp_rejects_non_proper_order():
    G = nx.star_graph(3)                                   # K_{1,3}: an interval graph, not a proper one
    with pytest.raises(ValueError):
        interval_order(Graph.from_networkx(G), {0: 0.5, 1: 0.0, 2: 0.4, 3: 1.0})
    g = Graph.from_networkx(nx.path_graph(6))
    r = unit_interval_tds(g, {i: float(i) for i in range(6)})
    assert r.size == 4 and is_total_dominating_set(g.adj, r.vertices)


def test_is_k_tree():
    for n in (3, 4, 10, 60):
        assert is_k_tree(ktree(n, 2, n), 2)
    assert is_k_tree(nx.random_labeled_tree(20, seed=0), 1)
    assert not is_k_tree(nx.cycle_graph(5), 2)
    assert not is_k_tree(ktree(30, 3, 0), 2)
    assert not is_k_tree(nx.complete_bipartite_graph(3, 3), 2)   # 9 = 2*6 - 3 edges, but not chordal
    G = ktree(30, 2, 1)
    G.add_edge(30, 0)                                      # a pendant vertex: edge count of a 2-tree breaks
    assert not is_k_tree(G, 2)


def test_lex_weights_keep_weight_optimum():
    g, _ = graph_from_labelled([(1, 2), (2, 3), (3, 4), (4, 5)], {1: 3, 2: 4, 3: 4, 4: 1, 5: 1})
    assert list(lex_weights(g)) == [19, 25, 25, 7, 7]
    e = ilp_exact(g, 10, lexicographic=False)
    assert e.info["optimal"] and e.weight == 9
