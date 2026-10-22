"""Weight schemes (Section 2) and the exact hand examples (Section 7)."""
import math

import networkx as nx
import numpy as np
import pytest

from wtdsws import assign_weights, solve
from wtdsws.baselines import brute_force_weighted, ilp
from wtdsws.graph import Graph, NoTotalDominatingSetError
from wtdsws.graph_classes import ClassSchemeSampler
from wtdsws.weights import SCHEMES, weight_array

from ws_helpers import graph_from_labelled, class_scheme_graphs


# ------------------------------------------------------------------ properties
@pytest.mark.parametrize("scheme", SCHEMES)
def test_integer_range_and_deterministic(scheme):
    for g in class_scheme_graphs(80, seed=11, n_min=5, n_max=60):
        G = g.to_networkx()
        a = assign_weights(G, scheme, seed=123)
        b = assign_weights(G, scheme, seed=123)
        assert a == b
        assert set(a) == set(G.nodes())
        for x in a.values():
            assert isinstance(x, int) and 1 <= x <= 100
        # the same weights from a wtdsws Graph and from its networkx copy
        assert assign_weights(g, scheme, seed=123) == a


def test_random_and_mixed_depend_on_seed():
    G = nx.path_graph(40)
    assert assign_weights(G, "random", 1) != assign_weights(G, "random", 2)
    assert assign_weights(G, "mixed", 1) != assign_weights(G, "mixed", 2)


def test_degree_monotone_and_max_gets_100():
    for g in class_scheme_graphs(60, seed=12, n_min=5, n_max=80):
        d = g.degrees
        w = weight_array(d, "degree")
        o = np.argsort(d, kind="stable")
        assert (np.diff(w[o]) >= 0).all()                     # non-decreasing in degree
        assert (w[d == d.max()] == 100).all()
        assert all(w[v] == math.ceil(100 * d[v] / d.max()) for v in range(g.n))


def test_inverse_monotone_and_min_gets_100():
    for g in class_scheme_graphs(60, seed=13, n_min=5, n_max=80):
        d = g.degrees
        w = weight_array(d, "inverse")
        o = np.argsort(d, kind="stable")
        assert (np.diff(w[o]) <= 0).all()                     # non-increasing in degree
        assert (w[d == d.min()] == 100).all()
        assert all(w[v] == math.ceil(100 * d.min() / d[v]) for v in range(g.n))


def test_mixed_is_rounded_mean_of_the_other_three():
    for g in class_scheme_graphs(60, seed=14, n_min=5, n_max=80):
        G = g.to_networkx()
        for seed in (0, 7):
            wd, wi, wr, wm = (assign_weights(G, s, seed) for s in ("degree", "inverse", "random", "mixed"))
            for v in G.nodes():
                mean = (wd[v] + wi[v] + wr[v]) / 3
                assert wm[v] == min(100, max(1, int(math.floor(mean + 0.5))))
                assert abs(wm[v] - mean) <= 0.5


def test_isolated_vertex_raises():
    G = nx.Graph([(0, 1)])
    G.add_node(2)
    for s in SCHEMES:
        with pytest.raises(NoTotalDominatingSetError):
            assign_weights(G, s, 0)


def test_unknown_scheme():
    with pytest.raises(ValueError):
        assign_weights(nx.path_graph(3), "cubic", 0)


# --------------------------------------------------------------- hand examples
HOUSE = [(1, 2), (2, 3), (3, 4), (4, 1), (1, 5), (2, 5)]
PATH = [(1, 2), (2, 3), (3, 4)]
STAR = [(0, i) for i in range(1, 6)]
TRIANGLES = [(1, 2), (2, 3), (3, 1), (4, 5), (5, 6), (6, 4), (3, 4)]
KITE = [(1, 2), (1, 3), (1, 4), (1, 5), (2, 3), (4, 5), (5, 6), (6, 7)]

HAND = [
    # (name, edges, scheme, expected weights, min W, set of all optimal sets or None, one optimum)
    ("house", HOUSE, "degree", {1: 100, 2: 100, 3: 67, 4: 67, 5: 67}, 167, [{1, 4}, {2, 3}]),
    ("house", HOUSE, "inverse", {1: 67, 2: 67, 3: 100, 4: 100, 5: 100}, 134, [{1, 2}]),
    ("path", PATH, "degree", {1: 50, 2: 100, 3: 100, 4: 50}, 200, [{2, 3}]),
    ("path", PATH, "inverse", {1: 100, 2: 50, 3: 50, 4: 100}, 100, [{2, 3}]),
    ("star", STAR, "degree", {0: 100, **{i: 20 for i in range(1, 6)}}, 120, [{0, i} for i in range(1, 6)]),
    ("star", STAR, "inverse", {0: 20, **{i: 100 for i in range(1, 6)}}, 120, [{0, i} for i in range(1, 6)]),
    ("triangles", TRIANGLES, "degree", {1: 67, 2: 67, 3: 100, 4: 100, 5: 67, 6: 67}, 200, [{3, 4}]),
    ("triangles", TRIANGLES, "inverse", {1: 100, 2: 100, 3: 67, 4: 67, 5: 100, 6: 100}, 134, [{3, 4}]),
    ("kite", KITE, "inverse", {1: 25, 2: 50, 3: 50, 4: 50, 5: 34, 6: 50, 7: 100}, 109, [{1, 5, 6}]),
    ("kite", KITE, "degree", {1: 100, 2: 50, 3: 50, 4: 50, 5: 75, 6: 50, 7: 25}, 225, None),
]


@pytest.mark.parametrize("name,edges,scheme,weights,W,optima", HAND, ids=[f"{h[0]}-{h[2]}" for h in HAND])
def test_hand_example(name, edges, scheme, weights, W, optima):
    G = nx.Graph(edges)
    assert assign_weights(G, scheme, seed=0) == weights
    g, labels = graph_from_labelled(edges, weights)
    best, _, all_sets = brute_force_weighted(g)
    found = [{labels[v] for v in S} for S in all_sets]
    assert best == W
    if optima is not None:
        assert sorted(map(sorted, found)) == sorted(map(sorted, optima))
    else:                                                  # kite / degree: 5 optimal sets incl. {1,5,6}
        assert len(found) == 5 and {1, 5, 6} in found
    e = ilp(g, time_limit=10)
    assert e.info["optimal"] and e.weight == W and {labels[v] for v in e.vertices} in found
    # the public API on the labelled networkx graph, all methods valid, W recomputed
    for method in ("rl", "greedy", "greedy_rr", "greedy_rr_ls", "ilp"):
        r = solve(G, weights, method=method, ls_time=0.2)
        assert r.is_valid
        assert r.weight == sum(weights[labels[v]] for v in r.vertices) >= W
    assert solve(G, weights, method="ilp").weight == W


def test_path_inverse_is_forced():
    g, labels = graph_from_labelled(PATH, {1: 100, 2: 50, 3: 50, 4: 100})
    from wtdsws.reductions import forced_vertices
    assert sorted(labels[v] for v in forced_vertices(g)) == [2, 3]


# ------------------------------------------------------------------- seeds
def test_train_val_test_seed_ranges_disjoint():
    from wtdsws.seeds import SEED_RANGES, check_disjoint
    check_disjoint()
    ranges = sorted(SEED_RANGES.values())
    for (a0, a1), (b0, b1) in zip(ranges, ranges[1:]):
        assert a1 <= b0
    sam = {k: ClassSchemeSampler(1, seed_range=r) for k, r in SEED_RANGES.items()}
    seen = {k: {s.sample_spec(5, 10)["seed"] for _ in range(200)} for k, s in sam.items()}
    assert not (seen["train"] & seen["val"]) and not (seen["train"] & seen["test"]) and not (seen["val"] & seen["test"])


def test_pilot_seed_range_disjoint():
    from wtdsws.seeds import SEED_RANGES, check_disjoint, split_of
    check_disjoint()
    lo, hi = SEED_RANGES["pilot"]
    for k in ("train", "val", "test"):
        a, b = SEED_RANGES[k]
        assert hi <= a or b <= lo, k
    assert split_of(lo) == split_of(hi - 1) == "pilot" and split_of(hi) != "pilot"
    sam = ClassSchemeSampler(1, seed_range=SEED_RANGES["pilot"])
    pil = {s for _ in range(200) for s in (lambda sp: (sp["seed"], sp["weight_seed"]))(sam.sample_spec(5, 10))}
    assert all(split_of(s) == "pilot" for s in pil)
