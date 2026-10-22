"""Validation of the exact special-class algorithms (wtdsws/baselines/exact_special.py).

    brute force  : tree DP and unit-interval DP vs exhaustive search, n <= 12, all 5 weightings
    ILP          : tree DP and unit-interval DP vs CP-SAT (proven optimal only), n = 50-200, all 5 weightings

An instance = one (graph, weighting). A mismatch is any difference in W(S); the minimum |S| among the
minimum-weight sets is compared too (the DPs are lexicographic). Every DP set is checked with
is_total_dominating_set and its weight recomputed from the set.
"""
from __future__ import annotations

import networkx as nx
import numpy as np

from .baselines.exact_special import ilp_exact, tree_tds, unit_interval_tds
from .baselines.ilp import brute_force_weighted
from .checker import is_total_dominating_set
from .graph import Graph
from .graph_classes import is_unit_interval_realisation, unit_interval
from .weights import weight_array

WEIGHTINGS = ("unit", "degree", "inverse", "random", "mixed")


def weights_for(deg, scheme: str, seed) -> np.ndarray:
    """The 5 weightings of the exact-vs-ours study: unit (all 1) plus the four schemes."""
    if scheme == "unit":
        return np.ones(len(deg))
    return weight_array(deg, scheme, seed).astype(np.float64)


def with_scheme(base: Graph, scheme: str, seed) -> Graph:
    meta = dict(base.meta)
    meta.update(scheme=scheme, weight_seed=seed)
    return Graph(base.n, base.indptr, base.indices, weights_for(base.degrees, scheme, seed), base.name, meta)


def small_tree(n, seed):
    """Random labelled tree (even seeds) or BA with m = 1 (odd seeds): the two tree families of the study."""
    T = nx.random_labeled_tree(n, seed=seed) if seed % 2 == 0 else nx.barabasi_albert_graph(n, 1, seed=seed)
    assert nx.is_tree(T)
    return T, None


def small_interval(n, seed, rng):
    G = unit_interval(n, float(rng.uniform(2, 10)), seed)
    assert nx.is_chordal(G) and is_unit_interval_realisation(G, G.graph["points"], G.graph["r"])
    return G, G.graph["points"]


def _instances(kind, count_graphs, n_lo, n_hi, seed):
    rng = np.random.default_rng(seed)
    for _ in range(count_graphs):
        n = int(rng.integers(n_lo, n_hi + 1))
        s = int(rng.integers(1 << 30))
        G, pts = small_tree(n, s) if kind == "tree" else small_interval(n, s, rng)
        assert list(G.nodes()) == list(range(n)) and min(d for _, d in G.degree()) > 0
        base = Graph.from_networkx(G)
        ws = int(rng.integers(1 << 30))
        for sch in WEIGHTINGS:
            yield with_scheme(base, sch, ws), pts


def _dp(kind, g, pts):
    r = tree_tds(g) if kind == "tree" else unit_interval_tds(g, pts)
    assert is_total_dominating_set(g.adj, r.vertices)
    assert r.weight == float(g.w[r.vertices].sum())
    return r


def validate(n_bf: int = 500, n_ilp: int = 300, seed: int = 20261019, ilp_time: float = 60.0) -> dict:
    """Returns {kind: {"bruteforce": {...}, "ilp": {...}}} with instance and mismatch counts."""
    out = {}
    for ki, kind in enumerate(("tree", "interval")):
        lo = 2 if kind == "tree" else 3
        bf = dict(instances=0, weight_mismatches=0, size_mismatches=0)
        for g, pts in _instances(kind, n_bf // len(WEIGHTINGS), lo, 12, seed + 10 * ki):
            r = _dp(kind, g, pts)
            W, _, sets = brute_force_weighted(g)
            bf["instances"] += 1
            bf["weight_mismatches"] += int(abs(r.weight - W) > 1e-9)
            bf["size_mismatches"] += int(r.size != min(len(S) for S in sets))
        il = dict(instances=0, not_proven=0, weight_mismatches=0, size_mismatches=0)
        for g, pts in _instances(kind, n_ilp // len(WEIGHTINGS), 50, 200, seed + 10 * ki + 1):
            r = _dp(kind, g, pts)
            e = ilp_exact(g, ilp_time, lexicographic=False)      # plain min-weight ILP: compares W
            el = ilp_exact(g, ilp_time, lexicographic=True)      # (W, |S|) ILP: compares |S| as well
            il["instances"] += 1
            if not (e.info["optimal"] and el.info["optimal"]):
                il["not_proven"] += 1
                continue
            il["weight_mismatches"] += int(abs(r.weight - e.weight) > 1e-9) + int(abs(r.weight - el.weight) > 1e-9)
            il["size_mismatches"] += int(r.size != el.size)
        out[kind] = {"bruteforce": bf, "ilp": il}
    return out
