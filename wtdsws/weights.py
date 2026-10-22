"""The four weight-assignment schemes. Every scheme gives integer weights in [1, 100].

Let d(v) = degree, Δ = max degree, δ = min degree (δ >= 1: no isolated vertex).

    degree   w(v) = ceil(100 · d(v) / Δ)        busy hubs are expensive
    inverse  w(v) = ceil(100 · δ / d(v))        hubs are cheap, remote sites are costly
    random   w(v) = uniform integer in [1, 100] from default_rng(seed)
    mixed    w(v) = round((w_degree + w_inverse + w_random) / 3), clipped to [1, 100]
             (reuses the same random draw as "random" for that seed)

All are deterministic given (G, seed). Ceilings are computed in exact integer
arithmetic, so no floating-point rounding can move a weight by one.
"""
from __future__ import annotations

import numpy as np

from .graph import Graph, NoTotalDominatingSetError

SCHEMES = ("degree", "inverse", "random", "mixed")

MEANING = {
    "degree": "Cost grows with the number of neighbours served, e.g. the monitoring load or energy "
              "of a busy router or sensor.",
    "inverse": "Well-connected sites are cheap per neighbour covered (existing infrastructure at hubs), "
               "so remote low-degree sites are costly to equip.",
    "random": "Independent site costs, e.g. land, installation or permits.",
    "mixed": "A realistic combination of load and site cost (equal blend of the other three).",
}

FORMULA = {
    "degree": "w(v) = ⌈100 · d(v) / Δ⌉",
    "inverse": "w(v) = ⌈100 · δ / d(v)⌉",
    "random": "w(v) ~ U{1, …, 100}  (seeded)",
    "mixed": "w(v) = round((w_deg + w_inv + w_rand) / 3)",
}


def _degrees_and_nodes(G):
    """(node list in a fixed order, degree array). Accepts a wtdsws Graph or a networkx graph."""
    if isinstance(G, Graph):
        return list(range(G.n)), np.asarray(G.degrees, dtype=np.int64)
    nodes = list(G.nodes())
    try:
        nodes = sorted(nodes)
    except TypeError:                     # mixed, unorderable labels: keep insertion order
        pass
    deg = np.array([G.degree(u) for u in nodes], dtype=np.int64)
    # self-loops do not count: total domination uses the open neighbourhood
    loops = {u for u, v in G.edges() if u == v}
    if loops:
        deg = np.array([sum(1 for x in G.neighbors(u) if x != u) for u in nodes], dtype=np.int64)
    return nodes, deg


def degree_weights(deg: np.ndarray) -> np.ndarray:
    Dmax = int(deg.max())
    return (100 * deg + Dmax - 1) // Dmax                    # ceil(100 d / Δ)


def inverse_weights(deg: np.ndarray) -> np.ndarray:
    dmin = int(deg.min())
    return (100 * dmin + deg - 1) // deg                      # ceil(100 δ / d)


def random_weights(n: int, seed) -> np.ndarray:
    return np.random.default_rng(seed).integers(1, 101, size=n).astype(np.int64)


def mixed_weights(deg: np.ndarray, seed) -> np.ndarray:
    s = degree_weights(deg) + inverse_weights(deg) + random_weights(len(deg), seed)
    # s/3 has fractional part 0, 1/3 or 2/3, so "round" has no ties: round(s/3) = (s + 1) // 3
    return np.clip((s + 1) // 3, 1, 100)


def weight_array(deg, scheme: str, seed=None) -> np.ndarray:
    """Integer weights (int64 array aligned with `deg`)."""
    deg = np.asarray(deg, dtype=np.int64)
    if deg.size == 0:
        raise NoTotalDominatingSetError("empty graph")
    if (deg <= 0).any():
        raise NoTotalDominatingSetError(
            f"graph has {int((deg <= 0).sum())} isolated vertex/vertices; weights (and a TDS) need min degree >= 1")
    if scheme == "degree":
        w = degree_weights(deg)
    elif scheme == "inverse":
        w = inverse_weights(deg)
    elif scheme == "random":
        if seed is None:
            raise ValueError("scheme 'random' needs a seed")
        w = random_weights(len(deg), seed)
    elif scheme == "mixed":
        if seed is None:
            raise ValueError("scheme 'mixed' needs a seed")
        w = mixed_weights(deg, seed)
    else:
        raise ValueError(f"unknown weight scheme {scheme!r}; choose from {SCHEMES}")
    assert w.min() >= 1 and w.max() <= 100, (scheme, w.min(), w.max())
    return w


def assign_weights(G, scheme: str = "degree", seed=0) -> dict:
    """{vertex: integer weight in [1, 100]} for a networkx graph or a wtdsws Graph.

    Vertices are taken in sorted order (insertion order if labels are not sortable), so the
    "random" draw for vertex v is deterministic given (G, seed).
    """
    nodes, deg = _degrees_and_nodes(G)
    w = weight_array(deg, scheme, seed)
    return {u: int(x) for u, x in zip(nodes, w.tolist())}


def weighted_graph(graph: Graph, scheme: str, seed=0) -> Graph:
    """A copy of `graph` carrying the scheme's weights (float64, integer-valued)."""
    w = weight_array(graph.degrees, scheme, seed).astype(np.float64)
    meta = dict(graph.meta)
    meta.update(scheme=scheme, weight_seed=seed)
    return Graph(graph.n, graph.indptr, graph.indices, w, graph.name, meta)
