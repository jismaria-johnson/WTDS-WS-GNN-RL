"""Helpers for the wtds-ws tests (graph classes x weight schemes)."""
import networkx as nx
import numpy as np

from wtdsws.generators import make_graph
from wtdsws.graph import Graph
from wtdsws.graph_classes import ClassSchemeSampler


def graph_from_labelled(edges, weights: dict):
    """wtdsws Graph from labelled edges; returns (graph, index -> label)."""
    labels = sorted({u for e in edges for u in e})
    idx = {u: i for i, u in enumerate(labels)}
    g = Graph.from_edges(len(labels), [(idx[u], idx[v]) for u, v in edges],
                         weights=[weights[u] for u in labels])
    return g, labels


def class_scheme_graphs(count: int, seed: int, n_min: int = 5, n_max: int = 60):
    """`count` graphs cycling through every class x scheme (subtype random), with weights."""
    from wtdsws.graph_classes import CLASSES
    from wtdsws.weights import SCHEMES
    s = ClassSchemeSampler(seed, seed_range=(0, 10**9))
    out = []
    for i in range(count):
        c = CLASSES[i % len(CLASSES)]
        w = SCHEMES[(i // len(CLASSES)) % len(SCHEMES)]
        out.append(s.sample(n_min, n_max, cls=c, scheme=w))
    return out


def as_nx(g: Graph):
    return g.to_networkx()
