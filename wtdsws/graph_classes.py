"""The four graph classes (ER, BA, chordal, bipartite) with their subtypes.

    class      subtype         generator
    er         er              G(n, p), average degree ~ U[3, 15]
    ba         ba              Barabási–Albert, m ∈ {1..8}  (m = 1: trees with many leaves)
    chordal    chordal_gnp     G(n, p) completed with networkx.complete_to_chordal_graph
               ktree           random k-tree, k ∈ {1..4}: new vertex joined to a random k-clique
               unit_interval   points on a line, edge iff distance < r  (proper interval graph)
    bipartite  bip_random      sides n1 + n2 = n, edge probability p
               convex_bip      X ordered; each y ∈ Y is adjacent to a consecutive interval of X

Isolated vertices are repaired WITHOUT leaving the class:
  er, ba, chordal_gnp, ktree   attach as a leaf to a random vertex (a leaf keeps a graph chordal)
  bip_random                   attach to a random vertex of the OTHER side
  unit_interval, convex_bip    re-sample with a larger r / longer intervals (a leaf could break them)
After any repair the class assertion is re-run (make_graph -> assert_class).
"""
from __future__ import annotations

import networkx as nx
import numpy as np

CLASSES = ("er", "ba", "chordal", "bipartite")
SUBTYPES = {"er": ("er",), "ba": ("ba",), "chordal": ("chordal_gnp", "ktree", "unit_interval"),
            "bipartite": ("bip_random", "convex_bip")}
CLASS_LABEL = {"er": "ER", "ba": "BA", "chordal": "Chordal", "bipartite": "Bipartite"}
SUBTYPE_LABEL = {"er": "Erdős–Rényi G(n,p)", "ba": "Barabási–Albert", "chordal_gnp": "G(n,p) + chordal completion",
                 "ktree": "random k-tree", "unit_interval": "unit / proper interval",
                 "bip_random": "random bipartite", "convex_bip": "convex bipartite"}
MAX_RESAMPLE = 200


class ClassAssertionError(AssertionError):
    pass


# ------------------------------------------------------------------ checkers
def is_convex_bipartite(G, X_order, Y) -> bool:
    """True iff every y ∈ Y is adjacent to a set of CONSECUTIVE vertices in the order X_order
    (and G is bipartite with these sides)."""
    pos = {x: i for i, x in enumerate(X_order)}
    Ys = set(Y)
    if set(pos) & Ys or len(pos) + len(Ys) != G.number_of_nodes():
        return False
    for u, v in G.edges():
        if (u in Ys) == (v in Ys):
            return False                              # an edge inside one side
    for y in Y:
        idx = sorted(pos[x] for x in G.neighbors(y))
        if idx and idx[-1] - idx[0] + 1 != len(idx):
            return False
    return True


def is_unit_interval_realisation(G, points, r) -> bool:
    """The edge set equals {uv : |p_u - p_v| < r}."""
    order = sorted(G.nodes(), key=lambda u: points[u])
    expected = set()
    for i, u in enumerate(order):
        for v in order[i + 1:]:
            if points[v] - points[u] >= r:
                break
            expected.add((min(u, v), max(u, v)))
    return expected == {(min(u, v), max(u, v)) for u, v in G.edges()}


def is_k_tree(G, k: int) -> bool:
    """True iff G is a k-tree: repeatedly delete a vertex of degree exactly k whose neighbourhood is a clique
    until K_{k+1} remains. Sound (the deletions reversed build G as a k-tree) and complete (deleting a
    simplicial vertex of a k-tree leaves a k-tree, and a k-tree with > k+1 vertices has one of degree k)."""
    n = G.number_of_nodes()
    if n < k + 1 or G.number_of_edges() != k * n - k * (k + 1) // 2 or not nx.is_connected(G):
        return False
    H = nx.Graph(G)

    def simplicial_k(v):
        nb = list(H.neighbors(v))
        return len(nb) == k and all(H.has_edge(a, b) for i, a in enumerate(nb) for b in nb[i + 1:])

    stack = [v for v in H if simplicial_k(v)]
    while H.number_of_nodes() > k + 1:
        while stack and (stack[-1] not in H or not simplicial_k(stack[-1])):
            stack.pop()
        if not stack:
            return False
        v = stack.pop()
        nb = list(H.neighbors(v))
        H.remove_node(v)
        stack.extend(u for u in nb if simplicial_k(u))
    return H.number_of_edges() == k * (k + 1) // 2


def assert_class(G, spec: dict) -> None:
    """Raise ClassAssertionError unless G is in its class (and has no isolated vertex)."""
    cls, fam = spec.get("cls"), spec.get("family")
    iso = [u for u in G.nodes() if G.degree(u) == 0]
    if iso:
        raise ClassAssertionError(f"{fam}: {len(iso)} isolated vertices")
    if cls == "chordal" and not nx.is_chordal(G):
        raise ClassAssertionError(f"{fam}: graph is not chordal")
    if cls == "bipartite":
        if not nx.is_bipartite(G):
            raise ClassAssertionError(f"{fam}: graph is not bipartite")
        n1 = G.graph.get("n1")
        if n1 is not None:
            X = set(range(n1))
            if any((u in X) == (v in X) for u, v in G.edges()):
                raise ClassAssertionError(f"{fam}: an edge joins two vertices of the same side")
    if fam == "convex_bip":
        n1 = G.graph["n1"]
        if not is_convex_bipartite(G, list(range(n1)), list(range(n1, G.number_of_nodes()))):
            raise ClassAssertionError("convex_bip: some y is not adjacent to a consecutive interval of X")
    if fam == "unit_interval":
        if not is_unit_interval_realisation(G, G.graph["points"], G.graph["r"]):
            raise ClassAssertionError("unit_interval: edges do not match the interval model")


# ---------------------------------------------------------------- builders
def _leaf_repair(G, rng):
    nodes = list(G.nodes())
    for u in [u for u in nodes if G.degree(u) == 0]:
        if G.degree(u) == 0:
            v = u
            while v == u:
                v = nodes[int(rng.integers(len(nodes)))]
            G.add_edge(u, v)
    return G


def chordal_gnp(n, avg_deg, seed):
    rng = np.random.default_rng(seed)
    p = min(1.0, avg_deg / max(n - 1, 1))
    G0 = nx.gnp_random_graph(n, p, seed=int(rng.integers(2**31 - 1)))
    H, _ = nx.complete_to_chordal_graph(G0)              # returns (chordal graph, elimination ordering)
    H = nx.Graph(H)
    H.add_nodes_from(range(n))
    return _leaf_repair(H, rng)


def ktree(n, k, seed):
    rng = np.random.default_rng(seed)
    k = max(1, min(int(k), n - 1))
    G = nx.complete_graph(k + 1)
    cliques = [tuple(c) for c in _k_subsets(range(k + 1), k)]
    for v in range(k + 1, n):
        C = cliques[int(rng.integers(len(cliques)))]
        G.add_edges_from((v, u) for u in C)
        for i in range(k):
            cliques.append(tuple(sorted(C[:i] + C[i + 1:] + (v,))))
    return G


def _k_subsets(items, k):
    from itertools import combinations
    return combinations(sorted(items), k)


def unit_interval(n, avg_deg, seed):
    """Points ~ U[0, 1); edge iff |x_u - x_v| < r with r = avg_deg / (2n). If a vertex is isolated,
    r grows by 10% and the points are re-drawn (no leaves are attached: that could leave the class)."""
    rng = np.random.default_rng(seed)
    r = max(avg_deg, 1.0) / (2.0 * n)
    for _ in range(MAX_RESAMPLE):
        x = rng.random(n)
        order = np.argsort(x, kind="stable")
        G = nx.Graph()
        G.add_nodes_from(range(n))
        for i in range(n):
            u = int(order[i])
            for j in range(i + 1, n):
                v = int(order[j])
                if x[v] - x[u] >= r:
                    break
                G.add_edge(u, v)
        if min(d for _, d in G.degree()) > 0:
            G.graph.update(points={i: float(x[i]) for i in range(n)}, r=float(r))
            return G
        r *= 1.1
    raise RuntimeError("unit_interval: could not avoid isolated vertices")


def bip_random(n, avg_deg, seed):
    """Sides X = {0..n1-1}, Y = {n1..n-1}; an isolated vertex is joined to a random vertex of the other side."""
    rng = np.random.default_rng(seed)
    n1 = int(rng.integers(max(1, n // 4), max(2, (3 * n) // 4) + 1))
    n1 = min(max(n1, 1), n - 1)
    n2 = n - n1
    p = min(1.0, avg_deg * n / (2.0 * n1 * n2))          # expected average degree = 2 n1 n2 p / n
    G = nx.Graph()
    G.add_nodes_from(range(n))
    M = rng.random((n1, n2)) < p
    xs, ys = np.nonzero(M)
    G.add_edges_from(zip(xs.tolist(), (ys + n1).tolist()))
    for u in range(n):
        if G.degree(u) == 0:
            other = range(n1, n) if u < n1 else range(n1)
            G.add_edge(u, int(rng.choice(list(other))))
    G.graph["n1"] = n1
    return G


def convex_bip(n, avg_deg, seed):
    """X = {0..n1-1} in this order; y ∈ Y gets interval [a, a+L-1] of X, L ~ U{1..Lmax}.
    If some x ∈ X is in no interval, Lmax grows and the intervals are re-drawn."""
    rng = np.random.default_rng(seed)
    n1 = int(rng.integers(max(1, n // 3), max(2, (2 * n) // 3) + 1))
    n1 = min(max(n1, 1), n - 1)
    n2 = n - n1
    # mean interval length (Lmax+1)/2 gives average degree 2 n2 (Lmax+1)/2 / n
    Lmax = max(1, min(n1, int(round(avg_deg * n / n2)) - 1))
    for _ in range(MAX_RESAMPLE):
        G = nx.Graph()
        G.add_nodes_from(range(n))
        for y in range(n1, n):
            L = int(rng.integers(1, Lmax + 1))
            a = int(rng.integers(0, n1 - L + 1))
            G.add_edges_from((x, y) for x in range(a, a + L))
        if min(d for _, d in G.degree()) > 0:
            G.graph["n1"] = n1
            return G
        Lmax = min(n1, Lmax + max(1, Lmax // 5))
    raise RuntimeError("convex_bip: could not cover every x")


def build_class_family(spec: dict):
    f, n, seed = spec["family"], int(spec["n"]), spec.get("seed")
    if f == "chordal_gnp":
        return chordal_gnp(n, spec["avg_deg"], seed)
    if f == "ktree":
        return ktree(n, spec["k"], seed)
    if f == "unit_interval":
        return unit_interval(n, spec["avg_deg"], seed)
    if f == "bip_random":
        return bip_random(n, spec["avg_deg"], seed)
    if f == "convex_bip":
        return convex_bip(n, spec["avg_deg"], seed)
    raise ValueError(f)


# ------------------------------------------------------------------- specs
def class_spec(cls: str, n: int, rng: np.random.Generator, subtype=None, density=None) -> dict:
    """A random spec of class `cls` (subtype uniform if not given). `density` overrides the
    class's density parameter (average degree; m for BA; k for k-trees)."""
    if cls not in CLASSES:
        raise ValueError(f"unknown graph class {cls!r}; choose from {CLASSES}")
    subs = SUBTYPES[cls]
    sub = subtype or subs[int(rng.integers(len(subs)))]
    if sub not in subs:
        raise ValueError(f"subtype {sub!r} is not in class {cls!r} {subs}")
    seed = int(rng.integers(2**31 - 1))
    if sub == "er":
        spec = dict(family="er", n=n, avg_deg=float(density or rng.uniform(3, 15)), seed=seed)
    elif sub == "ba":
        spec = dict(family="ba", n=n, m=int(density or rng.integers(1, 9)), seed=seed)
    elif sub == "chordal_gnp":
        spec = dict(family="chordal_gnp", n=n, avg_deg=float(density or rng.uniform(1.0, 3.0)), seed=seed)
    elif sub == "ktree":
        spec = dict(family="ktree", n=n, k=int(density or rng.integers(1, 5)), seed=seed)
    elif sub == "unit_interval":
        spec = dict(family="unit_interval", n=n, avg_deg=float(density or rng.uniform(2, 10)), seed=seed)
    elif sub == "bip_random":
        spec = dict(family="bip_random", n=n, avg_deg=float(density or rng.uniform(3, 10)), seed=seed)
    else:
        spec = dict(family="convex_bip", n=n, avg_deg=float(density or rng.uniform(3, 10)), seed=seed)
    spec["cls"] = cls
    return spec


class ClassSchemeSampler:
    """Training/validation/test distribution: class ~ U{er, ba, chordal, bipartite},
    scheme ~ U{degree, inverse, random, mixed}, n ~ U[n_min, n_max].

    `seed_range` = [lo, hi): every graph seed and weight seed is drawn from it, so the
    training, validation and test ranges are disjoint by construction."""

    def __init__(self, seed: int, seed_range=(0, 10**9), classes=CLASSES, schemes=None):
        from .weights import SCHEMES
        self.rng = np.random.default_rng(seed)
        self.seed_range = tuple(int(x) for x in seed_range)
        self.classes = tuple(classes)
        self.schemes = tuple(schemes or SCHEMES)

    def _seed(self):
        return int(self.rng.integers(*self.seed_range))

    def sample_spec(self, n_min: int, n_max: int, cls=None, scheme=None, subtype=None) -> dict:
        rng = self.rng
        cls = cls or self.classes[int(rng.integers(len(self.classes)))]
        scheme = scheme or self.schemes[int(rng.integers(len(self.schemes)))]
        n = int(rng.integers(n_min, n_max + 1))
        spec = class_spec(cls, n, rng, subtype=subtype)
        spec["seed"] = self._seed()
        spec["scheme"] = scheme
        spec["weight_seed"] = self._seed()
        return spec

    def sample(self, n_min: int, n_max: int, **kw):
        from .generators import make_graph
        return make_graph(self.sample_spec(n_min, n_max, **kw))


def generate_graph(cls: str = "er", n: int = 50, seed: int = 0, subtype=None, density=None):
    """A networkx graph of the given class (no isolated vertices, class asserted).
    G.graph["spec"] holds the spec, so `make_graph(spec)` rebuilds the same graph."""
    from .generators import make_graph
    rng = np.random.default_rng(seed)
    spec = class_spec(cls, int(n), rng, subtype=subtype, density=density)
    g = make_graph(spec)
    G = g.to_networkx()
    G.graph.update(spec=spec, cls=cls, subtype=spec["family"])
    return G

