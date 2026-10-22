"""Exact minimum-WEIGHT total dominating sets on special graph classes.

    tree_tds(graph)                     forests: linear-time rooted DP (4 states per vertex)
    unit_interval_tds(graph, points)    unit / proper interval graphs: our exact DP over the sorted interval model
    ilp_exact(graph, time_limit=60)     any graph: CP-SAT ILP ("Exact (ILP)"), optimality proven or not

All three minimise the LEXICOGRAPHIC objective (W(S), |S|): minimum total weight first and, among the
minimum-weight sets, the smallest one. Weight-optimality is therefore unaffected, and the reported "exact |S|"
is well defined (it does not depend on how a solver breaks ties between equal-weight optima).

References
  Trees: Laskar, Pfaff, S. M. Hedetniemi & S. T. Hedetniemi (1984), "On the algorithmic complexity of total
  domination", SIAM J. Algebraic Discrete Methods 5(3):420-425 (linear algorithm on trees); the weighted
  version below is the standard rooted DP.
  Interval graphs: Bertossi & Gori (1988), SIAM J. Discrete Math. 1(3):317-327; M.-S. Chang (1998), SIAM J.
  Computing 27(6):1671-1694. Their algorithms were NOT reproduced; unit_interval_tds is our own DP for
  unit (= proper) interval graphs, derived and proved in docs/exact_interval.md.
"""
from __future__ import annotations

import time

import numpy as np

from ..checker import assert_tds
from ..graph import Graph, NoTotalDominatingSetError
from ..result import SolveResult

INF = (float("inf"), float("inf"))


def _add(a, b):
    return (a[0] + b[0], a[1] + b[1])


def _sub(a, b):
    return (a[0] - b[0], a[1] - b[1])


def _result(graph, S, t0, method, **info):
    S = sorted(int(v) for v in S)
    assert_tds(graph, S, method)
    return SolveResult.build(graph, S, time.perf_counter() - t0, method, optimal=True, **info)


# ------------------------------------------------------------------------------------------------ trees
# States of vertex v for its rooted subtree T_v (every vertex of T_v other than v is totally dominated
# inside T_v):
#   A  v in S,     dominated by a child          B  v in S,     no child in S (v needs its parent)
#   C  v not in S, dominated by a child          D  v not in S, no child in S (v needs its parent)
# A child c of v in S may be in any state (v dominates c); a child of v not in S must be in A or C.
# Costs are (weight, size) pairs compared lexicographically.
A, B, C, D = range(4)


def tree_tds(graph: Graph) -> SolveResult:
    """Minimum-weight (then minimum-size) TDS of a forest, exact, O(n). Rejects isolated vertices and cycles."""
    t0 = time.perf_counter()
    n = graph.n
    graph.require_tds_exists()
    if graph.num_edges != n - _components(graph):
        raise ValueError("tree_tds: the graph is not a forest")
    adj = graph.adj
    w = graph.w.tolist()
    cost = [None] * n          # cost[v] = [A, B, C, D]
    pick = [None] * n          # pick[v] = (child forced into S for A, child forced into A for C)
    parent = [-1] * n
    seen = [False] * n
    S = []
    for root in range(n):
        if seen[root]:
            continue
        order, stack = [], [root]
        seen[root] = True
        while stack:                                   # iterative DFS -> pre-order
            v = stack.pop()
            order.append(v)
            for u in adj[v]:
                if not seen[u]:
                    seen[u] = True
                    parent[u] = v
                    stack.append(u)
        for v in reversed(order):                      # post-order DP
            ch = [u for u in adj[v] if u != parent[v]]
            wv = (w[v], 1)
            any_sum, out_sum, ac_sum = (0, 0), (0, 0), (0, 0)
            best_in, best_a = None, None               # (extra cost, child)
            for c in ch:
                a, b, cc, d = cost[c]
                m_all = min(a, b, cc, d)
                m_in = min(a, b)
                m_ac = min(a, cc)
                any_sum = _add(any_sum, m_all)
                out_sum = _add(out_sum, cc)            # D[v]: children not in S and already dominated
                ac_sum = _add(ac_sum, m_ac)
                e_in = _sub(m_in, m_all) if m_in != INF else INF
                if best_in is None or e_in < best_in[0]:
                    best_in = (e_in, c)
                e_a = _sub(a, m_ac) if a != INF else INF
                if best_a is None or e_a < best_a[0]:
                    best_a = (e_a, c)
            # B[v]: v in S, no child in S; each child is in C or D (D is fine: v dominates it)
            b_sum = (0, 0)
            for c in ch:
                b_sum = _add(b_sum, min(cost[c][C], cost[c][D]))
            cA = _add(_add(wv, any_sum), best_in[0]) if ch and best_in[0] != INF else INF
            cB = _add(wv, b_sum)
            cC = _add(ac_sum, best_a[0]) if ch and best_a[0] != INF else INF
            cD = out_sum if all(cost[c][C] != INF for c in ch) else INF
            cost[v] = [cA, cB, cC, cD]
            pick[v] = (best_in[1] if ch else -1, best_a[1] if ch else -1)
        # reconstruction, top-down
        rs = A if cost[root][A] <= cost[root][C] else C
        if cost[root][rs] == INF:                       # pragma: no cover - impossible without isolated vertices
            raise NoTotalDominatingSetError("no TDS in a component")
        state = {root: rs}
        for v in order:
            s = state[v]
            ch = [u for u in adj[v] if u != parent[v]]
            if s in (A, B):
                S.append(v)
            for c in ch:
                cc = cost[c]
                if s == A:
                    if c == pick[v][0]:
                        state[c] = A if cc[A] <= cc[B] else B
                    else:
                        state[c] = min(range(4), key=lambda k: (cc[k], k))
                elif s == B:
                    state[c] = C if cc[C] <= cc[D] else D
                elif s == C:
                    state[c] = A if (c == pick[v][1] or cc[A] <= cc[C]) else C
                else:
                    state[c] = C
    return _result(graph, S, t0, "exact_tree_dp")


def _components(graph: Graph) -> int:
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components
    M = csr_matrix((np.ones(len(graph.indices)), graph.indices, graph.indptr), shape=(graph.n, graph.n))
    return int(connected_components(M, directed=False)[0])


# --------------------------------------------------------------------------------- unit-interval graphs
def interval_order(graph: Graph, points) -> list:
    """Vertices sorted by their point; asserts that every CLOSED neighbourhood is a contiguous block of this
    order (the property of a proper interval ordering that the DP needs)."""
    pts = np.asarray([points[v] for v in range(graph.n)], dtype=float)
    order = np.argsort(pts, kind="stable").tolist()
    pos = [0] * graph.n
    for i, v in enumerate(order):
        pos[v] = i
    for v in range(graph.n):
        idx = sorted([pos[u] for u in graph.adj[v]] + [pos[v]])
        if idx[-1] - idx[0] + 1 != len(idx):
            raise ValueError(f"interval_order: N[{v}] is not contiguous; not a proper interval ordering")
    return order


def unit_interval_tds(graph: Graph, points) -> SolveResult:
    """Our exact DP for unit (proper) interval graphs, O(n^2) (docs/exact_interval.md).

    In the sorted order 0..n-1, N[i] = [L(i), R(i)] with L, R non-decreasing. For S = s_1 < ... < s_k:
      S is a TDS  <=>  L(s_1) = 0,  R(s_k) = n-1,  L(s_{j+1}) <= R(s_j) + 1  for all j,
                       and every s_j is adjacent to s_{j-1} or to s_{j+1}.
    DP over the last chosen vertex t and a flag "t is already dominated by its predecessor".
    """
    t0 = time.perf_counter()
    graph.require_tds_exists()
    order = interval_order(graph, points)
    n = graph.n
    pos = [0] * n
    for i, v in enumerate(order):
        pos[v] = i
    L = [min(pos[u] for u in graph.adj[v] + [v]) for v in order]
    R = [max(pos[u] for u in graph.adj[v] + [v]) for v in order]
    if any(L[i] > L[i + 1] or R[i] > R[i + 1] for i in range(n - 1)):   # implied by contiguity; re-checked
        raise ValueError("unit_interval_tds: L/R not monotone; not a proper interval ordering")
    wt = [(float(graph.w[v]), 1) for v in order]
    # f[t][h]: best cost of a valid prefix ending with s_k = t, where every vertex < t is dominated, and
    # h = 1 iff t is dominated by s_{k-1}.  Predecessor stored for reconstruction.
    f = [[INF, INF] for _ in range(n)]
    prev = [[None, None] for _ in range(n)]
    for t in range(n):
        if L[t] == 0:
            f[t][0] = wt[t]                           # first chosen vertex: nothing before it
        for s in range(t - 1, -1, -1):                 # R is non-decreasing: once R[s] + 1 < L[t], stop
            if L[t] > R[s] + 1:                        # a vertex strictly between s and t would be undominated
                break
            adjacent = t <= R[s]
            for h in (0, 1):
                if f[s][h] == INF or (h == 0 and not adjacent):
                    continue                           # s must get its domination from t
                c = _add(f[s][h], wt[t])
                ht = 1 if adjacent else 0
                if c < f[t][ht]:
                    f[t][ht], prev[t][ht] = c, (s, h)
    best, end = INF, None
    for t in range(n):
        if R[t] == n - 1 and f[t][1] < best:          # the last chosen vertex must be dominated by its predecessor
            best, end = f[t][1], (t, 1)
    if end is None:                                    # pragma: no cover - V itself is a TDS
        raise NoTotalDominatingSetError("unit_interval_tds: no TDS")
    S, cur = [], end
    while cur is not None:
        S.append(order[cur[0]])
        cur = prev[cur[0]][cur[1]]
    return _result(graph, S, t0, "exact_interval_dp", dp_weight=best[0], dp_size=best[1])


# ------------------------------------------------------------------------------------------------- ILP
def lex_weights(graph: Graph) -> np.ndarray:
    """w'(v) = (n + 1) · w(v) + 1 for integer weights: minimising W'(S) = (n + 1) W(S) + |S| minimises W first
    (|S| <= n < n + 1 cannot outweigh one unit of W) and then |S|."""
    w = graph.w
    if not np.allclose(w, np.round(w)):
        raise ValueError("lex_weights needs integer weights")
    return (graph.n + 1) * np.round(w) + 1


def ilp_exact(graph: Graph, time_limit: float = 60.0, threads: int = 8, seed: int = 0, lexicographic: bool = True,
              hint=None) -> SolveResult:
    """CP-SAT on min Σ w'(v) x_v (lexicographic weights by default). info["optimal"] = proven optimal."""
    from .ilp import ilp
    t0 = time.perf_counter()
    g2 = Graph(graph.n, graph.indptr, graph.indices, lex_weights(graph) if lexicographic else graph.w,
               graph.name, dict(graph.meta))
    r = ilp(g2, time_limit=time_limit, threads=threads, seed=seed, hint=hint)
    S = sorted(int(v) for v in r.vertices)
    assert_tds(graph, S, "ilp_exact")
    return SolveResult.build(graph, S, time.perf_counter() - t0, "exact_ilp", optimal=bool(r.info["optimal"]),
                             status=r.info["status"], time_limit=float(time_limit), lexicographic=lexicographic)
