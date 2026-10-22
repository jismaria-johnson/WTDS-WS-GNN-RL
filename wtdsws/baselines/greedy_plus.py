"""Stronger greedy baselines (revision). Each is followed by redundancy removal (RR); `ls_time` > 0 adds the same
local search as greedy + RR + LS.

greedy_tiebreak        max gain/w; ties -> vertex adjacent to a leaf, then lower degree, then lower weight, then index.
                       Supports of leaves are already forced (preselected) by the degree-1 rule, so on graphs with leaves
                       the leaf rule only matters when it is switched off; in practice the tie-break is degree, weight.
greedy_lookahead       score(v) = gain(v)/w(v) + max_{u != v} gain'(u)/w(u), where gain' is the gain after adding v
                       (one-step lookahead over every candidate). Ties: lower weight, higher degree, index.
greedy_random_restart  restart 0 = weighted greedy; restarts >= 1 multiply each ratio by (1 + eps * U(-1, 1)) (fixed
                       per vertex and restart) and break ties by a random rank; repeated until `budget` seconds, the
                       cheapest set after RR (and LS) is kept.
lp_rounding            LP relaxation min w.x, sum_{u in N(v)} x_u >= 1, 0 <= x <= 1 (forced x = 1), solved by HiGHS;
                       vertices are added in decreasing x (ties: higher gain/w) until a TDS is formed, then RR.

All gains only decrease during construction and every score is non-decreasing in the gain, so the lazy max-heap of
greedy.py stays exact (a popped entry whose stored key equals the current key is a true maximum).
"""
from __future__ import annotations

import heapq
import time

import numpy as np

from ..graph import Graph
from ..reductions import preselected_vertices
from ..result import SolveResult


class _State:
    def __init__(self, graph: Graph):
        graph.require_tds_exists()
        self.g = graph
        self.adj = graph.adj
        self.n = graph.n
        self.deg = graph.degrees.tolist()
        self.w = graph.w.tolist()
        self.covered = [False] * self.n
        self.selected = [False] * self.n
        self.gain = self.deg[:]
        self.n_uncovered = self.n
        self.order, self.scores = [], []
        for v in preselected_vertices(graph).tolist():
            self.add(v, float("inf"))

    def add(self, v, score):
        self.selected[v] = True
        self.order.append(v)
        self.scores.append(score)
        for u in self.adj[v]:
            if not self.covered[u]:
                self.covered[u] = True
                self.n_uncovered -= 1
                for x in self.adj[u]:
                    self.gain[x] -= 1

    def ratio(self, v):
        return self.gain[v] / self.w[v] if self.w[v] > 0 else float("inf")


def _lazy(st: _State, key):
    """Add argmin key(v) (key must be non-increasing in priority as gains drop) until S is a TDS."""
    heap = [key(v) for v in range(st.n) if not st.selected[v] and st.gain[v] > 0]
    heapq.heapify(heap)
    while st.n_uncovered > 0:
        k = heapq.heappop(heap)
        v = k[-1]
        if st.selected[v] or st.gain[v] == 0:
            continue
        cur = key(v)
        if cur != k:
            heapq.heappush(heap, cur)
            continue
        st.add(v, float(st.gain[v]))
    return st


def _finish(graph, st, t0, name, ls_time, seed, **info):
    from ..postprocess import local_search, redundancy_removal
    S = redundancy_removal(graph, st.order, scores=dict(zip(st.order, st.scores)))
    if ls_time:
        S = local_search(graph, S, time_limit=ls_time, seed=seed)
    return SolveResult.build(graph, S, time.perf_counter() - t0, name + ("_ls" if ls_time else ""), **info)


def tiebreak_construct(graph: Graph) -> _State:
    st = _State(graph)
    leafadj = [any(st.deg[u] == 1 for u in st.adj[v]) for v in range(st.n)]
    return _lazy(st, lambda v: (-st.ratio(v), not leafadj[v], st.deg[v], st.w[v], v))


def greedy_tiebreak(graph: Graph, ls_time: float = 0.0, seed: int = 0) -> SolveResult:
    t0 = time.perf_counter()
    return _finish(graph, tiebreak_construct(graph), t0, "greedy_tiebreak_rr", ls_time, seed)


def lookahead_construct(graph: Graph) -> _State:
    st = _State(graph)
    adj, w = st.adj, st.w
    while st.n_uncovered > 0:
        cand = [v for v in range(st.n) if not st.selected[v] and st.gain[v] > 0]
        by_ratio = sorted(cand, key=lambda u: -st.ratio(u))
        best_key, best_v = None, None
        for v in cand:
            newly = [u for u in adj[v] if not st.covered[u]]
            if len(newly) == st.n_uncovered:            # v alone finishes the cover: nothing left to look ahead to
                after = 0.0
            else:
                drop = {}
                for u in newly:
                    for x in adj[u]:
                        drop[x] = drop.get(x, 0) + 1
                after = 0.0
                for u in by_ratio:                       # best unaffected candidate bounds the rest from below
                    if u == v:
                        continue
                    if u not in drop:
                        after = max(after, st.ratio(u))
                        break
                for x, d in drop.items():
                    if x != v and not st.selected[x] and st.gain[x] - d > 0:
                        after = max(after, (st.gain[x] - d) / w[x] if w[x] > 0 else float("inf"))
            k = (-(st.ratio(v) + after), w[v], -st.deg[v], v)
            if best_key is None or k < best_key:
                best_key, best_v = k, v
        st.add(best_v, float(st.gain[best_v]))
    return st


def greedy_lookahead(graph: Graph, ls_time: float = 0.0, seed: int = 0) -> SolveResult:
    t0 = time.perf_counter()
    return _finish(graph, lookahead_construct(graph), t0, "greedy_lookahead_rr", ls_time, seed)


def noisy_construct(graph: Graph, rng: np.random.Generator, eps: float) -> _State:
    st = _State(graph)
    mult = (1.0 + eps * rng.uniform(-1.0, 1.0, st.n)).tolist()
    rank = rng.permutation(st.n).tolist()
    return _lazy(st, lambda v: (-st.ratio(v) * mult[v], rank[v], v))


def greedy_random_restart(graph: Graph, budget: float = 0.16, eps: float = 0.05, ls_time: float = 0.0,
                          seed: int = 0, max_restarts: int = None) -> SolveResult:
    """Restarts until `budget` seconds (at least one). Restart 0 is the plain weighted greedy, so the result is never
    worse than weighted greedy + RR (+ LS). With ls_time > 0 each restart gets min(ls_time, time left) of LS."""
    from ..baselines.greedy import greedy_construct
    from ..postprocess import local_search, redundancy_removal
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    best, best_w, k = None, float("inf"), 0
    w = graph.w
    while True:
        if k == 0:
            order, scores = greedy_construct(graph)
        else:
            st = noisy_construct(graph, rng, eps)
            order, scores = st.order, st.scores
        S = redundancy_removal(graph, order, scores=dict(zip(order, scores)), check=False)
        if ls_time:
            left = budget - (time.perf_counter() - t0)
            S = local_search(graph, S, time_limit=max(min(ls_time, left), 0.0), seed=seed + k)
        W = float(w[S].sum())
        if W < best_w - 1e-12:
            best, best_w = S, W
        k += 1
        if time.perf_counter() - t0 >= budget or (max_restarts and k >= max_restarts):
            break
    return SolveResult.build(graph, best, time.perf_counter() - t0,
                             "greedy_random_restart_rr" + ("_ls" if ls_time else ""), restarts=k)


def lp_values(graph: Graph) -> np.ndarray:
    from scipy.optimize import linprog
    from scipy.sparse import csr_matrix
    n = graph.n
    A = csr_matrix((np.ones(len(graph.indices)), graph.indices, graph.indptr), shape=(n, n))
    lb = np.zeros(n)
    lb[preselected_vertices(graph)] = 1.0
    res = linprog(c=graph.w, A_ub=-A, b_ub=-np.ones(n), bounds=list(zip(lb, np.ones(n))), method="highs")
    if res.status != 0:
        raise RuntimeError(f"LP relaxation failed: {res.message}")
    return res.x


def lp_construct(graph: Graph):
    st = _State(graph)
    x = lp_values(graph)
    lp_obj = float(graph.w @ x)
    order = sorted(range(st.n), key=lambda v: (-round(x[v], 9), -st.ratio(v), v))
    for v in order:
        if st.n_uncovered == 0:
            break
        if not st.selected[v] and st.gain[v] > 0:
            st.add(v, float(x[v]))
    return st, lp_obj


def lp_rounding(graph: Graph, ls_time: float = 0.0, seed: int = 0) -> SolveResult:
    t0 = time.perf_counter()
    st, lp_obj = lp_construct(graph)
    return _finish(graph, st, t0, "lp_rounding_rr", ls_time, seed, lp_bound=lp_obj)


METHODS = {"greedy_tiebreak": greedy_tiebreak, "greedy_lookahead": greedy_lookahead,
           "greedy_random_restart": greedy_random_restart, "lp_rounding": lp_rounding}
