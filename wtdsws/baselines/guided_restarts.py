"""GNN-guided randomized restart greedy (revision 2, step 1) and its two controls. All three share ONE code path;
only the prior p(v) in [0, 1] changes:

ours_guided_restarts   p = rank of the trained GNN's Q(s0, v) over the valid vertices (one forward pass on the
                       initial state; optionally refreshed from the partial solution every `refresh` picks)
random_prior_restarts  p = a random permutation rank, redrawn for every restart (and at every refresh point)
plain_restarts         beta = 0 (the prior is ignored): weighted greedy with multiplicative noise only

Each restart builds a TDS greedily with score(v) = (gain(v) / w(v)) * exp(beta * p(v)) * exp(tau * eps_v),
eps_v ~ N(0, 1) drawn once per restart (restart 0 uses eps = 0, so plain restart 0 is exactly greedy_construct), then
applies RR. Ties: lower weight, higher degree, lower index (as greedy.py). The multiplier is fixed between refresh
points and gains only decrease, so the lazy heap of greedy_plus stays exact; the heap is rebuilt after a refresh.

Anytime LS: whenever a restart produces a new best RR solution, the same LS as greedy + RR + LS is applied to it,
limited to min(ls_time, time left); the returned set is the cheapest of all RR and LS sets. At the end of the budget
this equals "LS on the best restart solution" (the spec), and the result is valid at every point in time.

eps is drawn from rng([seed, 0]) and random priors from rng([seed, 1]), so the three variants see the SAME noise for
the same seed (common random numbers); with beta = 0 the guided variant reproduces plain_restarts exactly.
"""
from __future__ import annotations

import heapq
import math
import time

import numpy as np

from ..graph import Graph
from ..result import SolveResult
from .greedy_plus import _State


def rank_prior(q: np.ndarray) -> np.ndarray:
    """Rank-normalised prior: finite entries -> (rank - 1) / (count - 1) in [0, 1] (ties averaged), others 0."""
    p = np.zeros(len(q))
    ok = np.isfinite(q)
    k = int(ok.sum())
    if k == 1:
        p[ok] = 1.0
    elif k > 1:
        _, inv, cnt = np.unique(q[ok], return_inverse=True, return_counts=True)
        avg = np.cumsum(cnt) - cnt + (cnt - 1) / 2.0          # 0-based average rank of each distinct value
        p[ok] = avg[inv] / (k - 1.0)
    return p


class GNNPrior:
    """p(v) from Q(s, v) of a trained model; s = the initial state or the current partial solution."""

    def __init__(self, model, graph: Graph, device="cpu"):
        from ..env import TDSEnv
        self.model, self.device = model, device
        self.env = TDSEnv(graph, **dict(getattr(model, "env_kwargs", {}) or {}))
        self.forward_passes = 0

    def __call__(self, st: _State = None) -> np.ndarray:
        from ..rollout import q_values
        if st is not None:
            self.env.set_state(st.order)
        self.forward_passes += 1
        return rank_prior(q_values(self.model, [self.env], self.device)[0])


def guided_construct(graph: Graph, p: np.ndarray, beta: float, tau: float, eps: np.ndarray,
                     refresh_every: int = 0, prior_fn=None) -> _State:
    st = _State(graph)
    mult = []

    def set_mult(p):
        mult[:] = np.exp(beta * p + tau * eps).tolist() if (beta or tau) else [1.0] * st.n

    def key(v):
        return (-st.ratio(v) * mult[v], st.w[v], -st.deg[v], v)

    def heap():
        h = [key(v) for v in range(st.n) if not st.selected[v] and st.gain[v] > 0]
        heapq.heapify(h)
        return h

    set_mult(p)
    h, picks = heap(), 0
    while st.n_uncovered > 0:
        k = heapq.heappop(h)
        v = k[-1]
        if st.selected[v] or st.gain[v] == 0:
            continue
        cur = key(v)
        if cur != k:
            heapq.heappush(h, cur)
            continue
        st.add(v, float(st.gain[v]))
        picks += 1
        if refresh_every and prior_fn is not None and picks % refresh_every == 0 and st.n_uncovered > 0:
            set_mult(prior_fn(st))
            h = heap()
    return st


def restarts(graph: Graph, prior: str = "gnn", model=None, device="cpu", budget: float = 1.0, beta: float = 1.0,
             tau: float = 0.1, refresh: float = 0.0, ls_time: float = 1.0, seed: int = 0, max_restarts: int = None,
             trace: bool = False) -> SolveResult:
    """prior in {"gnn", "random", "none"}; refresh = fraction of n between prior refreshes (0 = never).
    trace=True stores info["trace"] = [(elapsed s, W, S)] for every new best set (S unverified; the caller checks).
    Restarts run until `budget` seconds have passed (at least one restart)."""
    from ..postprocess import local_search, redundancy_removal
    if prior not in ("gnn", "random", "none"):
        raise ValueError(prior)
    t0 = time.perf_counter()
    n, w = graph.n, graph.w
    if prior == "none":
        beta = 0.0
    rng_eps, rng_pri = np.random.default_rng([seed, 0]), np.random.default_rng([seed, 1])
    refresh_every = max(1, math.ceil(refresh * n)) if (refresh and beta) else 0
    gnn = None
    if prior == "gnn" and beta:
        gnn = GNNPrior(model, graph, device)
        p0 = gnn()                                        # Q(s0, v) once; reused by every restart
    tr, best, best_w, best_rr_w, k, n_ls = [], None, float("inf"), float("inf"), 0, 0

    def consider(S):
        nonlocal best, best_w
        W = float(w[S].sum())
        if W < best_w - 1e-12:
            best, best_w = S, W
            if trace:
                tr.append((time.perf_counter() - t0, W, list(S)))

    while True:
        eps = np.zeros(n) if k == 0 else rng_eps.standard_normal(n)
        if prior == "gnn":
            p, fn = (p0, gnn) if gnn is not None else (np.zeros(n), None)
        elif prior == "random":
            p = rng_pri.permutation(n) / max(n - 1, 1)
            fn = lambda st: rng_pri.permutation(n) / max(n - 1, 1)   # noqa: E731
        else:
            p, fn = np.zeros(n), None
        st = guided_construct(graph, p, beta, tau, eps, refresh_every, fn)
        S = redundancy_removal(graph, st.order, scores=dict(zip(st.order, st.scores)), check=False)
        W = float(w[S].sum())
        consider(S)
        if ls_time and W < best_rr_w - 1e-12:           # new best restart solution -> LS on it
            best_rr_w = W
            left = budget - (time.perf_counter() - t0)
            if left > 0:
                consider(local_search(graph, S, time_limit=min(ls_time, left), seed=seed + k))
                n_ls += 1
        k += 1
        if time.perf_counter() - t0 >= budget or (max_restarts and k >= max_restarts):
            break
    name = {"gnn": "ours_guided_restarts", "random": "random_prior_restarts", "none": "plain_restarts"}[prior]
    info = dict(restarts=k, ls_calls=n_ls, beta=beta, tau=tau, refresh=refresh,
                forward_passes=gnn.forward_passes if gnn else 0)
    if trace:
        info["trace"] = tr
    return SolveResult.build(graph, best, time.perf_counter() - t0, name, **info)


def guided_restarts(graph, model, device="cpu", **kw) -> SolveResult:
    return restarts(graph, "gnn", model, device, **kw)


def random_prior_restarts(graph, **kw) -> SolveResult:
    return restarts(graph, "random", **kw)


def plain_restarts(graph, **kw) -> SolveResult:
    kw.pop("beta", None)
    return restarts(graph, "none", **kw)
