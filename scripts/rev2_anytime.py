"""Revision 2: anytime harness. Every method runs ONCE up to its maximum budget and returns a trace
[(elapsed s, W, S)] of every new best valid set; results at a budget b = the last trace entry with elapsed <= b.

Timing: from the call (after one-time model loading) to each improvement; includes feature building, forward passes,
RR and LS. Verification is done AFTER the run (verify_trace: both TDS checkers, W recomputed and compared), so it is
never inside the timed region of any method.

Methods
  ours_greedy_ls   greedy policy rollout + RR, then LS in the remaining time (stops at a local optimum)
  ours_bok_ls      greedy rollout + RR (+ LS), then batches of 8 rollouts sampled from softmax(Q / 0.03), each + RR +
                   LS limited to min(1 s, time left) (rev_equal_time.ours_anytime). A new batch is not started when the
                   previous batch's duration (first batch: greedy rollout time x picks per pass) would overrun the budget.
  guided / random_prior / plain   wtdsws.baselines.guided_restarts (LS 1 s cap on each new best restart set)
  greedy_ls        weighted greedy + RR, then LS in the remaining time
  lp_ls            LP rounding + RR, then LS in the remaining time; LP by HiGHS interior point (much faster than the
                   default dual simplex at n >= 2000) with time limit = budget (not solved -> no set)
  cpsat            CP-SAT, 8 workers, random_seed 0, no warm start (as rev_equal_time); solution callback for the trace;
                   also returns its best bound and status
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402

from wtdsws.checker import is_total_dominating_set, verify  # noqa: E402

TEMPERATURE = 0.03
CPSAT_WORKERS = 8


def verify_trace(g, tr, method):
    """Both checkers on every set; W recomputed from S and compared with the recorded value."""
    out = []
    for t, W, S in tr:
        S = sorted(int(v) for v in set(S))
        if not (verify(g, S) and is_total_dominating_set(g.adj, S)):
            raise AssertionError(f"{method}: INVALID solution in trace on {g.name}")
        W2 = float(g.w[S].sum())
        if abs(W2 - W) > 1e-6:
            raise AssertionError(f"{method}: recorded W {W} != recomputed {W2}")
        out.append((float(t), W2, len(S)))
    for (_, a, _), (_, b, _) in zip(out, out[1:]):
        assert b < a + 1e-9, f"{method}: trace not improving"
    return out


def at_budget(tr, b):
    """(W, size) of the best set found within b seconds, or (nan, nan)."""
    best = (np.nan, np.nan)
    for t, W, sz in tr:
        if t <= b + 1e-9:
            best = (W, sz)
    return best


class _Trace:
    def __init__(self, g):
        self.g, self.t0, self.tr, self.best = g, time.perf_counter(), [], float("inf")

    def add(self, S):
        W = float(self.g.w[S].sum())
        if W < self.best - 1e-12:
            self.best = W
            self.tr.append((time.perf_counter() - self.t0, W, list(S)))

    def left(self, budget):
        return budget - (time.perf_counter() - self.t0)


def _ls_rest(g, T, S, budget, seed=0):
    from wtdsws.postprocess import local_search
    if T.left(budget) > 0:
        local_search(g, S, time_limit=T.left(budget), seed=seed, on_improve=T.add)


def ours_greedy_ls(g, model, device, budget):
    from wtdsws.postprocess import redundancy_removal
    from wtdsws.rollout import rollout
    T = _Trace(g)
    r = rollout(model, [g], device, env_kwargs=dict(getattr(model, "env_kwargs", {}) or {}), multi_select="auto")[0]
    S = redundancy_removal(g, r["solution"], scores=r["q"], protected=r["forced"], check=False)
    T.add(S)
    t_roll = time.perf_counter() - T.t0
    _ls_rest(g, T, S, budget)
    return T.tr, dict(rollout_rr_time=t_roll)


def ours_bok_ls(g, model, device, budget, rng):
    from wtdsws.postprocess import local_search, redundancy_removal
    from wtdsws.rollout import picks_per_forward, rollout
    ek = dict(getattr(model, "env_kwargs", {}) or {})
    T = _Trace(g)
    k, n_batches, last = 0, 0, 0.0

    def consider(rs):
        nonlocal k
        for r in rs:
            S = redundancy_removal(g, r["solution"], scores=r["q"], protected=r["forced"], check=False)
            T.add(S)
            if T.left(budget) > 0:
                T.add(local_search(g, S, time_limit=min(1.0, T.left(budget)), seed=k))
            k += 1

    consider(rollout(model, [g], device, env_kwargs=ek, multi_select="auto"))
    t_first = time.perf_counter() - T.t0
    last = t_first * picks_per_forward(g.n, "auto")     # a sampled rollout adds 1 vertex per pass: estimate
    while T.left(budget) > last:
        tb = time.perf_counter()
        consider(rollout(model, [g] * 8, device, env_kwargs=ek, mode="sample", temperature=TEMPERATURE,
                         rng=rng, batch_size=8))
        last = time.perf_counter() - tb
        n_batches += 1
    return T.tr, dict(candidates=k, batches=n_batches, first_rollout_time=t_first)


def restarts_run(g, prior, model, device, budget, params, seed):
    from wtdsws.baselines.guided_restarts import restarts
    r = restarts(g, prior, model, device, budget=budget, beta=params.get("beta", 0.0), tau=params["tau"],
                 refresh=params.get("refresh", 0.0), ls_time=1.0, seed=seed, trace=True)
    return r.info["trace"], dict(candidates=r.info["restarts"], ls_calls=r.info["ls_calls"],
                                 forward_passes=r.info["forward_passes"])


def greedy_ls(g, budget):
    from wtdsws.baselines.greedy import greedy_construct
    from wtdsws.postprocess import redundancy_removal
    T = _Trace(g)
    order, scores = greedy_construct(g)
    S = redundancy_removal(g, order, scores=dict(zip(order, scores)), check=False)
    T.add(S)
    _ls_rest(g, T, S, budget)
    return T.tr, {}


def lp_ls(g, budget):
    from wtdsws.baselines.greedy_plus import lp_construct
    from wtdsws.postprocess import redundancy_removal
    T = _Trace(g)
    try:
        st, lp_obj = lp_construct(g, method="highs-ipm", time_limit=budget)
    except RuntimeError:                                  # LP not solved within the budget: no set
        return T.tr, dict(lp_bound=None, lp_failed=True)
    S = redundancy_removal(g, st.order, scores=dict(zip(st.order, st.scores)), check=False)
    T.add(S)
    _ls_rest(g, T, S, budget)
    return T.tr, dict(lp_bound=lp_obj)


def cpsat(g, budget, workers=CPSAT_WORKERS, seed=0):
    """Same model as baselines.ilp._cpsat (forced x = 1), no hint; trace from a solution callback."""
    from ortools.sat.python import cp_model
    from wtdsws.baselines.ilp import _int_weights
    from wtdsws.reductions import forced_vertices
    t0 = time.perf_counter()
    m = cp_model.CpModel()
    x = [m.NewBoolVar(f"x{v}") for v in range(g.n)]
    for v in range(g.n):
        m.AddBoolOr([x[u] for u in g.adj[v]])
    for f in forced_vertices(g).tolist():
        m.Add(x[f] == 1)
    wi, scale = _int_weights(g)
    m.Minimize(sum(int(wi[v]) * x[v] for v in range(g.n)))
    tr = []

    class CB(cp_model.CpSolverSolutionCallback):
        def __init__(self):
            super().__init__()

        def on_solution_callback(self):
            t = time.perf_counter() - t0
            S = [v for v in range(g.n) if self.BooleanValue(x[v])]
            W = float(g.w[S].sum())
            if not tr or W < tr[-1][1] - 1e-12:
                tr.append((t, W, S))

    s = cp_model.CpSolver()
    s.parameters.max_time_in_seconds = float(budget) - (time.perf_counter() - t0)
    s.parameters.num_workers = int(workers)
    s.parameters.random_seed = int(seed)
    st = s.Solve(m, CB())
    return tr, dict(status=s.StatusName(st), bound=float(s.BestObjectiveBound()) / scale,
                    num_workers=int(workers), wall=time.perf_counter() - t0)
