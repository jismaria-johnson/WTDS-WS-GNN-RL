"""Revision step 3: every stronger greedy variant returns a valid TDS on 200 random graphs (4 classes x 4 schemes,
n = 10-80, train-range seeds), with W(S) recomputed; random restart is never worse than greedy + RR; small cases agree
with brute force on being >= the optimum."""
import numpy as np
import pytest

from wtdsws.baselines import brute_force_weighted, greedy_rr
from wtdsws.baselines.greedy_plus import (greedy_lookahead, greedy_random_restart, greedy_tiebreak,
                                          lp_rounding, lp_values)
from wtdsws.checker import is_total_dominating_set, verify
from wtdsws.generators import make_graph
from wtdsws.graph_classes import CLASSES, class_spec
from wtdsws.postprocess import is_minimal
from wtdsws.weights import SCHEMES, weighted_graph


def graphs(count=200, seed=4242):
    rng = np.random.default_rng(seed)
    for i in range(count):
        cls = CLASSES[i % 4]
        sch = SCHEMES[(i // 4) % 4]
        n = int(rng.integers(10, 81))
        spec = class_spec(cls, n, rng)
        spec["seed"] = int(rng.integers(0, 1_000_000_000))
        yield weighted_graph(make_graph(spec), sch, int(rng.integers(0, 1_000_000_000)))


@pytest.mark.parametrize("fn", [greedy_tiebreak, greedy_lookahead, lp_rounding,
                                lambda g: greedy_random_restart(g, budget=0.01),
                                lambda g: greedy_tiebreak(g, ls_time=0.05),
                                lambda g: lp_rounding(g, ls_time=0.05),
                                lambda g: greedy_random_restart(g, budget=0.02, ls_time=0.01)],
                         ids=["tiebreak", "lookahead", "lp", "restart", "tiebreak_ls", "lp_ls", "restart_ls"])
def test_valid_on_200_graphs(fn):
    for g in graphs():
        r = fn(g)
        S = sorted(set(int(v) for v in r.vertices))
        assert verify(g, S) and is_total_dominating_set(g.adj, S), g.name
        assert abs(float(g.w[S].sum()) - r.weight) < 1e-9
        assert r.is_valid


def test_rr_output_minimal_and_restart_not_worse():
    for g in list(graphs(60, seed=7)):
        for fn in (greedy_tiebreak, greedy_lookahead, lp_rounding):
            assert is_minimal(g, fn(g).vertices)
        assert greedy_random_restart(g, budget=0.01).weight <= greedy_rr(g).weight + 1e-9


def test_lp_bound_and_brute_force():
    rng = np.random.default_rng(3)
    for i in range(30):
        spec = class_spec(CLASSES[i % 4], int(rng.integers(6, 13)), rng)
        spec["seed"] = int(rng.integers(0, 1_000_000_000))
        g = weighted_graph(make_graph(spec), SCHEMES[i % 4], i)
        opt = brute_force_weighted(g)[0]
        assert float(g.w @ lp_values(g)) <= opt + 1e-6               # LP relaxation is a lower bound
        for fn in (greedy_tiebreak, greedy_lookahead, lp_rounding):
            assert fn(g).weight >= opt - 1e-9
