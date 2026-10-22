"""Revision 2, step 1e: guided restarts (and both controls) always return a valid TDS on 200 random graphs (4 classes
x 4 schemes, n = 10-80, train-range seeds) with W(S) recomputed; beta = 0 reproduces plain_restarts for the same seed;
restart 0 of plain_restarts is weighted greedy + RR; every anytime trace is valid and strictly improving."""
import sys
from pathlib import Path

import numpy as np
import pytest

from wtdsws.baselines import greedy_rr
from wtdsws.baselines.guided_restarts import (guided_construct, guided_restarts, plain_restarts,
                                              random_prior_restarts, rank_prior)
from wtdsws.baselines.greedy import greedy_construct
from wtdsws.checker import is_total_dominating_set, verify
from wtdsws.solve import get_model

from test_greedy_plus import graphs

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


@pytest.fixture(scope="module")
def model():
    return get_model(ROOT / "checkpoints" / "best.pt", "cpu")


def _valid(g, r):
    S = sorted(set(int(v) for v in r.vertices))
    assert verify(g, S) and is_total_dominating_set(g.adj, S), g.name
    assert abs(float(g.w[S].sum()) - r.weight) < 1e-9
    assert r.is_valid


@pytest.mark.parametrize("refresh", [0.0, 0.1])
def test_guided_valid_on_200_graphs(model, refresh):
    for i, g in enumerate(graphs()):
        _valid(g, guided_restarts(g, model, "cpu", budget=1e9, beta=2.0, tau=0.1, refresh=refresh,
                                  ls_time=0.01, seed=i, max_restarts=3))


def test_controls_valid_on_200_graphs():
    for i, g in enumerate(graphs()):
        _valid(g, random_prior_restarts(g, budget=1e9, beta=2.0, tau=0.1, refresh=0.1, ls_time=0.01, seed=i,
                                        max_restarts=3))
        _valid(g, plain_restarts(g, budget=1e9, tau=0.1, ls_time=0.01, seed=i, max_restarts=3))


def test_beta0_matches_plain(model):
    for i, g in enumerate(graphs(60, seed=11)):
        for ls in (0.0, 0.01):
            a = guided_restarts(g, model, "cpu", budget=1e9, beta=0.0, tau=0.2, refresh=0.1, ls_time=ls, seed=i,
                                max_restarts=5)
            b = plain_restarts(g, budget=1e9, tau=0.2, ls_time=ls, seed=i, max_restarts=5)
            assert a.vertices == b.vertices and a.info["restarts"] == b.info["restarts"] == 5


def test_restart0_is_greedy_and_never_worse():
    for g in graphs(60, seed=7):
        st = guided_construct(g, np.zeros(g.n), 0.0, 0.0, np.zeros(g.n))
        assert st.order == greedy_construct(g)[0]
        assert plain_restarts(g, budget=1e9, tau=0.1, ls_time=0.0, max_restarts=4).weight <= greedy_rr(g).weight + 1e-9


def test_rank_prior():
    p = rank_prior(np.array([3.0, -np.inf, 1.0, 2.0, 2.0]))
    assert p.tolist() == [1.0, 0.0, 0.0, 0.5, 0.5]
    assert rank_prior(np.array([-np.inf, 5.0])).tolist() == [0.0, 1.0]


def test_anytime_traces_valid(model):
    import rev2_anytime as A
    for i, g in enumerate(graphs(8, seed=5)):
        runs = [A.restarts_run(g, "gnn", model, "cpu", 0.05, dict(beta=1.0, tau=0.1, refresh=0.1), i),
                A.restarts_run(g, "random", None, "cpu", 0.05, dict(beta=1.0, tau=0.1), i),
                A.restarts_run(g, "none", None, "cpu", 0.05, dict(tau=0.1), i),
                A.ours_greedy_ls(g, model, "cpu", 0.05), A.ours_bok_ls(g, model, "cpu", 0.05, np.random.default_rng(i)),
                A.greedy_ls(g, 0.05), A.lp_ls(g, 0.05), A.cpsat(g, 1.0)]
        for tr, _ in runs:
            out = A.verify_trace(g, tr, "trace")
            assert out and A.at_budget(out, 1e9)[0] == out[-1][1]
