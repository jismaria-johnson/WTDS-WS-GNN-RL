"""ILP = brute force, all solvers valid, checkpoint loading, input errors (Section 7)."""
import math
from pathlib import Path

import networkx as nx
import numpy as np
import pytest
import torch

from wtdsws import solve
from wtdsws.baselines import brute_force_weighted, greedy, greedy_rr, greedy_rr_ls, ilp
from wtdsws.checker import is_total_dominating_set
from wtdsws.graph import InvalidWeightError, NoTotalDominatingSetError

from ws_helpers import class_scheme_graphs

ROOT = Path(__file__).resolve().parents[1]
PRETRAINED = [ROOT / "checkpoints" / "pretrained" / f"wtds_seed{s}_best.pt" for s in (0, 1)]


def test_weighted_ilp_equals_brute_force_200():
    graphs = class_scheme_graphs(200, seed=21, n_min=3, n_max=10)
    for g in graphs:
        assert g.n <= 10
        bw, _, sets = brute_force_weighted(g)
        e = ilp(g, time_limit=10)
        assert e.info["optimal"], g.meta
        assert e.weight == bw, (g.meta, e.weight, bw)
        assert sorted(e.vertices) in [sorted(S) for S in sets]
        assert is_total_dominating_set(g.adj, e.vertices)


def test_all_solvers_valid_300():
    graphs = class_scheme_graphs(300, seed=22, n_min=4, n_max=60)
    for i, g in enumerate(graphs):
        rs = [greedy(g), greedy_rr(g), greedy_rr_ls(g, ls_time=0.05, seed=i),
              solve(g, None, method="rl", ls_time=0.0),
              solve(g, None, method="rl", ls_time=0.05),
              solve(g, None, method="wtds_zero_shot", ls_time=0.0)]
        if i % 5 == 0:
            rs.append(ilp(g, time_limit=5))
        for r in rs:
            assert r.is_valid and is_total_dominating_set(g.adj, r.vertices), (r.method, g.meta)
            assert r.weight == pytest.approx(float(g.w[r.vertices].sum()))       # W(S) recomputed


@pytest.mark.parametrize("path", PRETRAINED, ids=["seed0", "seed1"])
def test_pretrained_checkpoint_loads_strict(path):
    from wtdsws.model import QNet
    assert path.exists()
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = QNet()                                                     # the default 3-layer MPNN, hidden 64
    res = m.load_state_dict(ck["online"], strict=True)
    assert list(res.missing_keys) == [] and list(res.unexpected_keys) == []
    assert ck["model_cfg"] == dict(hidden=64, layers=3, encoder="mpnn", readout="meanmax")
    res = m.load_state_dict(ck["target"], strict=True)
    assert list(res.missing_keys) == [] and list(res.unexpected_keys) == []


def test_isolated_vertex_raises():
    G = nx.path_graph(4)
    G.add_node(9)
    for method in ("rl", "greedy_rr", "ilp"):
        with pytest.raises(NoTotalDominatingSetError):
            solve(G, None, method=method)


@pytest.mark.parametrize("bad", [-1.0, 0.0, float("nan"), float("inf")])
def test_bad_user_weights_raise(bad):
    G = nx.cycle_graph(5)
    w = {v: 10 for v in G.nodes()}
    w[2] = bad
    with pytest.raises(InvalidWeightError, match="vertex 2"):
        solve(G, w, method="greedy_rr")


def test_bad_weight_file_raises(tmp_path):
    from wtdsws.io import read_graph
    e = tmp_path / "g.txt"
    e.write_text("a b\nb c\nc a\n")
    for bad in ("0", "-3", "nan"):
        wf = tmp_path / f"w{bad}.txt"
        wf.write_text(f"a 5\nb {bad}\nc 2\n")
        with pytest.raises(InvalidWeightError, match="vertex b"):    # reader (< 0, NaN) or solver (0)
            solve(read_graph(e, wf), None, method="greedy_rr")


def test_cli_scheme_and_custom_weights(tmp_path):
    import json
    import subprocess
    import sys
    e = tmp_path / "edges.txt"
    e.write_text("1 2\n2 3\n3 4\n4 1\n1 5\n2 5\n2 2\n1 2\n")     # house + a self-loop + a duplicate
    out = tmp_path / "s.json"
    r = subprocess.run([sys.executable, "-m", "wtdsws.solve", "--graph", str(e), "--scheme", "inverse", "--seed", "0",
                        "--method", "ilp", "--out", str(out)], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "self-loop" in r.stdout and "duplicate" in r.stdout
    d = json.loads(out.read_text())
    assert d["weight"] == 134 and sorted(d["vertex_labels"]) == ["1", "2"] and d["is_valid"]
    wf = tmp_path / "w.txt"
    wf.write_text("1 1\n2 1\n3 1\n4 1\n5 1\n")
    r = subprocess.run([sys.executable, "-m", "wtdsws.solve", "--graph", str(e), "--weights", str(wf),
                        "--method", "rl", "--out", str(out), "--plot", str(tmp_path / "p.png")],
                       cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    d = json.loads(out.read_text())
    assert d["is_valid"] and d["weight"] == 2 and (tmp_path / "p.png").exists()
