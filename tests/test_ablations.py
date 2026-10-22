"""Ablation switches (A1 no mask, A2 degree-1 rule off, A3 unit reward, A4 sum readout, A5 scratch):
every switch gives a valid TDS after RR on 100 random graphs (W recomputed); each ablation config differs from the
reference in exactly the intended setting; with every switch OFF the evaluation path reproduces the reference."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from wtdsws.checker import is_total_dominating_set, verify
from wtdsws.config import load_config
from wtdsws.model import QNet
from wtdsws.postprocess import redundancy_removal
from wtdsws.reductions import forced_vertices
from wtdsws.rollout import rollout

from test_greedy_plus import graphs

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
ABL = ROOT / "configs" / "ablations"


def flat(d, p=""):
    out = {}
    for k, v in d.items():
        out.update(flat(v, p + k + ".") if isinstance(v, dict) else {p + k: v})
    return out


def diff(a, b):
    return {k for k in set(a) | set(b) if a.get(k) != b.get(k) and not k.startswith("out.")}


def test_configs_change_exactly_one_thing():
    ref = flat(load_config(ABL / "ref.yaml"))
    assert diff(ref, flat(load_config(ROOT / "configs" / "ws_finetune_seed1.yaml"))) == set()
    expect = {"a1_nomask": {"env.mask_mode"}, "a2_noforced": {"env.use_forced"}, "a3_unit": {"env.reward_mode"},
              "a4_sum": {"model.readout", "train.init_keys"}, "a5_scratch": {"train.init_checkpoint"}}
    for name, keys in expect.items():
        assert diff(ref, flat(load_config(ABL / f"{name}.yaml"))) == keys, name


@pytest.mark.parametrize("env,readout", [({"mask_mode": "notin"}, "meanmax"), ({"use_forced": False}, "meanmax"),
                                         ({"reward_mode": "unit"}, "meanmax"), ({}, "sum"), ({}, "meanmax")],
                         ids=["a1_nomask", "a2_noforced", "a3_unit", "a4_sum", "reference_arch"])
def test_switch_valid_after_rr_on_100_graphs(env, readout):
    torch.manual_seed(0)
    model = QNet(readout=readout).eval()
    for g in graphs(100, seed=99):
        r = rollout(model, [g], "cpu", env_kwargs=env, multi_select="auto")[0]
        S = redundancy_removal(g, r["solution"], scores=r["q"], protected=r["forced"])
        assert verify(g, S) and is_total_dominating_set(g.adj, S), g.name
        assert verify(g, r["solution"])
        assert abs(float(g.w[S].sum()) - sum(g.w[v] for v in S)) < 1e-9
        if env.get("use_forced") is False:
            assert r["forced"] == [v for v in r["forced"] if g.w[v] == 0]      # only free vertices pre-selected
        if not env.get("mask_mode"):                                            # gain mask: no wasted picks
            import abl_eval
            assert abl_eval.wasted_and_forced(g, r, True)[0] == 0
        assert set(forced_vertices(g).tolist()) <= set(r["solution"])          # any TDS contains them


def test_a4_partial_init_loads_encoder_only(tmp_path):
    from wtdsws.agent import DDQNAgent
    ck = torch.load(ROOT / "checkpoints" / "pretrained" / "wtds_seed0_best.pt", map_location="cpu", weights_only=False)
    cfg = load_config(ABL / "a4_sum.yaml")
    agent = DDQNAgent(cfg["model"], device="cpu")
    enc = {k: w for k, w in ck["online"].items() if k.startswith(("inp.", "layers."))}
    res = agent.online.load_state_dict(enc, strict=False)
    assert not res.unexpected_keys and set(res.missing_keys) == {k for k in agent.online.state_dict()
                                                                  if k.startswith(("w5", "w6", "w7"))}
    assert agent.online.w6.in_features == 64 + 3
    assert torch.equal(agent.online.inp.weight, ck["online"]["inp.weight"])


def test_switches_off_reproduce_reference():
    import pandas as pd
    import abl_eval
    from run_eval import checked
    from wtdsws.solve import get_model
    grid = pd.read_csv(ROOT / "results" / "raw_grid.csv")
    grid = grid[grid.method == "ours_rr"].set_index(["cls", "scheme", "n", "gidx", "model_seed"]).weight
    items = abl_eval.instances((20, 50))[::8]
    for s in (0, 1):
        model = get_model(abl_eval.ckpt("ref", s), "cpu")
        ek = dict(model.env_kwargs)
        assert ek["mask_mode"] == "gain" and ek["use_forced"] and ek["reward_mode"] == "weighted"
        for it in items:
            g = it["g"]
            r = rollout(model, [g], "cpu", env_kwargs=ek, multi_select="auto")[0]
            S = redundancy_removal(g, r["solution"], scores=r["q"], protected=r["forced"])
            _, W = checked(g, S, "ref")
            assert W == grid.loc[(it["cls"], it["scheme"], it["n"], it["gidx"], s)]
