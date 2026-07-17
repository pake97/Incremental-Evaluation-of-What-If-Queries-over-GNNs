"""Exploration layer: composition semantics, trace generation, and the
three-rung reuse engine (every rung exact on every state — Golden rule)."""
import os

import pytest
import torch

from src.data.graph_state import GraphState, Intervention
from src.data.loader import load_graph
from src.eval.exactness import assert_exact
from src.explore.state import ConflictError, Registry, compose, conflicts
from src.explore.trace import Trace, TraceConfig, generate
from src.explore.reuse import run_trace
from src.forward.forward import set_determinism, whatif_forward
from src.models.model import load_model
from src.paths import CHECKPOINTS, FINBENCH_PROC

TR = ("account", "transfer", "account")

pytestmark = pytest.mark.skipif(
    not os.path.exists(os.path.join(FINBENCH_PROC, "finbench_hetero.pt")),
    reason="processed FinBench graph not present",
)


@pytest.fixture(scope="module")
def env():
    set_determinism()
    data, schema = load_graph("finbench")
    return data, schema, GraphState(data)


def _reg(data):
    reg = Registry()
    d1 = torch.zeros_like(data["account"].x); d1[3, 0] = 1.0
    d2 = torch.zeros_like(data["account"].x); d2[3, 0] = 2.0; d2[5, 1] = 1.0
    reg.add(Intervention(feat_deltas={"account": d1}))
    reg.add(Intervention(feat_deltas={"account": d2}))
    reg.add(Intervention(del_cols={TR: torch.tensor([0, 1])}))
    reg.add(Intervention(del_cols={TR: torch.tensor([1, 2])}))  # conflicts with 2
    reg.add(Intervention(add_edges={TR: torch.tensor([[1], [2]])}))
    return reg


def test_composition_is_order_independent(env):
    data, schema, gs = env
    reg = _reg(data)
    a = compose(reg, [0, 1, 2])
    b = compose(reg, [2, 1, 0])
    assert torch.equal(a.feat_deltas["account"], b.feat_deltas["account"])
    assert torch.equal(a.del_cols[TR], b.del_cols[TR])
    # the resulting graphs are identical
    va, vb = gs.apply(a), gs.apply(b)
    assert torch.equal(va.x_dict["account"], vb.x_dict["account"])
    assert torch.equal(va.edge_index_dict[TR], vb.edge_index_dict[TR])


def test_conflicting_sets_rejected(env):
    data, _, _ = env
    reg = _reg(data)
    assert conflicts(reg, frozenset([2]), 3)
    assert not conflicts(reg, frozenset([2]), 4)
    with pytest.raises(ConflictError):
        compose(reg, [2, 3])


def test_trace_generation_structure(env):
    data, schema, gs = env
    cfg = TraceConfig(n_states=40, share_rate=0.5, backtrack_rate=0.4,
                      node_type="account", relation=TR, seed=1)
    trace = generate(schema, gs, cfg)
    s = trace.structure()
    assert s["n_states"] == 40
    assert s["max_depth"] <= cfg.max_depth
    assert s["n_dag_edges"] >= s["n_states"]
    # every visited state's composition is conflict-free by construction
    for st in trace.order:
        compose(trace.registry, st)
    # same seed -> same trace
    t2 = generate(schema, gs, cfg)
    assert t2.order == trace.order


@pytest.mark.parametrize("arch", ["rgcn", "hgat"])
def test_all_rungs_exact_on_trace(env, arch):
    """The reuse rung resumes from ancestor caches across a real DAG — every
    state of every rung must equal from-scratch on the same G^D."""
    data, schema, gs = env
    path = os.path.join(CHECKPOINTS, "finbench", f"nodelabel_{arch}_L2.pt")
    if not os.path.exists(path):
        pytest.skip("checkpoint missing")
    model = load_model(path)
    cfg = TraceConfig(n_states=12, share_rate=0.5, backtrack_rate=0.4,
                      cohort_size=16, topo_edits=4,
                      node_type="account", relation=TR, seed=3)
    trace = generate(schema, gs, cfg)
    for rung in ("incremental", "reuse"):
        rows = run_trace(model, gs, schema, trace, rung, check_every=1)
        assert all(r["exact_pass"] for r in rows), \
            f"{rung} rung not exact: {[r for r in rows if not r['exact_pass']]}"


def test_reuse_resumes_from_ancestors(env):
    data, schema, gs = env
    path = os.path.join(CHECKPOINTS, "finbench", "nodelabel_rgcn_L2.pt")
    if not os.path.exists(path):
        pytest.skip("checkpoint missing")
    model = load_model(path)
    cfg = TraceConfig(n_states=20, share_rate=0.6, backtrack_rate=0.3,
                      cohort_size=16, topo_edits=4,
                      node_type="account", relation=TR, seed=5)
    trace = generate(schema, gs, cfg)
    rows = run_trace(model, gs, schema, trace, "reuse")
    # at depth >= 2 the runner should find materialized non-root ancestors
    deep = [r for r in rows if r["depth"] >= 2]
    assert deep and any(r["ancestor_depth"] > 0 for r in deep)
    # residual work strictly smaller than replaying the full set for those
    assert all(r["residual_size"] <= r["depth"] for r in rows)
