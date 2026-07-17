"""Correctness of the copy-on-write intervention layer on the real FinBench graph.

Checks the invariants every incremental path relies on:
  * edits touch only the targeted relation / node type (untouched tensors SHARED),
  * the base graph is immutable (interventions are hypothetical),
  * pair -> column resolution finds the right base edges.
"""
import os

import pytest
import torch

from src.data.graph_state import GraphState, Intervention
from src.paths import FINBENCH_PROC

TR = ("account", "transfer", "account")
OTHER = ("account", "withdraw", "account")

pytestmark = pytest.mark.skipif(
    not os.path.exists(os.path.join(FINBENCH_PROC, "finbench_hetero.pt")),
    reason="processed FinBench graph not present",
)


@pytest.fixture(scope="module")
def data():
    return torch.load(os.path.join(FINBENCH_PROC, "finbench_hetero.pt"),
                      weights_only=False)


@pytest.fixture(scope="module")
def gs(data):
    return GraphState(data)


def test_delete_masks_only_target_relation(gs):
    E0 = gs.edge_index[TR].size(1)
    v = gs.apply(Intervention(del_cols={TR: torch.tensor([0, 5, 9])}))
    assert v.edge_index_dict[TR].size(1) == E0 - 3
    # untouched relations are SHARED (same object), not copied
    assert v.edge_index_dict[OTHER] is gs.edge_index[OTHER]
    # base graph unchanged (immutability)
    assert gs.edge_index[TR].size(1) == E0


def test_add_appends_edges_and_attr(gs):
    E0 = gs.edge_index[TR].size(1)
    v = gs.apply(Intervention(add_edges={TR: torch.tensor([[1, 2], [3, 4]])}))
    assert v.edge_index_dict[TR].size(1) == E0 + 2
    assert v.edge_attr_dict[TR].size(0) == E0 + 2


def test_feature_delta_touches_only_node_type(gs, data):
    d = torch.zeros_like(gs.x["account"])
    d[7] = 1.0
    v = gs.apply(Intervention(feat_deltas={"account": d}))
    assert torch.allclose(v.x_dict["account"][7], gs.x["account"][7] + 1.0)
    assert v.x_dict["company"] is gs.x["company"]
    # base intact
    assert torch.allclose(gs.x["account"][7], data["account"].x[7])


def test_pair_to_column_resolution(gs):
    E0 = gs.edge_index[TR].size(1)
    pairs = gs.edge_index[TR][:, [10, 20, 30]].t().tolist()
    cols = gs.cols_for_pairs(TR, pairs)
    v = gs.apply(Intervention(del_cols={TR: cols}))
    assert v.edge_index_dict[TR].size(1) == E0 - 3
