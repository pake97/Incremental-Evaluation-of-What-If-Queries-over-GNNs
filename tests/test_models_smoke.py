"""Smoke: all 8 architectures forward on the real graph; checkpoints load; the
intervention editor's output feeds the encoder unchanged."""
import os

import pytest
import torch

from src.data.graph_state import GraphState, Intervention
from src.models.model import build_model, load_model
from src.paths import CHECKPOINTS, FINBENCH_PROC

TR = ("account", "transfer", "account")

pytestmark = pytest.mark.skipif(
    not os.path.exists(os.path.join(FINBENCH_PROC, "finbench_hetero.pt")),
    reason="processed FinBench graph not present",
)


@pytest.fixture(scope="module")
def data():
    return torch.load(os.path.join(FINBENCH_PROC, "finbench_hetero.pt"),
                      weights_only=False)


@pytest.mark.parametrize("mt", ["rgcn", "hgat"])
@pytest.mark.parametrize("L", [1, 2, 3, 4])
def test_forward_shapes(data, mt, L):
    m = build_model(data, model_type=mt, num_layers=L)
    m.eval()
    pairs = data[TR].edge_index[:, :128]
    with torch.no_grad():
        h = m.encode(data.x_dict, data.edge_index_dict)
        fr = m.classify_nodes(h)
        lp = m.link_score(h, pairs)
        lr = m.link_regress(h, pairs)
    assert fr.shape == (data["account"].num_nodes,)
    assert lp.shape == (128,)
    assert lr.shape == (128,)


def test_checkpoint_loads_and_runs(data):
    path = os.path.join(CHECKPOINTS, "finbench", "nodelabel_rgcn_L2.pt")
    if not os.path.exists(path):
        pytest.skip("checkpoint not present")
    m = load_model(path)
    with torch.no_grad():
        h = m.encode(data.x_dict, data.edge_index_dict)
    assert h["account"].shape == (data["account"].num_nodes, m.cfg["hidden"])


def test_intervention_view_feeds_encoder(data):
    gs = GraphState(data)
    m = build_model(data, "rgcn", 2)
    m.eval()
    view = gs.apply(Intervention(del_cols={TR: torch.arange(1000)}))
    with torch.no_grad():
        h = m.encode(view.x_dict, view.edge_index_dict)
    assert h["account"].shape[0] == data["account"].num_nodes
