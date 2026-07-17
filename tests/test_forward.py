"""Ground-truth forward: deterministic, sensitive to interventions, schema-gated."""
import os

import pytest
import torch

from src.data.graph_state import GraphState, Intervention
from src.data.loader import load_graph
from src.data.schema import SchemaError
from src.eval.exactness import assert_exact, check
from src.forward.forward import encode, read_query, set_determinism, whatif_forward
from src.models.model import load_model
from src.paths import CHECKPOINTS, FINBENCH_PROC

TR = ("account", "transfer", "account")
CKPT = os.path.join(CHECKPOINTS, "finbench", "nodelabel_rgcn_L2.pt")

pytestmark = pytest.mark.skipif(
    not (os.path.exists(os.path.join(FINBENCH_PROC, "finbench_hetero.pt"))
         and os.path.exists(CKPT)),
    reason="processed FinBench graph or checkpoint not present",
)


@pytest.fixture(scope="module")
def env():
    set_determinism()
    data, schema = load_graph("finbench")
    return data, schema, GraphState(data), load_model(CKPT)


def test_forward_is_deterministic(env):
    data, schema, gs, model = env
    iv = Intervention(del_cols={TR: torch.arange(100)})
    h1 = whatif_forward(model, gs, iv, schema)
    h2 = whatif_forward(model, gs, iv, schema)
    rep = assert_exact(h1, h2)
    assert rep.max_abs_err == 0.0  # same code path twice must be bitwise equal


def test_intervention_changes_output(env):
    data, schema, gs, model = env
    h_base = encode(model, gs.base_view())
    d = torch.zeros_like(data["account"].x)
    d[:, 0] = 5.0
    h_iv = whatif_forward(model, gs, Intervention(feat_deltas={"account": d}), schema)
    assert not check(h_iv, h_base).passed  # a big Delta must move the output


def test_read_query_shapes(env):
    data, schema, gs, model = env
    h = encode(model, gs.base_view())
    pairs = data[TR].edge_index[:, :64]
    assert read_query(model, h, "nodelabel").shape == (data["account"].num_nodes,)
    assert read_query(model, h, "linkpred", pairs).shape == (64,)
    assert read_query(model, h, "linkreg", pairs).shape == (64,)


def test_whatif_rejects_schema_invalid(env):
    data, schema, gs, model = env
    bad = Intervention(add_edges={("account", "bogus", "loan"): torch.tensor([[0], [0]])})
    with pytest.raises(SchemaError):
        whatif_forward(model, gs, bad, schema)
