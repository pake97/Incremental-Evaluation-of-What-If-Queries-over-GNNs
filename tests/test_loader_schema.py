"""Common loader + schema validation (the intervention admissibility gate)."""
import json
import os

import pytest
import torch

from src.data.graph_state import Intervention
from src.data.loader import GRAPHS, load_graph, load_split
from src.data.schema import SchemaError
from src.paths import FINBENCH_PROC

TR = ("account", "transfer", "account")

pytestmark = pytest.mark.skipif(
    not os.path.exists(os.path.join(FINBENCH_PROC, "finbench_hetero.pt")),
    reason="processed FinBench graph not present",
)


@pytest.fixture(scope="module")
def finbench():
    return load_graph("finbench")


def test_schema_matches_graph(finbench):
    data, schema = finbench
    assert schema.num_nodes["account"] == data["account"].num_nodes
    assert schema.num_edges[TR] == data[TR].edge_index.size(1)
    assert set(schema.edge_types) == {tuple(et) for et in data.edge_types}


def test_dataset_card_written(finbench):
    card_path = os.path.join(GRAPHS["finbench"][0], "dataset_card.json")
    card = json.load(open(card_path))
    _, schema = finbench
    assert card["num_nodes"] == schema.num_nodes
    assert set(card["tasks"]) == {"linkpred", "linkreg", "nodelabel"}


def test_splits_load(finbench):
    nl = load_split("finbench", "nodelabel")
    assert "train_mask" in nl


def test_valid_intervention_passes(finbench):
    data, schema = finbench
    d = torch.zeros_like(data["account"].x)
    d[0, 0] = 1.0
    schema.validate(Intervention(
        add_edges={TR: torch.tensor([[0, 1], [2, 3]])},
        del_cols={TR: torch.tensor([0])},
        feat_deltas={"account": d},
    ))


def test_rejects_unknown_relation(finbench):
    _, schema = finbench
    with pytest.raises(SchemaError, match="not in schema"):
        schema.validate(Intervention(
            add_edges={("account", "bogus", "loan"): torch.tensor([[0], [0]])}))


def test_rejects_out_of_range_node(finbench):
    _, schema = finbench
    bad = schema.num_nodes["account"]  # first invalid id
    with pytest.raises(SchemaError, match="out of range"):
        schema.validate(Intervention(add_edges={TR: torch.tensor([[bad], [0]])}))


def test_rejects_out_of_range_del_col(finbench):
    _, schema = finbench
    with pytest.raises(SchemaError, match="column out of range"):
        schema.validate(Intervention(del_cols={TR: torch.tensor([schema.num_edges[TR]])}))


def test_rejects_bad_feat_shape(finbench):
    _, schema = finbench
    with pytest.raises(SchemaError, match="expected shape"):
        schema.validate(Intervention(feat_deltas={"account": torch.zeros(3, 3)}))


@pytest.mark.skipif(
    not os.path.exists(os.path.join(GRAPHS["faers"][0], "faers_hetero.pt")),
    reason="processed FAERS graph not present")
def test_faers_loads_same_contract():
    data, schema = load_graph("faers")
    assert "drug" in schema.num_nodes
    assert ("drug", "causes", "term") in schema.edge_types
