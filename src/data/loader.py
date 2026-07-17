"""Common entry point: every dataset loads into the same in-memory format.

``load_graph(name)`` returns ``(HeteroData, GraphSchema)`` and refreshes the
dataset card. Models, interventions, and maintenance code are dataset-agnostic
from here on — they only ever see the HeteroData dicts and the schema.
"""
from __future__ import annotations

import json
import os
from typing import Dict, Tuple

import torch

from src.data.schema import GraphSchema, schema_of
from src.paths import (EICU_FULL_PROC, EICU_PROC, FAERS_PROC, FINBENCH_PROC,
                       SNB_PROC)

# dataset name -> (processed dir, graph file). eicu_full shares the eICU schema
# (built by the same builder with --release full).
GRAPHS: Dict[str, Tuple[str, str]] = {
    "finbench": (FINBENCH_PROC, "finbench_hetero.pt"),
    "faers": (FAERS_PROC, "faers_hetero.pt"),
    "eicu": (EICU_PROC, "eicu_hetero.pt"),
    "eicu_full": (EICU_FULL_PROC, "eicu_hetero.pt"),
    "snb": (SNB_PROC, "snb_hetero.pt"),
}
TASK_SPLITS = ("linkpred", "linkreg", "nodelabel")


def load_graph(name: str, refresh_card: bool = True):
    if name not in GRAPHS:
        raise KeyError(f"unknown dataset '{name}'; known: {sorted(GRAPHS)}")
    proc, fname = GRAPHS[name]
    path = os.path.join(proc, fname)
    if not os.path.exists(path):
        raise FileNotFoundError(f"processed graph missing: {path} (run the {name} prepare script)")
    data = torch.load(path, weights_only=False)
    schema = schema_of(data)
    if refresh_card:
        write_dataset_card(name, data, schema)
    return data, schema


def load_split(name: str, task: str):
    proc, _ = GRAPHS[name]
    if task not in TASK_SPLITS:
        raise KeyError(f"unknown task '{task}'; known: {TASK_SPLITS}")
    return torch.load(os.path.join(proc, f"split_{task}.pt"), weights_only=False)


def write_dataset_card(name: str, data, schema: GraphSchema) -> dict:
    """Emit dataset_card.json next to the processed graph (see Data contract)."""
    proc, _ = GRAPHS[name]
    card = {
        "dataset": name,
        "num_nodes": schema.num_nodes,
        "num_edges": {"__".join(et): n for et, n in schema.num_edges.items()},
        "feat_dims": schema.feat_dims,
        "tasks": [t for t in TASK_SPLITS
                  if os.path.exists(os.path.join(proc, f"split_{t}.pt"))],
        "torch": torch.__version__,
    }
    with open(os.path.join(proc, "dataset_card.json"), "w") as fh:
        json.dump(card, fh, indent=2)
    return card
