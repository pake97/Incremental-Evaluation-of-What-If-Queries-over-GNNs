"""From-scratch forward inference — the ground truth.

Every incremental result is judged against ``whatif_forward``: apply the
intervention to the base graph (copy-on-write), run the frozen model's full
forward on the modified graph, read the task output. Nothing here is
incremental by design; this path must stay the simplest possible correct
implementation.
"""
from __future__ import annotations

import os
import random

import numpy as np
import torch

from src.data.graph_state import GraphState, GraphView, Intervention
from src.data.schema import GraphSchema

DEFAULT_SEED = 42


def set_determinism(seed: int = DEFAULT_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


@torch.no_grad()
def encode(model, view: GraphView):
    """Full forward of the frozen encoder on a graph view. h dict, no grad."""
    model.eval()
    return model.encode(view.x_dict, view.edge_index_dict)


@torch.no_grad()
def whatif_forward(model, gs: GraphState, iv: Intervention,
                   schema: GraphSchema | None = None):
    """Ground-truth answer to a what-if: h on G^Delta, computed from scratch."""
    if schema is not None:
        schema.validate(iv)
    return encode(model, gs.apply(iv))


@torch.no_grad()
def read_query(model, h, task: str, pair_index=None, node_type: str = "account",
               src_type: str = "account", dst_type: str | None = None):
    """Read a query off encoded state h. Output tensors are what exactness
    equivalence is asserted on. Node/edge types come from the dataset's
    TaskSpec (src.models.tasks)."""
    if task == "nodelabel":
        return model.classify_nodes(h, node_type=node_type)
    if task == "linkpred":
        return model.link_score(h, pair_index, src_type=src_type, dst_type=dst_type)
    if task == "linkreg":
        return model.link_regress(h, pair_index, src_type=src_type, dst_type=dst_type)
    raise KeyError(f"unknown task '{task}'")
