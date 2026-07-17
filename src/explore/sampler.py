"""Random schema-valid interventions with controlled locality.

Used by the exactness experiment (many random Δ), the efficiency sweeps
(intervention size/locality is the swept variable), and the trace generator
(intervention mix). All randomness flows through a caller-owned
torch.Generator so runs are reproducible from the config seed.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch

from src.data.graph_state import GraphState, Intervention
from src.data.schema import GraphSchema

EdgeType = Tuple[str, str, str]


def sample_feature_iv(schema: GraphSchema, gs: GraphState, g: torch.Generator,
                      node_type: str, cohort_size: int = 32,
                      feature_k: Optional[int] = None,
                      delta_scale: float = 1.0) -> Intervention:
    """Shift one feature coordinate by a fixed δ over a random cohort —
    the rank-one feature intervention of the paper."""
    n, d = schema.num_nodes[node_type], schema.feat_dims[node_type]
    cohort = torch.randperm(n, generator=g)[:min(cohort_size, n)]
    k = int(torch.randint(0, d, (1,), generator=g)) if feature_k is None else feature_k
    col = gs.x[node_type][:, k]
    scale = float(col.std())
    delta_val = delta_scale * (scale if scale > 1e-8 else 1.0)
    delta = torch.zeros(n, d)
    delta[cohort, k] = delta_val
    return Intervention(feat_deltas={node_type: delta})


def sample_topo_iv(schema: GraphSchema, gs: GraphState, g: torch.Generator,
                   relation: EdgeType, n_add: int = 8, n_del: int = 8) -> Intervention:
    """Insert/delete random edges on one relation (schema-valid by
    construction). A 'swap' is n_del = n_add on the same relation."""
    s, _, d = relation
    add_edges, del_cols = {}, {}
    if n_add > 0:
        add_edges[relation] = torch.stack([
            torch.randint(0, schema.num_nodes[s], (n_add,), generator=g),
            torch.randint(0, schema.num_nodes[d], (n_add,), generator=g)])
    if n_del > 0:
        E = schema.num_edges[relation]
        del_cols[relation] = torch.randperm(E, generator=g)[:min(n_del, E)]
    return Intervention(add_edges=add_edges, del_cols=del_cols)


def sample_mixed_iv(schema: GraphSchema, gs: GraphState, g: torch.Generator,
                    node_type: str, relation: EdgeType,
                    cohort_size: int = 32, n_add: int = 8, n_del: int = 8) -> Intervention:
    f = sample_feature_iv(schema, gs, g, node_type, cohort_size)
    t = sample_topo_iv(schema, gs, g, relation, n_add, n_del)
    return Intervention(add_edges=t.add_edges, del_cols=t.del_cols,
                        feat_deltas=f.feat_deltas)
