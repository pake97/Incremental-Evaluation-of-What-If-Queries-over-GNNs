"""Graph schema: the contract every intervention is validated against.

An ``Intervention`` is only admissible if it references node/edge types that
exist in the graph, ids inside their type's range, and feature deltas of the
right shape. ``GraphSchema.validate`` raises ``SchemaError`` otherwise —
schema-invalid what-ifs are rejected before any tensor work happens.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from src.data.graph_state import Intervention

EdgeType = Tuple[str, str, str]


class SchemaError(ValueError):
    """The intervention references something the schema does not admit."""


@dataclass(frozen=True)
class GraphSchema:
    num_nodes: Dict[str, int]           # node type -> |V_t|
    feat_dims: Dict[str, int]           # node type -> feature dim
    num_edges: Dict[EdgeType, int]      # relation  -> |E_r| in the base graph
    edge_types: List[EdgeType]          # admissible (src, rel, dst) triples

    @property
    def node_types(self) -> List[str]:
        return list(self.num_nodes)

    def validate(self, iv: Intervention) -> None:
        for et, new in iv.add_edges.items():
            if et not in self.num_edges:
                raise SchemaError(f"add_edges: relation {et} not in schema")
            if new.dim() != 2 or new.size(0) != 2:
                raise SchemaError(f"add_edges[{et}]: expected shape [2, k], got {tuple(new.shape)}")
            if new.numel():
                s, _, d = et
                if int(new[0].min()) < 0 or int(new[0].max()) >= self.num_nodes[s]:
                    raise SchemaError(f"add_edges[{et}]: src id out of range for '{s}'")
                if int(new[1].min()) < 0 or int(new[1].max()) >= self.num_nodes[d]:
                    raise SchemaError(f"add_edges[{et}]: dst id out of range for '{d}'")

        for et, cols in iv.del_cols.items():
            if et not in self.num_edges:
                raise SchemaError(f"del_cols: relation {et} not in schema")
            if cols.numel():
                if int(cols.min()) < 0 or int(cols.max()) >= self.num_edges[et]:
                    raise SchemaError(f"del_cols[{et}]: column out of range (|E|={self.num_edges[et]})")

        for nt, delta in iv.feat_deltas.items():
            if nt not in self.num_nodes:
                raise SchemaError(f"feat_deltas: node type '{nt}' not in schema")
            want = (self.num_nodes[nt], self.feat_dims[nt])
            if tuple(delta.shape) != want:
                raise SchemaError(f"feat_deltas[{nt}]: expected shape {want}, got {tuple(delta.shape)}")


def schema_of(data) -> GraphSchema:
    """Derive the schema from a PyG ``HeteroData``."""
    return GraphSchema(
        num_nodes={nt: int(data[nt].num_nodes) for nt in data.node_types},
        feat_dims={nt: int(data[nt].x.size(1)) for nt in data.node_types},
        num_edges={et: int(data[et].edge_index.size(1)) for et in data.edge_types},
        edge_types=[tuple(et) for et in data.edge_types],
    )
