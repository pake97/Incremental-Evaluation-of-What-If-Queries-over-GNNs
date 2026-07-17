"""RGCN and HeteroGAT encoders for FinBench, with 3 task heads.

Two architectures x four depths = eight models:
    model_type in {"rgcn", "hgat"};  num_layers in {1, 2, 3, 4}.

Both are deliberately the most basic form:
  * RGCN cell  : out = mean_r( W_msg h_src ) + W_self h_dst   (per-relation W).
  * HGAT cell  : single-head GAT; LeakyReLU(0.2) inside the attention score
                 (canonical GAT), summed over relations.
ReLU is the activation between layers; the last layer has none (so the
dot-product link scorer is not forced onto the non-negative orthant).

Every cell accepts a per-edge ``edge_weight`` in [0, 1] -- multiplicative for
RGCN, a log-domain bias before the attention softmax for HGAT -- so the
continuous-relaxation soft path (``GraphState.soft_apply``) is differentiable
end to end.  Reverse relations are built on the fly from the forward edges, so
the encoder consumes exactly the dicts a ``GraphView`` produces.

Heads:
  * node labelling   : Linear(hidden -> 1) on account  (BCEWithLogits)
  * link prediction  : <h_a, h_b>           on account pairs (BCEWithLogits)
  * link regression  : MLP([h_a || h_b])    on account pairs (target log1p amt)
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import scatter, softmax as scatter_softmax

EdgeType = Tuple[str, str, str]
ACCOUNT = "account"


def _key(et: EdgeType) -> str:
    return "__".join(et)


# ---------------------------------------------------------------------------
# Per-relation cells
# ---------------------------------------------------------------------------
class _RgcnCell(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.lin_msg = nn.Linear(in_dim, out_dim, bias=False)
        self.lin_self = nn.Linear(in_dim, out_dim, bias=True)

    def forward(self, h_src, h_dst, edge_index, edge_weight, dst_size):
        if edge_index.numel() == 0:
            agg = h_src.new_zeros((dst_size, self.lin_msg.out_features))
        else:
            src, dst = edge_index[0], edge_index[1]
            msg = self.lin_msg(h_src)[src] * edge_weight.unsqueeze(-1)
            num = scatter(msg, dst, dim=0, dim_size=dst_size, reduce="sum")
            deg = scatter(edge_weight, dst, dim=0, dim_size=dst_size,
                          reduce="sum").clamp(min=1e-8)
            agg = num / deg.unsqueeze(-1)
        return agg + self.lin_self(h_dst)


class _HgatCell(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim, bias=False)
        self.lin_self = nn.Linear(in_dim, out_dim, bias=True)
        self.att_src = nn.Parameter(torch.empty(out_dim))
        self.att_dst = nn.Parameter(torch.empty(out_dim))
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.xavier_uniform_(self.lin_self.weight)
        nn.init.normal_(self.att_src, std=0.1)
        nn.init.normal_(self.att_dst, std=0.1)

    def forward(self, h_src, h_dst, edge_index, edge_weight, dst_size):
        if edge_index.numel() == 0:
            return self.lin_self(h_dst)
        src, dst = edge_index[0], edge_index[1]
        hs, hd = self.lin(h_src), self.lin(h_dst)
        e = (hs * self.att_src).sum(-1)[src] + (hd * self.att_dst).sum(-1)[dst]
        e = F.leaky_relu(e, 0.2) + torch.log(edge_weight.clamp(min=1e-6))
        alpha = scatter_softmax(e, dst, num_nodes=dst_size)
        msg = hs[src] * alpha.unsqueeze(-1)
        agg = scatter(msg, dst, dim=0, dim_size=dst_size, reduce="sum")
        return agg + self.lin_self(h_dst)


_CELL = {"rgcn": _RgcnCell, "hgat": _HgatCell}


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------
class HeteroEncoder(nn.Module):
    def __init__(self, in_dims: Dict[str, int], edge_types: List[EdgeType],
                 hidden: int = 64, num_layers: int = 2, model_type: str = "rgcn"):
        super().__init__()
        assert model_type in _CELL and 1 <= num_layers <= 4
        self.model_type, self.num_layers, self.hidden = model_type, num_layers, hidden
        self.node_types = list(in_dims)

        # forward relations + their reverses (rev_ derived at forward time)
        self.fwd_types = list(edge_types)
        self.rev_types = [(d, "rev_" + r, s) for (s, r, d) in edge_types]
        self.all_types = self.fwd_types + self.rev_types

        self.proj = nn.ModuleDict(
            {nt: nn.Linear(d, hidden) for nt, d in in_dims.items()})
        Cell = _CELL[model_type]
        self.layers = nn.ModuleList([
            nn.ModuleDict({_key(et): Cell(hidden, hidden) for et in self.all_types})
            for _ in range(num_layers)
        ])

    def forward(self, x_dict, edge_index_dict, edge_weight_dict=None):
        h = {nt: self.proj[nt](x_dict[nt]) for nt in self.node_types}

        # build reverse edges (and weights) once per forward
        ei_all, ew_all = {}, {}
        for (s, r, d) in self.fwd_types:
            et = (s, r, d)
            ei = edge_index_dict[et]
            ew = (edge_weight_dict or {}).get(et)
            if ew is None:
                ew = torch.ones(ei.size(1), device=h[s].device)
            ei_all[et] = ei
            ew_all[et] = ew
            rev = (d, "rev_" + r, s)
            ei_all[rev] = ei.flip(0)
            ew_all[rev] = ew

        for li, layer in enumerate(self.layers):
            acc = {nt: [] for nt in self.node_types}
            for et in self.all_types:
                s, _, d = et
                acc[d].append(layer[_key(et)](h[s], h[d], ei_all[et],
                                              ew_all[et], h[d].size(0)))
            h_new = {}
            for nt in self.node_types:
                z = sum(acc[nt]) if acc[nt] else h[nt]
                h_new[nt] = F.relu(z) if li < self.num_layers - 1 else z
            h = h_new
        return h


# ---------------------------------------------------------------------------
# Full model: encoder + 3 heads
# ---------------------------------------------------------------------------
class FinBenchGNN(nn.Module):
    def __init__(self, in_dims, edge_types, hidden=64, num_layers=2, model_type="rgcn"):
        super().__init__()
        self.cfg = dict(in_dims=dict(in_dims), edge_types=[tuple(e) for e in edge_types],
                        hidden=hidden, num_layers=num_layers, model_type=model_type)
        self.encoder = HeteroEncoder(in_dims, edge_types, hidden, num_layers, model_type)
        self.node_head = nn.Linear(hidden, 1)                       # fraud logit
        self.reg_head = nn.Sequential(                              # amount (log1p)
            nn.Linear(2 * hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    # encode from raw data or from a GraphView's dicts
    def encode(self, x_dict, edge_index_dict, edge_weight_dict=None):
        return self.encoder(x_dict, edge_index_dict, edge_weight_dict)

    def classify_nodes(self, h, node_type=ACCOUNT):
        return self.node_head(h[node_type]).squeeze(-1)            # [N]

    def link_score(self, h, pair_index, src_type=ACCOUNT, dst_type=None):
        dst_type = dst_type or src_type
        a, b = pair_index
        return (h[src_type][a] * h[dst_type][b]).sum(-1)          # [E] (logit)

    def link_regress(self, h, pair_index, src_type=ACCOUNT, dst_type=None):
        dst_type = dst_type or src_type
        a, b = pair_index
        z = torch.cat([h[src_type][a], h[dst_type][b]], dim=-1)
        return self.reg_head(z).squeeze(-1)                       # [E]


def build_model(data, model_type="rgcn", num_layers=2, hidden=64):
    in_dims = {nt: data[nt].x.size(1) for nt in data.node_types}
    return FinBenchGNN(in_dims, list(data.edge_types), hidden, num_layers, model_type)


def save_model(model, path):
    import os
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "cfg": model.cfg}, path)


def load_model(path, map_location="cpu"):
    p = torch.load(path, map_location=map_location, weights_only=False)
    m = FinBenchGNN(**p["cfg"])
    m.load_state_dict(p["state_dict"]); m.eval()
    return m
