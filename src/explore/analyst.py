"""Scripted "analyst" traces for the clinical datasets — the Figure-1 story.

One hand-written exploration session per clinical dataset: a concrete,
semantically framed what-if narrative (illustration for the paper's case
study, explicitly NOT the main evidence — the swept synthetic traces in
trace.py are). Each step names its question; the runner evaluates the session
under incremental+reuse, verifies exactness per state, and prints the cost of
each step.

  python -m src.explore.analyst --dataset eicu
  python -m src.explore.analyst --dataset faers
"""
from __future__ import annotations

import argparse
import time

import torch

from src.data.graph_state import GraphState, Intervention
from src.data.loader import load_graph
from src.eval.exactness import check
from src.eval.runner import ckpt_path
from src.explore.state import Registry, compose
from src.forward.forward import set_determinism, whatif_forward
from src.incremental.cache import build_cache
from src.incremental.closure import closure_forward
from src.incremental.delta import delta_forward
from src.incremental.engine import (feat_dirty_rows, is_topological,
                                    struct_dirty_rows)
from src.models.model import load_model
from src.models.tasks import spec


def _feature_shift(data, nt, k, rows, delta):
    d = torch.zeros_like(data[nt].x)
    d[rows, k] = delta
    return Intervention(feat_deltas={nt: d})


def eicu_session(data, schema):
    """An intensivist explores a hypothetical sicker-cohort scenario.

    stay features: [unittype 1-hot | unitstaytype 1-hot | weight z | height z]
    """
    RECV = ("stay", "received", "medication")
    reg = Registry()
    deg = torch.zeros(schema.num_nodes["stay"], dtype=torch.long)
    deg.index_add_(0, data[RECV].edge_index[0],
                   torch.ones(data[RECV].edge_index.size(1), dtype=torch.long))
    heavy = torch.topk(deg, 50).indices          # 50 most-medicated stays
    w_col = schema.feat_dims["stay"] - 2         # admissionweight (z) column

    steps = [
        ("Δ1: what if the 50 most-medicated stays had admitted +1σ heavier?",
         _feature_shift(data, "stay", w_col, heavy, 1.0)),
        ("Δ2: ...and their 20 highest-dose medication links were absent?",
         Intervention(del_cols={RECV: torch.topk(
             data[RECV].edge_attr[:, 0], 20).indices})),
        ("Δ3: ...and they additionally received the 5 most common medications?",
         Intervention(add_edges={RECV: torch.stack([
             heavy[:5].repeat_interleave(5),
             torch.topk(torch.zeros(schema.num_nodes["medication"]).index_add_(
                 0, data[RECV].edge_index[1],
                 torch.ones(data[RECV].edge_index.size(1))), 5).indices.repeat(5)])})),
    ]
    # states: Δ1 -> Δ1+Δ2 -> Δ1+Δ3 (branch) -> Δ1+Δ2+Δ3 (re-convergence)
    ids = [reg.add(iv) for _, iv in steps]
    states = [
        (steps[0][0], frozenset({ids[0]})),
        (steps[1][0], frozenset({ids[0], ids[1]})),
        (steps[2][0] + "  [branch from Δ1]", frozenset({ids[0], ids[2]})),
        ("Δ1+Δ2+Δ3: both edits together  [re-convergence]",
         frozenset({ids[0], ids[1], ids[2]})),
    ]
    return reg, states


def faers_session(data, schema):
    """A pharmacovigilance analyst probes a drug's adverse-event profile."""
    CAUSES = ("drug", "causes", "term")
    reg = Registry()
    deg = torch.zeros(schema.num_nodes["drug"], dtype=torch.long)
    deg.index_add_(0, data[CAUSES].edge_index[0],
                   torch.ones(data[CAUSES].edge_index.size(1), dtype=torch.long))
    top_drugs = torch.topk(deg, 20).indices

    steps = [
        ("Δ1: what if the 20 most-reported drugs' first ATC-class flag flipped +1?",
         _feature_shift(data, "drug", 0, top_drugs, 1.0)),
        ("Δ2: ...and their 50 strongest co-report links were absent?",
         Intervention(del_cols={CAUSES: torch.topk(
             data[CAUSES].edge_attr[:, 0], 50).indices})),
        ("Δ3: ...and 25 new drug–event links appeared for them?",
         Intervention(add_edges={CAUSES: torch.stack([
             top_drugs[:5].repeat_interleave(5),
             torch.arange(25) % schema.num_nodes["term"]])})),
    ]
    ids = [reg.add(iv) for _, iv in steps]
    states = [
        (steps[0][0], frozenset({ids[0]})),
        (steps[1][0], frozenset({ids[0], ids[1]})),
        (steps[2][0] + "  [branch from Δ1]", frozenset({ids[0], ids[2]})),
        ("Δ1+Δ2+Δ3: both edits together  [re-convergence]",
         frozenset({ids[0], ids[1], ids[2]})),
    ]
    return reg, states


SESSIONS = {"eicu": eicu_session, "faers": faers_session}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=sorted(SESSIONS), required=True)
    ap.add_argument("--arch", default="rgcn")
    ap.add_argument("--layers", type=int, default=2)
    args = ap.parse_args()

    set_determinism()
    data, schema = load_graph(args.dataset)
    gs = GraphState(data)
    sp = spec(args.dataset)
    model = load_model(ckpt_path(args.dataset, "nodelabel", args.arch, args.layers))
    reg, states = SESSIONS[args.dataset](data, schema)

    print(f"=== analyst session: {args.dataset} ({args.arch}-L{args.layers}) ===")
    caches = {frozenset(): build_cache(model, gs.base_view())}
    all_ok = True
    for desc, state in states:
        anc = max((s for s in caches if s <= state), key=len)
        residual = state - anc
        iv_res = compose(reg, residual)
        iv_full = compose(reg, state)
        t0 = time.perf_counter()
        if model.cfg["model_type"] == "rgcn" and not is_topological(iv_res):
            h, nc, stats = delta_forward(model, caches[anc], iv_res.feat_deltas)
        else:
            h, nc, stats = closure_forward(model, caches[anc], gs.apply(iv_full),
                                           feat_dirty_rows(iv_res),
                                           struct_dirty_rows(gs, iv_res))
        wall = time.perf_counter() - t0
        caches[state] = nc

        rep = check(h, whatif_forward(model, gs, iv_full, schema))
        all_ok &= rep.passed
        p = model.classify_nodes(h, node_type=sp.label_nt).sigmoid()
        print(f"\n{desc}\n  resumed from |D'|={len(anc)}, residual={len(residual)}; "
              f"{stats['method']}: {wall*1e3:.1f} ms, "
              f"{stats['nodes_recomputed']} rows, {stats['edges_touched']} edges; "
              f"exact={rep.passed} (max_err={rep.max_abs_err:.1e})\n"
              f"  predicted-positive {sp.label_nt}s: {int((p > 0.5).sum())}")

    print(f"\n{'PASS' if all_ok else 'FAIL'} analyst_{args.dataset}: "
          f"{len(states)} states, all exact={all_ok}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
