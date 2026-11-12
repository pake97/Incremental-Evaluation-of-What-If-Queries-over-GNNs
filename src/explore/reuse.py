"""Run an exploration trace under the three internal-ablation rungs.

  scratch      — from-scratch forward per state (the honest baseline)
  incremental  — incremental, NO reuse: every state replays its full D from
                 the base graph's cache
  reuse        — incremental + prefix/node reuse: resume from the nearest
                 materialized ancestor state (largest |D'|, D' ⊆ D) and apply
                 only the residual D \\ D'; materialize the resulting cache
                 (LRU-bounded) so later states can resume from it

All three produce exact results (rung (a) IS the ground truth; (b) and (c) are
gated by the exactness tests and can be spot-checked per state with
`check_against_scratch`). Every state row records wall-clock and the
hardware-independent work proxies (nodes recomputed, edges aggregated).
"""
from __future__ import annotations

import time
from collections import OrderedDict
from typing import Dict, List, Optional

from src.data.graph_state import GraphState
from src.data.schema import GraphSchema
from src.explore.state import Registry, State, compose
from src.explore.trace import Trace
from src.forward.forward import whatif_forward
from src.incremental.cache import ForwardCache, build_cache
from src.incremental.closure import closure_forward
from src.incremental.closure_v2 import closure_forward_v2
from src.incremental.delta import delta_forward
from src.incremental.delta_v2 import delta_forward_v2
from src.incremental.engine import (feat_dirty_rows, incremental_forward,
                                    is_topological, struct_dirty_rows,
                                    struct_dirty_rows_v2)
from src.incremental.index import get_index
from src.incremental.overlay import RowOverlay

RUNGS = ("scratch", "incremental", "reuse")


def _scratch_work(gs: GraphState, model) -> Dict[str, int]:
    """Work proxy of one full forward: every node re-encoded and every edge
    (forward + derived reverse) aggregated, at every layer."""
    L = model.cfg["num_layers"]
    n = sum(t.size(0) for t in gs.x.values())
    e = sum(t.size(1) for t in gs.edge_index.values())
    return {"nodes_recomputed": n * L, "edges_touched": 2 * e * L}


def run_trace(model, gs: GraphState, schema: GraphSchema, trace: Trace,
              rung: str, max_caches: int = 64,
              check_every: int = 0, impl: str = "v1",
              n_states: Optional[int] = None,
              account: bool = False) -> List[dict]:
    """Evaluate every state of the trace under one rung. Returns tidy rows.

    impl="v2": the incremental rungs run the v2 maintainers on the session
    index (built with the base cache, inside base_cache_wall_s); reuse resumes
    from the ancestor's OverlayCache. n_states evaluates a prefix of the visit
    order. account=True adds retained_bytes per row (untimed): base cache +
    index + everything reachable from the materialized caches (an evicted
    ancestor stays counted while a retained child's overlays point into it)."""
    assert rung in RUNGS and impl in ("v1", "v2")
    arch = model.cfg["model_type"]
    rows = []

    t0 = time.perf_counter()
    base_cache = build_cache(model, gs.base_view())
    index = get_index(base_cache) if impl == "v2" else None
    base_wall = time.perf_counter() - t0
    acct = _Retained(index) if account else None

    caches: "OrderedDict[State, ForwardCache]" = OrderedDict()
    caches[frozenset()] = base_cache

    for si, state in enumerate(trace.order[:n_states]):
        iv = compose(trace.registry, state)
        t0 = time.perf_counter()

        if rung == "scratch":
            h = whatif_forward(model, gs, iv)
            stats = {"method": "scratch", **_scratch_work(gs, model)}
        elif rung == "incremental":
            res = incremental_forward(model, gs, base_cache, iv, impl=impl)
            h, stats = res.h, res.stats
        elif impl == "v2":  # reuse on the session index + row overlays
            anc = _nearest_ancestor(caches, state)
            residual = state - anc
            iv_res = compose(trace.registry, residual)
            cache = caches[anc]
            caches.move_to_end(anc)
            # v2 delta reads topology from the base index only, so the whole
            # state (not just the residual) must be edge-free
            if arch == "rgcn" and not is_topological(iv):
                h, new_cache, stats = delta_forward_v2(model, cache, gs, iv_res, index)
                new_cache._iv = iv        # the state's graph is gs ⊕ iv, not ⊕ iv_res
            else:
                h, new_cache, stats = closure_forward_v2(
                    model, cache, gs, iv, index,
                    feat_dirty=feat_dirty_rows(iv_res),
                    struct_dirty=struct_dirty_rows_v2(gs, iv_res))
            stats = {**stats, "ancestor_depth": len(anc),
                     "residual_size": len(residual)}
            _materialize(caches, state, new_cache, max_caches)
        else:  # reuse
            anc = _nearest_ancestor(caches, state)
            residual = state - anc
            iv_res = compose(trace.registry, residual)
            cache = caches[anc]
            caches.move_to_end(anc)
            if arch == "rgcn" and not is_topological(iv_res):
                # cache topology == query topology (residual is feature-only)
                h, new_cache, stats = delta_forward(model, cache, iv_res.feat_deltas)
            else:
                view = gs.apply(iv)
                h, new_cache, stats = closure_forward(
                    model, cache, view,
                    feat_dirty=feat_dirty_rows(iv_res),
                    struct_dirty=struct_dirty_rows(gs, iv_res))
            stats = {**stats, "ancestor_depth": len(anc),
                     "residual_size": len(residual)}
            _materialize(caches, state, new_cache, max_caches)

        wall = time.perf_counter() - t0

        row = {"state_idx": si, "depth": len(state), "rung": rung,
               "wall_s": wall, **{k: v for k, v in stats.items()
                                  if k != "dirty_frontier"}}
        if acct is not None:
            row["retained_bytes"] = acct.bytes(caches)
        if check_every and si % check_every == 0:
            from src.eval.exactness import check
            rep = check(h, whatif_forward(model, gs, iv, schema))
            row["exact_max_abs_err"] = rep.max_abs_err
            row["exact_pass"] = rep.passed
        rows.append(row)

    if rows:
        rows[0]["base_cache_wall_s"] = base_wall  # amortized one-off, reported once
    return rows


class _Retained:
    """Unique bytes reachable from the materialized caches (dense tensors and
    overlay rows/values, following overlay chains) + the session index.
    Incremental while caches only grow; recounted after an eviction."""

    def __init__(self, index):
        self.extra = index.nbytes() if index is not None else 0
        self.seen, self.total, self.counted = set(), 0, set()

    def _t(self, t):                      # count each tensor once (h and z overlays share rows)
        if id(t) not in self.seen:
            self.seen.add(id(t))
            self.total += t.numel() * t.element_size()

    def _add(self, e):
        while isinstance(e, RowOverlay) and id(e) not in self.seen:
            self.seen.add(id(e))
            self._t(e.rows)
            self._t(e.vals)
            e = e.base
        if not isinstance(e, RowOverlay):
            self._t(e)

    def bytes(self, caches) -> int:
        if not self.counted <= set(caches):          # something was evicted
            self.seen, self.total, self.counted = set(), 0, set()
        for s, c in caches.items():
            if s in self.counted:
                continue
            self.counted.add(s)
            for layers in (c.h, c.z):
                for layer in layers:
                    for e in getattr(layer, "raw", layer).values():
                        self._add(e)
        return self.total + self.extra


def _nearest_ancestor(caches: Dict[State, ForwardCache], state: State) -> State:
    best: Optional[State] = None
    for s in caches:
        if s <= state and (best is None or len(s) > len(best)):
            best = s
    return best if best is not None else frozenset()


def _materialize(caches: "OrderedDict[State, ForwardCache]", state: State,
                 cache: ForwardCache, max_caches: int) -> None:
    caches[state] = cache
    while len(caches) > max_caches:
        for s in caches:
            if s != frozenset():        # never evict the base cache
                del caches[s]
                break
        else:
            break
