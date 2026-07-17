"""Diagnostic: can torch.compile close the Python-orchestration gap?

Two questions, reported honestly (including failures):

  1. Baseline forward — torch.compile the frozen encoder. Expected: small gain,
     because it is already a few large kernels (little dispatch overhead to
     remove). This also confirms the baseline we compare against is near its
     compiled optimum, so our speedups are not measured against a hobbled forward.

  2. Incremental maintainer — torch.compile delta_forward. Expected: heavy graph
     breaks / fallback, because the hot path is data-dependent control flow
     (torch.unique, .nonzero(), boolean masking, dict-of-tensor plumbing) that
     TorchDynamo cannot trace into a static graph. If so, that is itself the
     finding: the overhead is dynamic-shape dispatch, which torch.compile does
     not remove — a custom fused kernel (future work) would.

Exactness is re-checked for every compiled variant.

  python -m src.eval.bench_compile --dataset finbench
"""
from __future__ import annotations

import argparse
import statistics
import time
import warnings

import torch

from src.data.graph_state import GraphState
from src.data.loader import load_graph
from src.eval.exactness import check
from src.eval.runner import ckpt_path
from src.explore.sampler import sample_feature_iv
from src.forward.forward import set_determinism, whatif_forward
from src.incremental.cache import build_cache
from src.incremental.delta import delta_forward
from src.incremental.engine import incremental_forward
from src.models.model import load_model
from src.models.tasks import spec


def med(fn, reps=11, warmup=2):
    for _ in range(warmup):
        fn()
    ws = []
    for _ in range(reps):
        t = time.perf_counter(); fn(); ws.append(time.perf_counter() - t)
    return statistics.median(ws)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="finbench")
    ap.add_argument("--arch", default="rgcn")
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--cohort", type=int, default=100)
    args = ap.parse_args()

    set_determinism()
    data, schema = load_graph(args.dataset)
    gs = GraphState(data)
    sp = spec(args.dataset)
    model = load_model(ckpt_path(args.dataset, "nodelabel", args.arch, args.layers))
    cache = build_cache(model, gs.base_view())
    g = torch.Generator().manual_seed(0)
    iv = sample_feature_iv(schema, gs, g, sp.label_nt, args.cohort)
    truth = whatif_forward(model, gs, iv)

    print(f"=== torch.compile diagnostic: {args.dataset} {args.arch}-L{args.layers}, "
          f"feature Δ cohort={args.cohort} (torch {torch.__version__}) ===\n")

    # ---- 1. baseline forward, eager vs compiled ---------------------------
    x, ei = gs.base_view().x_dict, gs.base_view().edge_index_dict
    eager_fwd = med(lambda: model.encode(x, ei))
    print(f"[baseline forward]  eager: {eager_fwd*1e3:8.2f} ms")
    try:
        cencode = torch.compile(model.encoder, dynamic=False)
        with torch.no_grad():
            h_c = cencode(x, ei)          # triggers compilation (in warmup too)
        comp_fwd = med(lambda: cencode(x, ei))
        rep = check({k: v for k, v in h_c.items()},
                    model.encode(x, ei))
        print(f"[baseline forward]  compiled: {comp_fwd*1e3:8.2f} ms  "
              f"({eager_fwd/comp_fwd:4.2f}x, exact={rep.passed})")
    except Exception as e:
        print(f"[baseline forward]  compiled: FAILED — {type(e).__name__}: "
              f"{str(e)[:120]}")

    # ---- 2. incremental maintainer, eager vs compiled ---------------------
    eager_inc = med(lambda: incremental_forward(model, gs, cache, iv))
    print(f"\n[incremental delta] eager: {eager_inc*1e3:8.2f} ms  "
          f"(exact={check(incremental_forward(model, gs, cache, iv).h, truth).passed})")

    n_breaks = {"count": 0}
    try:
        import torch._dynamo as dynamo
        dynamo.reset()

        def _wrapped(fd):
            return delta_forward(model, cache, fd)[0]

        # count graph breaks explicitly rather than let them pass silently
        try:
            explanation = dynamo.explain(_wrapped)(iv.feat_deltas)
            n_breaks["count"] = getattr(explanation, "graph_break_count",
                                        len(getattr(explanation, "break_reasons", [])))
        except Exception as e:
            n_breaks["err"] = f"{type(e).__name__}: {str(e)[:100]}"

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cdelta = torch.compile(_wrapped, dynamic=True)
            h_c = cdelta(iv.feat_deltas)              # compile
            comp_inc = med(lambda: cdelta(iv.feat_deltas))
        rep = check(h_c, truth)
        breaks = n_breaks.get("err", f"{n_breaks['count']} graph breaks")
        print(f"[incremental delta] compiled: {comp_inc*1e3:8.2f} ms  "
              f"({eager_inc/comp_inc:4.2f}x, exact={rep.passed})  [{breaks}]")
    except Exception as e:
        print(f"[incremental delta] compiled: FAILED — {type(e).__name__}: "
              f"{str(e)[:160]}")

    print("\nInterpretation: the maintainer's speedup gap is dynamic-shape "
          "dispatch that torch.compile cannot fuse; the algorithmic (edges-avoided) "
          "advantage is the implementation-independent measure — see exp_overhead.")


if __name__ == "__main__":
    main()
