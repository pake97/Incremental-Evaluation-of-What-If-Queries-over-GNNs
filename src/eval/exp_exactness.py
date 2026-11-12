"""Experiment 1 — exactness (THE GATE).

For every (dataset, model, task, intervention kind): assert incremental ==
from-scratch on many random Δ, on the encoder output AND the task readout.
Both maintainers are exercised (delta additionally to closure wherever it
applies). No timing experiment is meaningful unless this passes.

  python -m src.eval.exp_exactness --config configs/exactness.yaml
"""
from __future__ import annotations

import argparse

import torch

from src.data.graph_state import GraphState
from src.data.loader import load_graph
from src.eval.exactness import check
from src.eval.runner import (ckpt_path, dataset_card, load_config,
                             missing_checkpoints, summary_line, write_csv,
                             write_manifest)
from src.explore.sampler import sample_feature_iv, sample_mixed_iv, sample_topo_iv
from src.forward.forward import read_query, set_determinism, whatif_forward
from src.incremental.cache import build_cache
from src.incremental.engine import incremental_forward, is_topological
from src.models.model import load_model
from src.models.tasks import spec

EXP = "exactness"


def sample_iv(kind, schema, gs, g, sp, cohort, edits):
    if kind == "feature":
        return sample_feature_iv(schema, gs, g, sp.label_nt, cohort)
    if kind == "topo":
        return sample_topo_iv(schema, gs, g, sp.core_rel, n_add=edits, n_del=edits)
    return sample_mixed_iv(schema, gs, g, sp.label_nt, sp.core_rel, cohort, edits, edits)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/exactness.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    set_determinism(cfg["seed"])

    rows, n_checks, all_ok, max_err = [], 0, True, 0.0
    for dataset in cfg["datasets"]:
        data, schema = load_graph(dataset)
        gs = GraphState(data)
        sp = spec(dataset)
        pairs = data[tuple(sp.core_rel)].edge_index[:, :cfg["n_readout_pairs"]]

        for arch in cfg["archs"]:
            for L in cfg["layers"]:
                for task in (t for t in cfg["tasks"] if t in sp.tasks):
                    if missing_checkpoints(dataset, [arch], [L], [task]):
                        print(f"SKIP (no checkpoint): {dataset} {task} {arch} L{L}")
                        continue
                    model = load_model(ckpt_path(dataset, task, arch, L))
                    cache = build_cache(model, gs.base_view())
                    g = torch.Generator().manual_seed(cfg["seed"])

                    for kind in cfg["kinds"]:
                        for i in range(cfg["n_per_kind"]):
                            cohort = int(torch.randint(1, cfg["max_cohort"], (1,), generator=g))
                            edits = int(torch.randint(1, cfg["max_edits"], (1,), generator=g))
                            iv = sample_iv(kind, schema, gs, g, sp, cohort, edits)
                            truth = whatif_forward(model, gs, iv, schema)

                            methods = ["auto"]
                            if arch == "rgcn" and not is_topological(iv):
                                methods.append("closure")  # auto == delta here
                            for method, impl in ((m, i) for m in methods
                                                 for i in cfg.get("impls", ["v1"])):
                                res = incremental_forward(model, gs, cache, iv,
                                                          schema, method=method, impl=impl)
                                rep_h = check(res.h, truth)
                                rep_v1 = None
                                if impl != "v1":   # v2 must also equal v1
                                    rep_v1 = check(res.h, incremental_forward(
                                        model, gs, cache, iv, schema, method=method).h)
                                out_inc = read_query(model, res.h, task, pairs,
                                                     node_type=sp.label_nt,
                                                     src_type=sp.core_rel[0],
                                                     dst_type=sp.core_rel[2])
                                out_tru = read_query(model, truth, task, pairs,
                                                     node_type=sp.label_nt,
                                                     src_type=sp.core_rel[0],
                                                     dst_type=sp.core_rel[2])
                                rep_o = check(out_inc, out_tru)
                                ok = rep_h.passed and rep_o.passed and \
                                    (rep_v1 is None or rep_v1.passed)
                                all_ok &= ok
                                n_checks += 1
                                max_err = max(max_err, rep_h.max_abs_err, rep_o.max_abs_err)
                                rows.append({
                                    "dataset": dataset, "arch": arch, "layers": L,
                                    "task": task, "kind": kind, "i": i,
                                    "method": res.stats["method"], "impl": impl,
                                    "h_max_abs_err": rep_h.max_abs_err,
                                    "h_mean_abs_err": rep_h.mean_abs_err,
                                    "out_max_abs_err": rep_o.max_abs_err,
                                    "vs_v1_max_abs_err": rep_v1.max_abs_err if rep_v1 else 0.0,
                                    "passed": ok,
                                    "nodes_recomputed": res.stats["nodes_recomputed"],
                                    "edges_touched": res.stats["edges_touched"],
                                })
                                if not ok:
                                    print(f"FAIL: {rows[-1]}")

    write_manifest(EXP, cfg, {d: dataset_card(d) for d in cfg["datasets"]})
    write_csv(EXP, rows)
    return summary_line(EXP, all_ok,
                        f"{n_checks} checks, max_abs_err={max_err:.3e} "
                        f"(atol={cfg['atol']}, rtol={cfg['rtol']})")


if __name__ == "__main__":
    raise SystemExit(main())
