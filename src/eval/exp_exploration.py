"""Experiment 4 — exploration reuse (the DAG experiment).

Run whole synthetic traces under the three internal-ablation rungs and compare
total cost:
  (a) scratch      — from-scratch per state
  (b) incremental  — incremental, no reuse (always replay D from G)
  (c) reuse        — incremental + prefix/node reuse (nearest materialized
                     ancestor)
sweeping the trace parameters. The primary reuse metric is *recomputation
avoided* (work proxies), wall-clock is reported alongside; the headline plot is
reuse benefit vs the prefix-sharing knob (and vs the realized sharing).

  python -m src.eval.exp_exploration --config configs/exploration_reuse.yaml
"""
from __future__ import annotations

import argparse
import itertools

from src.data.graph_state import GraphState
from src.data.loader import load_graph
from src.eval.runner import (ckpt_path, dataset_card, load_config,
                             missing_checkpoints, summary_line, write_csv,
                             write_manifest)
from src.explore.reuse import RUNGS, run_trace
from src.explore.trace import TraceConfig, generate
from src.forward.forward import set_determinism
from src.models.model import load_model
from src.models.tasks import spec

EXP = "exploration_reuse"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/exploration_reuse.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    set_determinism(cfg["seed"])

    state_rows, summary_rows, all_ok = [], [], True
    sweep = list(itertools.product(cfg["share_rates"], cfg["backtrack_rates"],
                                   cfg["branchings"]))

    for dataset in cfg["datasets"]:
        data, schema = load_graph(dataset)
        gs = GraphState(data)
        sp = spec(dataset)
        for arch in cfg["archs"]:
            task, L = cfg["task"], cfg["layers"]
            if missing_checkpoints(dataset, [arch], [L], [task]):
                print(f"SKIP (no checkpoint): {dataset} {task} {arch} L{L}")
                continue
            model = load_model(ckpt_path(dataset, task, arch, L))

            for share, backtrack, branching in sweep:
                tc = TraceConfig(
                    n_states=cfg["n_states"], branching=branching,
                    max_depth=cfg["max_depth"], backtrack_rate=backtrack,
                    share_rate=share, feature_frac=cfg["feature_frac"],
                    cohort_size=cfg["cohort_size"], topo_edits=cfg["topo_edits"],
                    node_type=sp.label_nt, relation=sp.core_rel, seed=cfg["seed"])
                trace = generate(schema, gs, tc)
                struct = trace.structure()

                totals = {}
                for rung in RUNGS:
                    rows = run_trace(model, gs, schema, trace, rung,
                                     max_caches=cfg["max_caches"],
                                     check_every=cfg["check_every"])
                    checked = [r for r in rows if "exact_pass" in r]
                    ok = all(r["exact_pass"] for r in checked)
                    all_ok &= ok
                    tot = {
                        "wall_s": sum(r["wall_s"] for r in rows),
                        "nodes": sum(r["nodes_recomputed"] for r in rows),
                        "edges": sum(r["edges_touched"] for r in rows),
                    }
                    totals[rung] = tot
                    for r in rows:
                        state_rows.append({"dataset": dataset, "arch": arch,
                                           "share_rate": share,
                                           "backtrack_rate": backtrack,
                                           "branching": branching, **r})

                summary_rows.append({
                    "dataset": dataset, "arch": arch,
                    "share_rate": share, "backtrack_rate": backtrack,
                    "branching": branching, **struct,
                    **{f"{r}_{k}": v for r in RUNGS for k, v in totals[r].items()},
                    # primary reuse metrics: recomputation avoided
                    "reuse_ratio_nodes": totals["reuse"]["nodes"]
                                         / max(totals["incremental"]["nodes"], 1),
                    "reuse_ratio_edges": totals["reuse"]["edges"]
                                         / max(totals["incremental"]["edges"], 1),
                    "reuse_ratio_wall": totals["reuse"]["wall_s"]
                                        / max(totals["incremental"]["wall_s"], 1e-12),
                    "speedup_vs_scratch_wall": totals["scratch"]["wall_s"]
                                               / max(totals["reuse"]["wall_s"], 1e-12),
                })
                s = summary_rows[-1]
                print(f"{dataset} {arch} share={share} bt={backtrack} br={branching} "
                      f"sharing={struct['sharing']:.2f} "
                      f"reuse_ratio_edges={s['reuse_ratio_edges']:.3f} "
                      f"speedup_vs_scratch={s['speedup_vs_scratch_wall']:.1f}x",
                      flush=True)

    write_manifest(EXP, cfg, {d: dataset_card(d) for d in cfg["datasets"]})
    write_csv(EXP, state_rows, "states.csv")
    write_csv(EXP, summary_rows, "summary.csv")
    return summary_line(EXP, all_ok,
                        f"{len(summary_rows)} trace configs x {len(RUNGS)} rungs, "
                        f"spot-checks exact={all_ok}")


if __name__ == "__main__":
    raise SystemExit(main())
