"""Store-backed what-if experiment: per-hypothetical DB round-trip cost on
Neo4j / Memgraph, folded into the total what-if cost.

  whatif_ms = maintenance_ms + db_roundtrip_ms

where db_roundtrip is:
  from_scratch : Δ writes + FULL graph download + rollback   (paid every query)
  streaming    : Δ writes + L-hop closure read + rollback     (paid every query)
  ours         : 0 per query (in-memory overlay; full download paid once -> cache)

Reports the one-time load and a break-even (queries after which ours' resident
cache beats re-downloading). Reads connections from configs/db.yaml.

  python -m src.eval.exp_db_roundtrip --config configs/db.yaml
"""
from __future__ import annotations

import argparse
import os

import torch
import yaml

from src.data.graph_state import GraphState
from src.data.loader import load_graph
from src.db.roundtrip import BoltStore
from src.eval.runner import dataset_card, summary_line, write_csv, write_manifest
from src.explore.sampler import sample_feature_iv, sample_topo_iv
from src.forward.forward import set_determinism
from src.models.tasks import spec
from src.paths import RESULTS

EXP = "db_roundtrip"


def bolt_stores(cfg):
    out = {}
    for name in ("neo4j", "memgraph"):
        c = cfg.get(name, {})
        if c.get("enabled"):
            out[name] = BoltStore(c["uri"], c.get("user", ""), c.get("password", ""),
                                  c.get("database"))
    return out


def tigergraph_store(cfg, dataset):
    c = cfg.get("tigergraph", {})
    if not c.get("enabled"):
        return None
    from src.db.roundtrip_tg import TigerGraphStore
    graph = c.get("graph_prefix", "rcf_") + dataset
    return TigerGraphStore(c["host"], c["rest_port"], c["gs_port"],
                           c["user"], c["password"], graph)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/db.yaml")
    ap.add_argument("--only", nargs="+", default=None,
                    help="restrict to these DBs (e.g. tigergraph)")
    ap.add_argument("--append", action="store_true",
                    help="merge into existing results.csv instead of overwriting "
                         "(keeps rows for DBs not measured this run)")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    set_determinism(cfg.get("seed", 42))
    L = cfg.get("layers", 2)
    reps = cfg.get("reps", 5)
    read_lhop = cfg.get("read_lhop", True)

    stores = bolt_stores(cfg)
    if args.only:
        stores = {k: v for k, v in stores.items() if k in args.only}
    print("connected:", ", ".join(stores) or "(bolt: none)")

    rows = []
    for dataset in cfg["datasets"]:
        data, schema = load_graph(dataset, refresh_card=False)
        gs = GraphState(data)
        sp = spec(dataset)
        labels = list(schema.node_types)
        relations = [tuple(et) for et in schema.edge_types]

        ds_stores = dict(stores)
        if args.only is None or "tigergraph" in args.only:
            tg = tigergraph_store(cfg, dataset)
            if tg is not None:
                ds_stores["tigergraph"] = tg
        if not ds_stores:
            print("FAIL db_roundtrip: no DB enabled/selected")
            return 1
        for db_name, store in ds_stores.items():
            # all engines namespace types per dataset (labels/types are shared,
            # e.g. person appears in finbench and snb)
            pfx = f"{dataset}_"
            load_s, nnodes, nedges = store.download_full_once(labels, relations, pfx=pfx)
            print(f"[{db_name}] {dataset}: one-time full download "
                  f"{load_s*1e3:.0f} ms ({nnodes:,} nodes, {nedges:,} edges)")
            for kind in ("feature", "topo"):
                for size in cfg["sizes"]:
                    g = torch.Generator().manual_seed(cfg.get("seed", 42))
                    if kind == "feature":
                        iv = sample_feature_iv(schema, gs, g, sp.label_nt, size)
                    else:
                        iv = sample_topo_iv(schema, gs, g, sp.core_rel,
                                            n_add=size, n_del=size)
                    scratch_ms = store.time_from_scratch(gs, iv, labels, relations, reps, pfx=pfx) * 1e3
                    stream_ms = store.time_streaming(gs, iv, L, reps, read_lhop, pfx=pfx) * 1e3
                    rows.append({
                        "db": db_name, "dataset": dataset, "kind": kind, "size": size,
                        "one_time_load_ms": round(load_s * 1e3, 2),
                        "from_scratch_db_ms": round(scratch_ms, 3),
                        "streaming_db_ms": round(stream_ms, 3),
                        "ours_db_ms": 0.0,
                        # break-even: after how many hypotheticals ours' one-time
                        # download is cheaper than from-scratch re-downloading
                        "breakeven_queries": round(load_s * 1e3 / max(scratch_ms, 1e-9), 2),
                    })
                    print(f"  {db_name} {dataset} {kind:7} sz={size:<5} "
                          f"scratch_db={scratch_ms:8.1f}ms stream_db={stream_ms:7.1f}ms ours=0",
                          flush=True)
        # clean up any stray props from feature Δ (rolled back, but belt-and-braces)
    for st in stores.values():
        st.close()

    if args.append:
        # keep existing rows for DBs we did NOT measure this run, replace the rest
        import csv
        path = os.path.join(RESULTS, EXP, "results.csv")
        measured = {r["db"] for r in rows}
        if os.path.exists(path):
            kept = [r for r in csv.DictReader(open(path)) if r["db"] not in measured]
            rows = kept + rows
    write_manifest(EXP, cfg, {d: dataset_card(d) for d in cfg["datasets"]})
    write_csv(EXP, rows)
    return summary_line(EXP, True, f"{len(rows)} rows")


if __name__ == "__main__":
    raise SystemExit(main())
