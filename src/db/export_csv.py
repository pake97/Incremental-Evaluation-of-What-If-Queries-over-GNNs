"""Export each processed graph to CSVs for loading into a property-graph DB.

The DB node key is (label, idx) where idx is the node's TENSOR ROW INDEX — the
same index our interventions reference — so a Δ (which names tensor ids) maps to
DB node lookups with no translation table. Emits, per dataset, into
db_export/<dataset>/:

  nodes_<label>.csv     columns: idx            (one row per node of that type)
  edges_<rel>.csv       columns: src, dst       (tensor idx of endpoints)
  schema.json           labels, relations (src/dst types), counts

Run:  python -m src.db.export_csv                 (all datasets)
      python -m src.db.export_csv --datasets finbench eicu
"""
from __future__ import annotations

import argparse
import csv
import json
import os

import torch

from src.data.loader import load_graph
from src.paths import ROOT

OUT_ROOT = os.path.join(ROOT, "db_export")


def export(dataset: str):
    data, schema = load_graph(dataset, refresh_card=False)
    outdir = os.path.join(OUT_ROOT, dataset)
    os.makedirs(outdir, exist_ok=True)

    for nt in schema.node_types:
        with open(os.path.join(outdir, f"nodes_{nt}.csv"), "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["idx"])
            for i in range(schema.num_nodes[nt]):
                w.writerow([i])

    rel_meta = []
    for et in schema.edge_types:
        s, r, d = et
        ei = data[et].edge_index
        fname = f"edges_{s}__{r}__{d}.csv"
        with open(os.path.join(outdir, fname), "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["src", "dst"])
            for a, b in ei.t().tolist():
                w.writerow([a, b])
        rel_meta.append({"file": fname, "src": s, "rel": r, "dst": d,
                         "count": int(ei.size(1))})

    meta = {
        "dataset": dataset,
        "labels": {nt: schema.num_nodes[nt] for nt in schema.node_types},
        "relations": rel_meta,
        "node_key": "idx (tensor row index; unique within a label)",
    }
    with open(os.path.join(outdir, "schema.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    total_e = sum(m["count"] for m in rel_meta)
    print(f"{dataset:9} -> db_export/{dataset}/  "
          f"|V|={sum(schema.num_nodes.values()):>8,}  |E|={total_e:>10,}  "
          f"({len(schema.node_types)} labels, {len(rel_meta)} relations)")
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+",
                    default=["finbench", "eicu", "faers", "snb"])
    args = ap.parse_args()
    os.makedirs(OUT_ROOT, exist_ok=True)
    for ds in args.datasets:
        export(ds)
    print(f"\nWrote CSVs under {OUT_ROOT}/  (load per db/README.md)")


if __name__ == "__main__":
    main()
