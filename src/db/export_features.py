"""Export node feature matrices to CSV so X can live in the property store.

The store-backed pipeline experiment downloads X (features) and A (adjacency)
from the DB to rebuild the GNN input; that requires the node features to be
stored as node properties. Emits, per dataset, db_export/<dataset>/:

  feats_<label>.csv   columns: idx, f0, f1, ... f{d-1}

Kept separate from nodes_<label>.csv so the already-loaded structure/edges are
untouched; load with src/db/load_features.py.

Run:  python -m src.db.export_features
"""
from __future__ import annotations

import argparse
import csv
import os

from src.data.loader import load_graph
from src.paths import ROOT

OUT_ROOT = os.path.join(ROOT, "db_export")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+",
                    default=["finbench", "eicu", "faers", "snb"])
    args = ap.parse_args()
    for ds in args.datasets:
        data, schema = load_graph(ds, refresh_card=False)
        outdir = os.path.join(OUT_ROOT, ds)
        os.makedirs(outdir, exist_ok=True)
        for nt in schema.node_types:
            X = data[nt].x
            d = X.size(1)
            with open(os.path.join(outdir, f"feats_{nt}.csv"), "w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["idx"] + [f"f{i}" for i in range(d)])
                for i, row in enumerate(X.tolist()):
                    w.writerow([i] + row)
        print(f"{ds:9} -> feats for {len(schema.node_types)} labels "
              f"(dims {[schema.feat_dims[nt] for nt in schema.node_types]})")
    print(f"\nWrote feats_*.csv under {OUT_ROOT}/ (load with src.db.load_features)")


if __name__ == "__main__":
    main()
