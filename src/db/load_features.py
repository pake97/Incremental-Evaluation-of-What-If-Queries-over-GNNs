"""Attach node feature vectors (feats_<label>.csv) as a `feat` list property on
already-loaded nodes, for Neo4j / Memgraph. Namespaced labels (<dataset>_<label>).

  python -m src.db.load_features --backend neo4j --uri bolt://localhost:7687 \
      --user neo4j --password rcfpassword --dataset finbench
"""
from __future__ import annotations

import argparse
import csv
import json
import os

from src.paths import ROOT

EXPORT = os.path.join(ROOT, "db_export")
BATCH = 10_000


def _rows(path):
    with open(path, newline="") as fh:
        rd = csv.reader(fh)
        next(rd)  # header
        for r in rd:
            yield {"idx": int(r[0]), "feat": [float(x) for x in r[1:]]}


def _batched(it, n):
    buf = []
    for x in it:
        buf.append(x)
        if len(buf) >= n:
            yield buf; buf = []
    if buf:
        yield buf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["neo4j", "memgraph"], required=True)
    ap.add_argument("--uri", required=True)
    ap.add_argument("--user", default="")
    ap.add_argument("--password", default="")
    ap.add_argument("--dataset", required=True)
    args = ap.parse_args()

    from neo4j import GraphDatabase
    auth = (args.user, args.password) if args.user else None
    drv = GraphDatabase.driver(args.uri, auth=auth)
    meta = json.load(open(os.path.join(EXPORT, args.dataset, "schema.json")))
    pfx = f"{args.dataset}_"
    with drv.session() as s:
        for label in meta["labels"]:
            path = os.path.join(EXPORT, args.dataset, f"feats_{label}.csv")
            q = (f"UNWIND $rows AS r MATCH (n:`{pfx}{label}` {{idx: r.idx}}) "
                 f"SET n.feat = r.feat")
            n = 0
            for batch in _batched(_rows(path), BATCH):
                s.run(q, rows=batch); n += len(batch)
            print(f"  {label:14} feat set on {n:>9,} nodes")
    drv.close()
    print(f"features loaded for {args.dataset} into {args.backend}")


if __name__ == "__main__":
    main()
