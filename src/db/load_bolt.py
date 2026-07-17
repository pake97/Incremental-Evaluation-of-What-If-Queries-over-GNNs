"""Load an exported graph into a Bolt DB (Neo4j or Memgraph).

Streams the db_export CSVs over Bolt in batched UNWIND statements — identical
Cypher for both engines, so no LOAD CSV dialect differences. Creates a per-label
index on `idx` (the tensor row index that interventions reference). For FAERS's
5.5M edges prefer neo4j-admin bulk import (see db/README.md); this streaming path
is fine for finbench/eicu/snb.

  python -m src.db.load_bolt --backend neo4j    --uri bolt://localhost:7687 \
      --user neo4j --password rcfpassword --dataset finbench
  python -m src.db.load_bolt --backend memgraph --uri bolt://localhost:7688 \
      --dataset eicu
"""
from __future__ import annotations

import argparse
import csv
import json
import os

from src.paths import ROOT

EXPORT = os.path.join(ROOT, "db_export")
BATCH = 20_000


def _index_ddl(backend: str, label: str) -> str:
    if backend == "memgraph":
        return f"CREATE INDEX ON :`{label}`(idx);"
    return (f"CREATE INDEX rcf_{label}_idx IF NOT EXISTS "
            f"FOR (n:`{label}`) ON (n.idx);")


def _rows(path):
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            yield {k: int(v) for k, v in r.items()}


def _batched(it, n):
    buf = []
    for x in it:
        buf.append(x)
        if len(buf) >= n:
            yield buf
            buf = []
    if buf:
        yield buf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["neo4j", "memgraph"], required=True)
    ap.add_argument("--uri", required=True)
    ap.add_argument("--user", default="")
    ap.add_argument("--password", default="")
    ap.add_argument("--database", default=None)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--wipe", action="store_true", help="delete all nodes first")
    args = ap.parse_args()

    from neo4j import GraphDatabase
    auth = (args.user, args.password) if args.user else None
    drv = GraphDatabase.driver(args.uri, auth=auth)
    meta = json.load(open(os.path.join(EXPORT, args.dataset, "schema.json")))
    sess_kw = {"database": args.database} if args.database else {}
    pfx = f"{args.dataset}_"          # namespace labels: datasets share names (person)
    lab = lambda x: f"{pfx}{x}"

    with drv.session(**sess_kw) as s:
        if args.wipe:
            # dataset-scoped wipe over the namespaced labels
            for label in meta["labels"]:
                s.run(f"MATCH (n:`{lab(label)}`) DETACH DELETE n")
        for label in meta["labels"]:
            s.run(_index_ddl(args.backend, lab(label)))
        # nodes
        for label in meta["labels"]:
            path = os.path.join(EXPORT, args.dataset, f"nodes_{label}.csv")
            q = f"UNWIND $rows AS r CREATE (:`{lab(label)}` {{idx: r.idx}})"
            n = 0
            for batch in _batched(_rows(path), BATCH):
                s.run(q, rows=batch)
                n += len(batch)
            print(f"  {label:14} {n:>9,} nodes")
        # edges
        for rm in meta["relations"]:
            path = os.path.join(EXPORT, args.dataset, rm["file"])
            q = (f"UNWIND $rows AS r "
                 f"MATCH (a:`{lab(rm['src'])}` {{idx: r.src}}), (b:`{lab(rm['dst'])}` {{idx: r.dst}}) "
                 f"CREATE (a)-[:`{rm['rel']}`]->(b)")
            n = 0
            for batch in _batched(_rows(path), BATCH):
                s.run(q, rows=batch)
                n += len(batch)
            print(f"  {rm['src']}-{rm['rel']}->{rm['dst']}: {n:,} edges")
    drv.close()
    print(f"loaded {args.dataset} into {args.backend} @ {args.uri}")


if __name__ == "__main__":
    main()
