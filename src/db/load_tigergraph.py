"""Load an exported graph into TigerGraph (Community Edition) via pyTigerGraph.

Generates a GSQL schema (one VERTEX per label with PRIMARY_ID idx; one DIRECTED
EDGE per relation, named <src>__<rel>__<dst> to stay unique in the heterogeneous
schema) and a loading job over the mounted db_export CSVs, then runs it.

Assumes the container mounts db_export at /home/tigergraph/db_export (see
docker-compose.yml) and services are started (`gadmin start all`).

  python -m src.db.load_tigergraph --host http://localhost --graph rcf_finbench \
      --user tigergraph --password tigergraph --dataset finbench
"""
from __future__ import annotations

import argparse
import json
import os

from src.paths import ROOT

EXPORT = os.path.join(ROOT, "db_export")
CONTAINER_EXPORT = "/home/tigergraph/db_export"   # mount point inside the container


def build_schema_gsql(graph, meta, pfx):
    # TG vertex/edge types are GLOBAL, so namespace them per dataset (datasets
    # share label names, e.g. person). Types created before the graph gathers them.
    lines = []
    for label in meta["labels"]:
        lines.append(f'CREATE VERTEX {pfx}{label} (PRIMARY_ID idx INT) '
                     f'WITH primary_id_as_attribute="true"')
    for rm in meta["relations"]:
        etype = f"{pfx}{rm['src']}__{rm['rel']}__{rm['dst']}"
        lines.append(f"CREATE DIRECTED EDGE {etype} "
                     f"(FROM {pfx}{rm['src']}, TO {pfx}{rm['dst']})")
    verts = ", ".join(f"{pfx}{l}" for l in meta["labels"])
    edges = ", ".join(f"{pfx}{rm['src']}__{rm['rel']}__{rm['dst']}"
                      for rm in meta["relations"])
    lines.append(f"CREATE GRAPH {graph}({verts}, {edges})")
    return "\n".join(lines)


def build_loading_gsql(graph, dataset, meta, pfx):
    job = f"load_{dataset}"
    defines, loads = [], []
    for label in meta["labels"]:
        f = f"{CONTAINER_EXPORT}/{dataset}/nodes_{label}.csv"
        defines.append(f'  DEFINE FILENAME v_{label} = "{f}";')
        loads.append(f'  LOAD v_{label} TO VERTEX {pfx}{label} VALUES ($"idx") '
                     f'USING HEADER="true", SEPARATOR=",";')
    for rm in meta["relations"]:
        etype = f"{pfx}{rm['src']}__{rm['rel']}__{rm['dst']}"
        f = f"{CONTAINER_EXPORT}/{dataset}/{rm['file']}"
        defines.append(f'  DEFINE FILENAME e_{etype} = "{f}";')
        loads.append(f'  LOAD e_{etype} TO EDGE {etype} VALUES ($"src", $"dst") '
                     f'USING HEADER="true", SEPARATOR=",";')
    # all DEFINE FILENAME first, then all LOAD statements
    body = ([f"USE GRAPH {graph}", f"CREATE LOADING JOB {job} FOR GRAPH {graph} {{"]
            + defines + loads + ["}"])
    return "\n".join(body), job


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="http://localhost")
    ap.add_argument("--rest-port", type=int, default=9000)
    ap.add_argument("--gs-port", type=int, default=14240)
    ap.add_argument("--user", default="tigergraph")
    ap.add_argument("--password", default="tigergraph")
    ap.add_argument("--graph", required=True)
    ap.add_argument("--dataset", required=True)
    args = ap.parse_args()

    import pyTigerGraph as tg
    meta = json.load(open(os.path.join(EXPORT, args.dataset, "schema.json")))
    conn = tg.TigerGraphConnection(host=args.host, restppPort=args.rest_port,
                                   gsPort=args.gs_port, username=args.user,
                                   password=args.password, graphname=args.graph)
    pfx = f"{args.dataset}_"          # namespace global vertex/edge types
    print("creating schema ..."); print(conn.gsql(build_schema_gsql(args.graph, meta, pfx)))
    load_gsql, job = build_loading_gsql(args.graph, args.dataset, meta, pfx)
    print("creating loading job ..."); print(conn.gsql(load_gsql))
    print("running loading job ..."); print(conn.gsql(f"USE GRAPH {args.graph}\nRUN LOADING JOB {job}"))
    print(f"loaded {args.dataset} into TigerGraph graph {args.graph}")


if __name__ == "__main__":
    main()
