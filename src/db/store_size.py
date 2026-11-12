"""Per-dataset size of the graph as stored by Neo4j and Memgraph (Table "memory" row).

Each dataset is loaded alone into a throwaway container, so the numbers are
per dataset and do not depend on what the running rcf-neo4j / rcf-memgraph
containers currently hold. The throwaway containers use host ports 17687 /
17688 and are removed afterwards. Needs docker on the host and the neo4j
driver in the venv (same as src.db.load_bolt).

Usage (from the repo root, on the Mac, not in the Linux VM):
  venv/bin/python -m src.db.store_size                       # all four datasets
  venv/bin/python -m src.db.store_size --datasets eicu       # quick check (~1 min)

Output: results/revision_v2/store_size.csv with one row per (dataset, backend):
  store_bytes  Neo4j: size of /data/databases/neo4j after a clean shutdown (checkpoint)
               Memgraph: memory_usage (memory_tracked in newer versions) after load
               minus that of the empty instance
  log_bytes    Neo4j: size of /data/transactions/neo4j (the transaction log written
               while loading, i.e. the undo information for creating the whole graph)
               Memgraph: disk_usage after load minus before (WAL + snapshots)
  load_s       wall time of src.db.load_bolt
Raw SHOW STORAGE INFO rows go to results/revision_v2/store_size_memgraph.json.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time

from src.paths import ROOT

EXPORT = os.path.join(ROOT, "db_export")
NEO_PORT, MG_PORT = 17687, 17688
NEO_AUTH = ("neo4j", "rcfpassword")


def sh(cmd: list[str], check: bool = True) -> str:
    r = subprocess.run(cmd, capture_output=True, text=True)
    if check and r.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)}\n{r.stderr}")
    return r.stdout.strip()


def rm(name: str) -> None:
    subprocess.run(["docker", "rm", "-f", "-v", name], capture_output=True)


def wait_bolt(uri: str, auth, timeout: float = 180.0):
    from neo4j import GraphDatabase
    t0 = time.time()
    while True:
        try:
            drv = GraphDatabase.driver(uri, auth=auth)
            drv.verify_connectivity()
            return drv
        except Exception as e:  # noqa: BLE001
            if time.time() - t0 > timeout:
                raise RuntimeError(f"{uri} not reachable after {timeout:.0f}s: {e}")
            time.sleep(2)


def load(backend: str, uri: str, ds: str) -> float:
    cmd = [sys.executable, "-m", "src.db.load_bolt", "--backend", backend, "--uri", uri,
           "--dataset", ds]
    if backend == "neo4j":
        cmd += ["--user", NEO_AUTH[0], "--password", NEO_AUTH[1]]
    t0 = time.time()
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"load {ds} into {backend} failed:\n{r.stderr[-2000:]}")
    return time.time() - t0


def counts(ds: str) -> tuple[int, int]:
    meta = json.load(open(os.path.join(EXPORT, ds, "schema.json")))
    return sum(meta["labels"].values()), sum(r["count"] for r in meta["relations"])


def to_bytes(v) -> int | None:
    """Memgraph reports bytes as int in older versions and as '12.3MiB' strings in newer."""
    if isinstance(v, (int, float)):
        return int(v)
    m = re.fullmatch(r"\s*([\d.]+)\s*([KMGT]?i?B)?\s*", str(v))
    if not m:
        return None
    num, unit = float(m.group(1)), (m.group(2) or "B")
    mult = {"B": 1, "KiB": 2**10, "MiB": 2**20, "GiB": 2**30, "TiB": 2**40,
            "KB": 10**3, "MB": 10**6, "GB": 10**9, "TB": 10**12}[unit]
    return int(num * mult)


def storage_info(drv) -> dict:
    with drv.session() as s:
        s.run("FREE MEMORY").consume()  # drop allocator/query memory retained after the load
        rows = s.run("SHOW STORAGE INFO").data()
    out = {}
    for r in rows:
        vals = list(r.values())
        out[str(vals[0])] = vals[1]
    return out


def measure_neo4j(ds: str) -> dict:
    name = "rcf-size-neo4j"
    rm(name)
    sh(["docker", "run", "-d", "--name", name, "-p", f"{NEO_PORT}:7687",
        "-e", f"NEO4J_AUTH={NEO_AUTH[0]}/{NEO_AUTH[1]}",
        "-e", "NEO4J_server_memory_heap_max__size=2G",
        "-e", "NEO4J_server_memory_pagecache_size=1G",
        "-e", "NEO4J_db_tx__log_preallocate=false",  # else log = 256 MB preallocated file
        "neo4j:5"])
    try:
        uri = f"bolt://localhost:{NEO_PORT}"
        drv = wait_bolt(uri, NEO_AUTH)
        load_s = load("neo4j", uri, ds)
        drv.close()
        # db.checkpoint() is Enterprise-only; a clean shutdown checkpoints too
        sh(["docker", "stop", "-t", "120", name])
        du = lambda d: int(sh(["docker", "run", "--rm", "--volumes-from", name,
                               "--entrypoint", "du", "neo4j:5", "-sb", d]).split()[0])
        store, logs = du("/data/databases/neo4j"), du("/data/transactions/neo4j")
        return {"store_bytes": store, "log_bytes": logs, "load_s": round(load_s, 1),
                "note": "store=/data/databases/neo4j after clean shutdown; log=/data/transactions/neo4j"}
    finally:
        rm(name)


def measure_memgraph(ds: str, raw: dict) -> dict:
    name = "rcf-size-memgraph"
    rm(name)
    sh(["docker", "run", "-d", "--name", name, "-p", f"{MG_PORT}:7687",
        "memgraph/memgraph-mage:latest", "--also-log-to-stderr"])
    try:
        uri = f"bolt://localhost:{MG_PORT}"
        drv = wait_bolt(uri, None)
        before = storage_info(drv)
        load_s = load("memgraph", uri, ds)
        after = storage_info(drv)
        drv.close()
        raw[ds] = {"before": before, "after": after}
        key = "memory_usage" if "memory_usage" in after else "memory_tracked"  # renamed in newer Memgraph
        mem = to_bytes(after.get(key)); mem0 = to_bytes(before.get(key))
        dsk = to_bytes(after.get("disk_usage")); dsk0 = to_bytes(before.get("disk_usage"))
        return {"store_bytes": (mem - mem0) if None not in (mem, mem0) else "",
                "log_bytes": (dsk - dsk0) if None not in (dsk, dsk0) else "",
                "load_s": round(load_s, 1),
                "note": f"store={key} delta; log=disk_usage delta (WAL+snapshots); see store_size_memgraph.json"}
    finally:
        rm(name)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["eicu", "finbench", "snb", "faers"])
    ap.add_argument("--backends", nargs="+", default=["neo4j", "memgraph"])
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "revision_v2", "store_size.csv"))
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    raw_path = os.path.join(os.path.dirname(args.out), "store_size_memgraph.json")
    raw = json.load(open(raw_path)) if os.path.exists(raw_path) else {}
    fields = ["dataset", "backend", "nodes", "edges", "store_bytes", "store_MB", "log_bytes",
              "log_MB", "bytes_per_edge_log", "load_s", "note"]
    rows = []
    if os.path.exists(args.out):
        rows = [r for r in csv.DictReader(open(args.out))
                if not (r["dataset"] in args.datasets and r["backend"] in args.backends)]
    for ds in args.datasets:
        n, e = counts(ds)
        for be in args.backends:
            print(f"== {ds} / {be} ({n:,} nodes, {e:,} edges)", flush=True)
            r = measure_neo4j(ds) if be == "neo4j" else measure_memgraph(ds, raw)
            sb, lb = r["store_bytes"], r["log_bytes"]
            rows.append({"dataset": ds, "backend": be, "nodes": n, "edges": e,
                         "store_bytes": sb, "store_MB": round(sb / 2**20, 1) if sb != "" else "",
                         "log_bytes": lb, "log_MB": round(lb / 2**20, 1) if lb != "" else "",
                         "bytes_per_edge_log": round(lb / e, 1) if lb != "" else "",
                         "load_s": r["load_s"], "note": r["note"]})
            print(f"   store {rows[-1]['store_MB']} MB, log {rows[-1]['log_MB']} MB, load {r['load_s']} s", flush=True)
            with open(args.out, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
            json.dump(raw, open(raw_path, "w"), indent=1, default=str)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
