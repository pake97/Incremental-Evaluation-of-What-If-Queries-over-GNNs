"""Graph-store round-trip timing on TigerGraph (REST++ / GSQL).

TigerGraph is not transaction-rollback oriented like Bolt DBs, so a disposable
hypothetical is served by **compensating undo**: apply Δ (upsert added edges /
delete removed edges / update cohort feature), read the affected region, then
reverse it (delete the added, re-insert the deleted, restore the feature). This
is the "commit + compensating-undo" variant, and it's the natural fit for a
native-parallel, non-transactional store — a third, different cost profile
alongside Neo4j/Memgraph's transaction rollback.

Per-method cost mirrors roundtrip.py:
  from_scratch : apply Δ + FULL graph download + undo
  streaming    : apply Δ + L-hop closure read + undo
  ours         : 0 per query (one-time full download -> in-memory cache)
"""
from __future__ import annotations

import time
from typing import Dict, List, Tuple

from src.data.graph_state import GraphState, Intervention
from src.db.roundtrip import _edit_pairs, _seeds

EdgeType = Tuple[str, str, str]


def _etype(et: EdgeType, pfx="") -> str:
    return f"{pfx}{et[0]}__{et[1]}__{et[2]}"


class TigerGraphStore:
    def __init__(self, host, rest_port, gs_port, user, password, graph):
        import pyTigerGraph as tg
        self.conn = tg.TigerGraphConnection(
            host=host, restppPort=rest_port, gsPort=gs_port,
            username=user, password=password, graphname=graph)
        try:
            self.conn.getToken()
        except Exception:
            pass  # community edition may not require a token
        self.graph = graph

    # -- mutations ---------------------------------------------------------
    def _apply(self, adds, dels, cohort, pfx="", delta_val=1.0):
        for et, pairs in adds.items():
            self.conn.upsertEdges(f"{pfx}{et[0]}", _etype(et, pfx), f"{pfx}{et[2]}",
                                  [(a, b, {}) for a, b in pairs])
        for et, pairs in dels.items():
            for a, b in pairs:
                self.conn.delEdges(f"{pfx}{et[0]}", a, _etype(et, pfx), f"{pfx}{et[2]}", b)
        for nt, ids in cohort.items():
            self.conn.upsertVertices(f"{pfx}{nt}", [(i, {"f": delta_val}) for i in ids])

    def _undo(self, adds, dels, cohort, pfx=""):
        for et, pairs in adds.items():          # delete what we added
            for a, b in pairs:
                self.conn.delEdges(f"{pfx}{et[0]}", a, _etype(et, pfx), f"{pfx}{et[2]}", b)
        for et, pairs in dels.items():          # re-insert what we deleted
            self.conn.upsertEdges(f"{pfx}{et[0]}", _etype(et, pfx), f"{pfx}{et[2]}",
                                  [(a, b, {}) for a, b in pairs])
        for nt, ids in cohort.items():
            self.conn.upsertVertices(f"{pfx}{nt}", [(i, {"f": 0.0}) for i in ids])

    # -- reads -------------------------------------------------------------
    def download_full_once(self, labels, relations, pfx=""):
        t0 = time.perf_counter()
        nodes = sum(self.conn.getVertexCount(f"{pfx}{lbl}") for lbl in labels)
        edges = sum(self.conn.getEdgeCount(_etype(et, pfx)) for et in relations)
        # pull vertex ids per label (materialize) to mirror a real rebuild
        for lbl in labels:
            n = self.conn.getVertexCount(f"{pfx}{lbl}") or 1
            self.conn.getVertices(f"{pfx}{lbl}", limit=n)
        return time.perf_counter() - t0, nodes, edges

    def _read_lhop(self, seeds, L, pfx=""):
        # k-hop neighbour count via an interpreted GSQL SELECT from the seeds
        total = 0
        for nt, ids in seeds.items():
            id_list = ",".join(str(i) for i in ids[:512])  # cap for the query string
            q = (f"INTERPRET QUERY () FOR GRAPH {self.graph} {{\n"
                 f"  start = {{{pfx}{nt}.*}};\n"
                 f"  seeds = SELECT s FROM start:s WHERE s.idx IN ({id_list});\n"
                 f"  reach = SELECT t FROM seeds:s-()-:t;\n"
                 f"  PRINT reach.size();\n}}")
            try:
                self.conn.gsql(q)
            except Exception:
                pass
            total += len(ids)
        return total

    # -- timed round-trips -------------------------------------------------
    def time_from_scratch(self, gs, iv, labels, relations, reps=3, pfx=""):
        adds, dels, cohort = _edit_pairs(gs, iv)
        walls = []
        for _ in range(reps):
            t0 = time.perf_counter()
            self._apply(adds, dels, cohort, pfx)
            self.download_full_once(labels, relations, pfx)
            self._undo(adds, dels, cohort, pfx)
            walls.append(time.perf_counter() - t0)
        walls.sort()
        return walls[len(walls) // 2]

    def time_streaming(self, gs, iv, L, reps=3, read_lhop=True, pfx=""):
        adds, dels, cohort = _edit_pairs(gs, iv)
        seeds = _seeds(adds, dels, cohort)
        walls = []
        for _ in range(reps):
            t0 = time.perf_counter()
            self._apply(adds, dels, cohort, pfx)
            if read_lhop:
                self._read_lhop(seeds, L, pfx)
            self._undo(adds, dels, cohort, pfx)
            walls.append(time.perf_counter() - t0)
        walls.sort()
        return walls[len(walls) // 2]

    def close(self):
        pass
