"""Graph-store round-trip timing for the store-backed what-if workload.

Times the per-hypothetical DB work each method pays when the authoritative graph
lives in a property store (Neo4j / Memgraph via Bolt). Every hypothetical runs
inside a transaction that is ROLLED BACK, so the base graph stays pristine for
the next sibling — the disposable-overlay semantics.

Per-method cost (see db/README.md):
  from_scratch : apply Δ writes + DOWNLOAD FULL GRAPH (all nodes+edges) + rollback
  streaming    : apply Δ writes + read L-hop closure of edited endpoints + rollback
  ours         : 0 per query  (one-time: full download once -> in-memory cache)

Only Bolt DBs (Neo4j, Memgraph) here; TigerGraph goes through roundtrip_tg.py.
"""
from __future__ import annotations

import time
from typing import Dict, List, Tuple

from src.data.graph_state import GraphState, Intervention

EdgeType = Tuple[str, str, str]


def _edit_pairs(gs: GraphState, iv: Intervention):
    """Translate an intervention into per-relation add / delete idx-pairs and
    the feature cohort (node type -> ids), all in tensor-index space (= DB idx)."""
    adds: Dict[EdgeType, List[List[int]]] = {}
    dels: Dict[EdgeType, List[List[int]]] = {}
    for et, new in iv.add_edges.items():
        if new is not None and new.numel():
            adds[et] = new.t().tolist()
    for et, cols in iv.del_cols.items():
        if cols is not None and cols.numel():
            base = gs.edge_index[et][:, cols]
            dels[et] = base.t().tolist()
    cohort = {}
    for nt, delta in iv.feat_deltas.items():
        rows = (delta.abs().sum(-1) > 0).nonzero(as_tuple=False).squeeze(1)
        if rows.numel():
            cohort[nt] = rows.tolist()
    return adds, dels, cohort


def _apply_delta(tx, adds, dels, cohort, pfx="", delta_val=1.0):
    """Issue the Δ writes on an open transaction (no commit). `pfx` namespaces
    labels (datasets share label names, e.g. person)."""
    for (s, r, d), pairs in adds.items():
        tx.run(f"UNWIND $p AS p MATCH (a:`{pfx}{s}`{{idx:p[0]}}),(b:`{pfx}{d}`{{idx:p[1]}}) "
               f"CREATE (a)-[:`{r}`]->(b)", p=pairs).consume()
    for (s, r, d), pairs in dels.items():
        tx.run(f"UNWIND $p AS p MATCH (a:`{pfx}{s}`{{idx:p[0]}})-[e:`{r}`]->(b:`{pfx}{d}`{{idx:p[1]}}) "
               f"WITH e LIMIT 1 DELETE e", p=pairs).consume()
    for nt, ids in cohort.items():
        tx.run(f"UNWIND $ids AS i MATCH (n:`{pfx}{nt}`{{idx:i}}) "
               f"SET n.f = coalesce(n.f,0.0)+$dv", ids=ids, dv=delta_val).consume()


def _seeds(adds, dels, cohort):
    seeds = {}
    for (s, r, d), pairs in list(adds.items()) + list(dels.items()):
        seeds.setdefault(s, set()).update(p[0] for p in pairs)
        seeds.setdefault(d, set()).update(p[1] for p in pairs)
    for nt, ids in cohort.items():
        seeds.setdefault(nt, set()).update(ids)
    return {nt: sorted(v) for nt, v in seeds.items()}


def _read_lhop(tx, seeds, L, pfx=""):
    """Read (materialize) the L-hop closure of the edited endpoints — the
    affected region a committed-timeline system fetches to update. Expands the
    frontier one hop at a time (distinct nodes), not all-paths, so it matches
    what an incremental system actually fetches rather than enumerating paths."""
    frontier = []
    for nt, ids in seeds.items():
        frontier += tx.run(f"UNWIND $ids AS i MATCH (n:`{pfx}{nt}`{{idx:i}}) "
                           f"RETURN elementId(n) AS e", ids=ids).value("e")
    visited = set(frontier)
    for _ in range(L):
        if not frontier:
            break
        nxt = tx.run("MATCH (n)-[]-(m) WHERE elementId(n) IN $f "
                     "RETURN collect(DISTINCT elementId(m)) AS ms", f=frontier).single()["ms"]
        frontier = [e for e in nxt if e not in visited]
        visited.update(frontier)
    return len(visited)


def _download_full(tx, labels, relations, pfx=""):
    """Pull the entire graph (all node ids per label + all edges per relation) —
    what from-scratch needs to rebuild G^Δ in memory."""
    nodes = 0
    for lbl in labels:
        nodes += len(tx.run(f"MATCH (n:`{pfx}{lbl}`) RETURN n.idx AS i").values())
    edges = 0
    for (s, r, d) in relations:
        edges += len(tx.run(f"MATCH (a:`{pfx}{s}`)-[e:`{r}`]->(b:`{pfx}{d}`) "
                            f"RETURN a.idx AS a, b.idx AS b").values())
    return nodes, edges


def _download_raw(tx, schema, pfx=""):
    """DOWNLOAD step only: pull raw rows (node idx+feat, edge idx pairs) from the
    store. No tensor construction — that is a separate, method-independent step."""
    nodes = {}
    for nt in schema.node_types:
        nodes[nt] = tx.run(f"MATCH (v:`{pfx}{nt}`) RETURN v.idx AS i, v.feat AS f").values()
    edges = {}
    for (s, r, d) in schema.edge_types:
        edges[(s, r, d)] = tx.run(f"MATCH (a:`{pfx}{s}`)-[e:`{r}`]->(b:`{pfx}{d}`) "
                                  f"RETURN a.idx AS a, b.idx AS b").values()
    return nodes, edges


def _build_tensors(raw, schema):
    """BUILD step only: construct the GNN input tensors X (per node type) and A
    (edge_index per relation) from downloaded raw rows. Pure CPU, no DB — the
    same work for every method that needs G^Δ in memory."""
    import torch
    nodes, edges = raw
    x_dict = {}
    for nt in schema.node_types:
        X = torch.zeros(schema.num_nodes[nt], schema.feat_dims[nt])
        for i, f in nodes[nt]:
            X[i] = torch.tensor(f)
        x_dict[nt] = X
    ei_dict = {}
    for et in schema.edge_types:
        pairs = edges[tuple(et)]
        ei_dict[tuple(et)] = (torch.tensor(pairs, dtype=torch.long).t().contiguous()
                              if pairs else torch.zeros(2, 0, dtype=torch.long))
    return x_dict, ei_dict


def _build_XA_from_db(tx, schema, pfx=""):
    """Convenience: download + build in one call (used where the split isn't needed)."""
    return _build_tensors(_download_raw(tx, schema, pfx), schema)


class BoltStore:
    def __init__(self, uri, user, password, database=None):
        from neo4j import GraphDatabase
        auth = (user, password) if user else None
        try:
            self.drv = GraphDatabase.driver(uri, auth=auth,
                                            notifications_min_severity="OFF")
        except (TypeError, ValueError):
            self.drv = GraphDatabase.driver(uri, auth=auth)
        self.database = database

    def _session(self):
        return self.drv.session(database=self.database) if self.database else self.drv.session()

    def download_full_once(self, labels, relations, pfx="") -> Tuple[float, int, int]:
        """One-time cost ours pays at session start (and from-scratch pays every
        query). Returns (seconds, nodes, edges)."""
        with self._session() as s:
            t0 = time.perf_counter()
            tx = s.begin_transaction()
            n, e = _download_full(tx, labels, relations, pfx)
            tx.rollback()
            return time.perf_counter() - t0, n, e

    def time_from_scratch(self, gs, iv, labels, relations, reps=3, pfx=""):
        adds, dels, cohort = _edit_pairs(gs, iv)
        walls = []
        for _ in range(reps):
            with self._session() as s:
                tx = s.begin_transaction()
                t0 = time.perf_counter()
                _apply_delta(tx, adds, dels, cohort, pfx)
                _download_full(tx, labels, relations, pfx)
                tx.rollback()
                walls.append(time.perf_counter() - t0)
        walls.sort()
        return walls[len(walls) // 2]

    def time_streaming(self, gs, iv, L, reps=3, read_lhop=True, pfx=""):
        adds, dels, cohort = _edit_pairs(gs, iv)
        seeds = _seeds(adds, dels, cohort)
        walls = []
        for _ in range(reps):
            with self._session() as s:
                tx = s.begin_transaction()
                t0 = time.perf_counter()
                _apply_delta(tx, adds, dels, cohort, pfx)
                if read_lhop:
                    _read_lhop(tx, seeds, L, pfx)
                tx.rollback()
                walls.append(time.perf_counter() - t0)
        walls.sort()
        return walls[len(walls) // 2]

    def close(self):
        self.drv.close()
