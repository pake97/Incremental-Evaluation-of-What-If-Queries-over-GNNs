# Store-backed what-if experiment — DB setup

**Goal.** Measure the per-hypothetical **graph-store round-trip** a committed-timeline
system pays to serve a what-if on a property graph: apply Δ (create/delete edges,
update node props), optionally read the L-hop closure of the edited endpoints, then
**roll back** so the base stays pristine for the next sibling hypothetical. Our
overlay method touches the DB **zero** times per query (graph held in memory), so
this cost is exactly the deployment penalty of reason (iii). We time it on Neo4j,
Memgraph, and TigerGraph and add it to the streaming baselines.

## One-command reproducible stack (Docker)

```bash
python -m src.db.export_csv                 # writes db_export/<dataset>/  (done)
docker compose up -d                        # Neo4j:7687, Memgraph:7688, TigerGraph:9000/14240
docker compose exec tigergraph bash -lc 'gadmin start all'   # TigerGraph only
./db/load_all.sh finbench eicu              # stream graphs into Neo4j + Memgraph
# add TigerGraph:  DBS="neo4j memgraph tigergraph" ./db/load_all.sh finbench eicu
```

Compose defaults (match `configs/db.yaml.template`): Neo4j `bolt://localhost:7687`
user `neo4j` / pass `rcfpassword`; Memgraph `bolt://localhost:7688`, no auth;
TigerGraph `http://localhost:9000`, `tigergraph`/`tigergraph`. Neo4j & Memgraph
share the Bolt protocol but run on different host ports so both are up at once.
The loader streams CSVs over Bolt (`src/db/load_bolt.py`, portable Cypher — no
LOAD CSV dialect issues) and builds the per-label `idx` index automatically. For
FAERS (5.5M edges) prefer `neo4j-admin database import` for speed.

## Data contract (already generated)

`python -m src.db.export_csv` wrote `db_export/<dataset>/`:
- `nodes_<label>.csv` — column `idx` (the node's **tensor row index**, unique within a label)
- `edges_<src>__<rel>__<dst>.csv` — columns `src,dst` (tensor idx of endpoints)
- `schema.json` — labels, relations, counts

**The node key is `(label, idx)`.** `idx` is exactly the index our interventions use,
so no translation table is needed. Every DB must be loaded so a node is addressable by
its label + `idx`, and edges are typed by relation.

Sizes: finbench 55.6k/310k, eicu 6.3k/137k, snb 367k/1.44M, faers 25.9k/5.5M.
Start with **finbench + eicu** to validate the pipeline, then add snb/faers.

## Per-method DB cost model (what roundtrip.py will charge)

The store (Neo4j/Memgraph/TigerGraph) is the authoritative graph. Costs per
hypothetical Δ, all inside a transaction that is ROLLED BACK so the base stays
pristine for the next sibling:

| method | one-time, at session start | per hypothetical (DB) |
|---|---|---|
| **from-scratch** | — | apply Δ (writes) + **download the FULL graph** (all nodes+edges) to rebuild G^Δ in memory + rollback |
| **streaming** (Ripple / NeutronRT) | load once | apply Δ (writes) + read the **L-hop closure** of edited endpoints + rollback |
| **ours** | **one full download + one forward → cache `f(G)` in RAM** | **0 DB queries** — Δ is an in-memory overlay on the cached tensors |

Key point: because what-ifs never commit, the base graph in the store never
moves, so the in-memory cache `f(G)` stays valid for the whole session and is
built exactly once. From-scratch re-downloads the entire graph every query; a
committed-timeline system must at least read the affected subgraph and
apply/rollback each query. We report the one-time load separately and a
**break-even** (hypotheticals after which the resident cache beats
re-downloading — typically 1–2). This is the store-backed instance of the
warm / cold-pipeline / resident-system timing model in results/README.md.

## What the harness will query (so you index correctly)

Per hypothetical, inside one transaction, then ROLLBACK:
1. **point lookups** by `(label, idx)` — needs an index/constraint on each label's `idx`;
2. **edge create** for ΔE+ and **edge delete** for ΔE- on one relation type;
3. **node property update** for feature Δ (a scalar prop `f` on a cohort of nodes);
4. optional **L-hop neighborhood read** from the edited endpoints (2 hops).

## Neo4j (5.x)

```cypher
// one index per label (repeat for every label in schema.json)
CREATE INDEX idx_account IF NOT EXISTS FOR (n:account) ON (n.idx);
```
Load (from the DB import dir, or use LOAD CSV with file:/// ):
```cypher
LOAD CSV WITH HEADERS FROM 'file:///finbench/nodes_account.csv' AS r
CALL { WITH r CREATE (:account {idx: toInteger(r.idx)}) } IN TRANSACTIONS OF 50000 ROWS;

LOAD CSV WITH HEADERS FROM 'file:///finbench/edges_account__transfer__account.csv' AS r
CALL { WITH r
  MATCH (a:account {idx: toInteger(r.src)}), (b:account {idx: toInteger(r.dst)})
  CREATE (a)-[:transfer]->(b) } IN TRANSACTIONS OF 50000 ROWS;
```
(For FAERS's 5.5M edges use `neo4j-admin database import` for speed.)
Driver: `pip install neo4j`.

## Memgraph

Bolt + Cypher, so the same `CREATE INDEX` / `LOAD CSV` statements work (via `mgconsole`
or the Neo4j Python driver). Default has **no auth**. If co-hosting with Neo4j, run it
on a different Bolt port. Driver: the same `neo4j` package, or `pip install gqlalchemy`.

## TigerGraph

Different model — needs a GSQL schema + loading job:
```gsql
CREATE GRAPH rcf()
// per label:
CREATE VERTEX account (PRIMARY_ID idx INT)
// per relation (directed):
CREATE DIRECTED EDGE transfer (FROM account, TO account)
// then a LOADING JOB mapping nodes_*.csv -> vertices, edges_*.csv -> edges
```
Driver: `pip install pyTigerGraph`. Expose REST++ (9000) and GraphStudio (14240).

## What to give me back

Fill `configs/db.yaml` (copy of `configs/db.yaml.template`, gitignored) with, **for each
DB you brought up**:
- connection: Neo4j/Memgraph Bolt URI + user/password + database; TigerGraph host + ports
  + user/password + graph name;
- set that DB's `enabled: true`;
- `datasets:` — which datasets you actually loaded (must exist in `db_export/`);
- confirm the per-label `idx` indexes exist.

Then tell me it's ready and I'll build `src/db/roundtrip.py` (the timing method) and
`exp_db_roundtrip.py`, run it against your instances, and fold the DB cost into the
what-if workload comparison (`whatif_ms = maintenance + DB apply/rollback [+ L-hop read]`).
