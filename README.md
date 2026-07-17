# Incremental What-if Queries over GNNs

Experimental harness for a paper on **answering what-if (forward counterfactual)
queries over GNNs incrementally**. A what-if query evaluates a *frozen* trained
GNN on a hypothetically modified graph `G^Δ = G ⊕ Δ` and reads a query `Q` off the
output. Instead of recomputing the GNN from scratch, we **incrementally maintain**
cached forward state, and **reuse** it across an exploration graph of many related
hypotheticals.

The two claims the experiments demonstrate:

1. **Exactness** — the incremental answer equals the from-scratch answer
   (elementwise, within `ATOL/RTOL`). This is a hard gate: a result that fails it is
   never benchmarked or plotted.
2. **Efficiency** — the incremental answer is (often) much cheaper, especially in a
   store-backed deployment and across a composing exploration session.

Two intervention families, both exact:
- **Attribute (feature) Δ** — shift one feature coordinate over a cohort; maintained
  by low-rank propagation + a sparse ReLU sign-flip correction.
- **Structural (topological) Δ** — insert/delete edges; maintained by recomputing
  only the L-hop closure of the edited endpoints (+ degree/attention refresh).

Architectures: **RGCN** (mean aggregator) and **hGAT** (attention). Tasks: node
classification, link prediction, link regression. Depths L = 1..4.

---

## Repository layout

```
src/
  paths.py            # canonical repo paths
  data/               # per-dataset builders -> one common in-memory graph format
    finbench/ faers/ eicu/ snb/   (each has a README with modeling choices)
    graph_state.py    # copy-on-write Δ application (Intervention / GraphView)
    loader.py schema.py
  models/             # RGCN + hGAT encoders, 3 task heads, training (train.py)
  forward/            # from-scratch inference (the ground truth)
  incremental/        # cache.py (ForwardCache), closure.py, delta.py, engine.py
  explore/            # exploration DAG: state.py (compose), reuse.py, trace.py, analyst.py
  baselines/          # Ripple + NeutronRT re-implementations, shared op-counters
  db/                 # graph-store backends (Neo4j/Memgraph/TigerGraph) + loaders
  eval/               # experiments (exp_*.py), figures.py, runner.py, exactness.py
scripts/              # one run_<exp>.sh per experiment (+ run_all.sh)
configs/              # one yaml per experiment (no magic numbers in code)
tests/               # pytest; exactness tests gate everything
docker-compose.yml    # Neo4j + Memgraph (+ TigerGraph) for store-backed experiments

# regenerated, git-ignored (see .gitignore):
dataset/  checkpoints/  results/  figures/  db_export/
```

Python entrypoints run as modules from the repo root, e.g.
`venv/bin/python -m src.eval.exp_scaling --config configs/scaling.yaml`.

---

## 1. Environment

```bash
python3 -m venv venv
venv/bin/pip install -r requirements.txt
```

PyTorch + PyTorch Geometric on CPU is sufficient (the graphs are small–medium).
Determinism: every experiment sets all seeds, enables deterministic algorithms, and
logs the seed + library versions into its results manifest. Float-equivalence
tolerances are `ATOL=1e-5`, `RTOL=1e-4` (in `configs/common.yaml`).

---

## 2. Data

All four datasets load into **one common hetero-graph format**. Raw data goes under
`dataset/<name>/` (git-ignored); each `src/data/<name>/README.md` documents the
schema and modeling choices.

| dataset | source | access | notes |
|---|---|---|---|
| **finbench** | LDBC FinBench | open | native property graph — the reference dataset |
| **snb** | LDBC SNB | open | native property graph |
| **eicu** | PhysioNet eICU-CRD demo 2.0.1 | **credentialed** | constructed graph (see README) |
| **faers** | FDA FAERS | open (large) | constructed from relational tables |

> eICU/FAERS/PhysioNet raw data is access-controlled — download it yourself into the
> paths in `src/data/<name>/README.md`. Never commit raw data (it is git-ignored).

Build the processed graphs (idempotent — only builds what is missing):

```bash
./scripts/run_prepare_data.sh
# or individually:
venv/bin/python -m src.data.finbench.prepare_finbench
venv/bin/python -m src.data.snb.build_snb
venv/bin/python -m src.data.eicu.build_eicu
venv/bin/python -m src.data.faers.scan_faers && venv/bin/python -m src.data.faers.build_faers
```

Each build emits a `dataset_card.json` (|V|, |E| by type, feature dims, tasks, splits).

---

## 3. Models & checkpoints

Train **once** per `(dataset, arch, task, L)`, checkpoint, and **freeze** — every
what-if experiment loads a frozen checkpoint (we never retrain inside a query).
Checkpoints are git-ignored and regenerated:

```bash
./scripts/run_train.sh                      # trains every missing checkpoint
# under the hood, per dataset:
venv/bin/python -m src.eval.ensure_ckpts --dataset finbench --train
```

Checkpoints land in `checkpoints/<dataset>/<task>_<arch>_L<n>.pt`.

---

## 4. The exactness gate (run this first)

Every incremental path ships with an exactness-equivalence test against
from-scratch. **Timing is meaningless until this passes.**

```bash
./scripts/run_exactness.sh
# = venv/bin/python -m src.eval.exp_exactness --config configs/exactness.yaml
```

Writes `results/exactness/`. Also run the unit tests:

```bash
venv/bin/pytest -q
```

---

## 5. Experiments

Each experiment is **idempotent, resumable, and self-describing**: it writes
`results/<exp>/manifest.json` (config, seed, library versions, dataset cards) and a
tidy CSV/JSON. Every `run_<exp>.sh` first ensures the needed checkpoints exist.

| # | experiment | script | module | config | needs DB |
|---|---|---|---|---|---|
| 1 | exactness gate | `run_exactness.sh` | `exp_exactness` | `exactness.yaml` | no |
| 2 | single-query latency | `run_single_query_efficiency.sh` | `exp_single_query` | `single_query.yaml` | no |
| 3 | layer scaling (L=1..4) | `run_layer_scaling.sh` | `exp_layer_scaling` | `layer_scaling.yaml` | no |
| 4 | size scaling (5–30%) | `run_scaling.sh` | `exp_scaling` | `scaling.yaml` | no |
| 5 | overhead (proxy vs wall) | `run_overhead.sh` | `exp_overhead` | `overhead.yaml` | no |
| 6 | full single-query grid | `run_grid.sh` | `exp_grid` | `grid.yaml` | no |
| 7 | baselines (Ripple/NeutronRT) | `run_baselines.sh` | `exp_baselines` | `baselines.yaml` | no |
| 8 | exploration reuse (DAG) | `run_exploration_reuse.sh` | `exp_exploration` | `exploration_reuse.yaml` | no |
| 9 | store round-trip | `run_db.sh` | `exp_db_roundtrip` | `db.yaml` | **yes** |
| 10 | store per-step pipeline | `run_db_pipeline.sh` | `exp_db_pipeline` | `db.yaml` | **yes** |
| 11 | interactive session — eICU | `run_interactive_eicu.sh` | `exp_interactive_eicu` | (in-code) | **yes** |
| 12 | interactive session — SNB | `run_interactive_snb.sh` | `exp_interactive_snb` | (in-code) | **yes** |
| – | torch.compile diagnostic | `run_compile.sh` | `bench_compile` | – | no |

Run any experiment directly, e.g.:

```bash
venv/bin/python -m src.eval.exp_scaling --config configs/scaling.yaml
```

### Details per experiment

- **exactness** (`exp_exactness`) — asserts incremental == from-scratch on many random
  Δ for every (dataset, arch, task, intervention type). Gate. → `results/exactness/`.
- **single-query latency** (`exp_single_query`) — incremental vs from-scratch latency
  for attribute & structural Δ. Reports warm / cold / resident timings and the
  break-even. `reps: 9` (median). → `results/single_query/`.
- **layer scaling** (`exp_layer_scaling`) — the above swept over L = 1..4.
- **size scaling** (`exp_scaling`) — Δ size as a fraction of the dataset {5,10,15,20,30}%
  (nodes for attribute, core-relation edges for structure), swept over arch × kind ×
  frac × L. Feeds `scaling_L2_*`, `scaling_layers_*`, `scaling_signflips_*`,
  `scaling_speedup_flips_*`. → `results/scaling/`.
- **overhead** (`exp_overhead`) — the algorithmic proxy (edges/nodes) vs wall-clock and
  the framework tax between them (`wall_speedup = proxy_speedup / overhead_ratio`).
  The proxy is the primary efficiency metric; wall-clock is the prototype lower bound.
- **grid** (`exp_grid`) — full kind × size × L × arch single-query grid per dataset.
- **baselines** (`exp_baselines`) — ours vs **Ripple** (frozen degree/attention) vs
  **NeutronRT** (exact decoupled operators) vs from-scratch: op-counts, latency, and
  exactness. Ours & NeutronRT are exact everywhere; Ripple is exact only for RGCN +
  attribute Δ.
- **exploration reuse** (`exp_exploration`) — the DAG experiment under three rungs:
  from-scratch per state, incremental-no-reuse, incremental+reuse (resume from nearest
  materialized ancestor). Headline: reuse benefit vs prefix-sharing / branching.
- **figures** — see §6.

### Store-backed experiments (9–12) need the graph DBs

These model the naive tool: for each hypothetical the baselines apply Δ to the store
and **re-fetch the graph**, while ours reads it once and composes on a resident cache.

```bash
# 1. bring up the stack (Neo4j on 7687, Memgraph on 7688)
docker compose up -d

# 2. export CSVs + load the datasets you need (idempotent; --wipe inside)
./db/load_all.sh finbench eicu snb        # DBS="neo4j memgraph" by default

# 3. create your local DB config (git-ignored) from the template
cp configs/db.yaml.template configs/db.yaml   # defaults already match compose

# 4. run
./scripts/run_db.sh                 # per-hypothetical round-trip timing
./scripts/run_db_pipeline.sh        # per-step (download / update / GNN) breakdown
./scripts/run_interactive_eicu.sh   # composing-session case study (RGCN + hGAT)
./scripts/run_interactive_snb.sh
```

Notes:
- The Bolt loaders `--wipe` per dataset (labels are namespaced `<dataset>_<label>` so
  datasets coexist). If you re-load without `--wipe` you get duplicate edges.
- SNB is large; a single `DETACH DELETE` can exceed Neo4j's transaction memory —
  `--wipe` deletes per label, restart Neo4j first if it has stale transaction memory.
- The interactive studies verify the DB-materialized base graph is elementwise
  identical to the local processed graph before timing.
- `db_max_edits` in `configs/db.yaml` caps how many edge writes are actually issued to
  the store (so a huge Δ can't OOM a transaction); the GNN steps still use full size.

Outputs: `results/db_roundtrip/`, `results/db_pipeline/`, `results/interactive_eicu/`,
`results/interactive_snb/`.

---

## 6. Figures

```bash
./scripts/run_figures.sh
# = venv/bin/python -m src.eval.figures
```

Reads whatever is in `results/` and writes to `figures/` (git-ignored). Figures whose
inputs are absent are skipped. The interactive-session figures (PDF + PNG) are written
by their own experiments.

---

## 7. Full pipeline

```bash
./scripts/run_all.sh
```

Runs, in dependency order: data prep → train → **exactness gate** → single-query →
layer-scaling → overhead → grid → baselines → DB → compile diagnostic →
exploration reuse → figures. It fails loudly and early; a failed exactness gate stops
everything downstream. (The store-backed experiments are skipped unless
`configs/db.yaml` is present and the stack is up.)

---

## 8. Reproducibility

- Fixed seeds; deterministic algorithms; seed + library versions logged into every
  `results/<exp>/manifest.json` alongside the dataset cards it ran on.
- Re-running an experiment with the same config reproduces its numbers.
- Primary efficiency metric is the implementation-independent proxy (edges aggregated /
  nodes recomputed); wall-clock is reported as the unoptimized-prototype lower bound
  (see `results/README.md`, "Two currencies").

## Notes for reviewers

Baselines are **internal ablations** (from-scratch, incremental-no-reuse,
incremental+reuse) plus two re-implemented published primitives (Ripple, NeutronRT).
There is no external competitor system: streaming-GNN systems maintain a single
committed timeline with no branching/backtracking, so they do not answer the
exploration workload evaluated here. This framing is stated in `results/README.md`.
# Incremental-Evaluation-of-What-If-Queries-over-GNNs
