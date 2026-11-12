# Revision plan: experiments for the SIGMOD response

Reconstructed on 2026-10-29 from what the work left behind: `revision_plan.tex` (paper
folder), `results/revision/README.md`, `results/overhead_fix/README.md`,
`configs/revision.yaml`, `src/eval/exp_revision.py`, `scripts/run_revision.sh`,
`scripts/run_overhead_fix.sh`. The original brief was written in a chat that no longer
exists; this file is the durable copy. Keep it updated here.
Consolidated results: `results/RESULTS_revision_and_overhead_fix.md` (summarized in §5 below).

## 1. Reviewer items and what serves them

| RP item (revision_plan.tex) | What was promised | Experiment | Status | Where |
|---|---|---|---|---|
| R1.O3 / R2.O1 memory | cache size per dataset (MB), one-time build cost, break-even k*, rename baseline, state caching disabled | Parts A, B, C.1, C.2, E | **done** (v1 and v2) | `results/revision/`, `results/revision_v2/`, T1, T-e, `figures/revision_breakeven.pdf`, `revision_memory_L2.pdf` |
| R1.Q2 convergence / density | closure fraction next to update ratio; behaviour at saturation | Part D (`closure_frac`, `edges_frac`) | **done** | `latency.csv`, `revision_speedup_vs_closure.pdf`, T-c |
| Fig. 3 / Fig. 4 | measured latency replacing the `latency_best_ms` projection | Part D (10 timed reps, 1 warm-up, median/p10/p90) | **done**, v1 and v2 | `revision_latency_L2.pdf`, `revision_latency_layers.pdf`, `v2_*.pdf` |
| R3.O1 cost from affected rows | re-derive the §4.1 cost from affected rows, not |Γ| | Part D (`dirty_frontier`, `sign_flips`, `edges_scanned`/`edges_touched`) | **done** | `revision_rows_vs_flips.pdf`, T-d |
| R3.O3 systems section | architecture figure, components, what is cached and at what cost | written from the v2 change list | **started** in the paper (`sections/05b_system.tex`, `figures/architecture.pdf`, Oct 26) | paper folder |
| R3.O4 Fig. 5 split, latency with deviation, lower-bound remark, operating regime | store-update vs inference panels; baselines | **not run**: no Ripple / NeutronRT timings against v2 yet | open | `src/baselines/` |
| R2.O2 sessions at scale | DAGs of 10 to 500 states, cumulative time, latency vs path depth, memory under retention policies | §7, `exp_sessions` | **done** 2026-10-31 (quiet run) | `results/sessions/`, `figures/sessions_{time,depth,memory}.pdf`, `kstar.csv` |
| R1.O1, R1.O2, R3.O2 | text only in the final RP | none needed | paper | |

## 2. Phase 1: revision measurements (`exp_revision`, parts A to E)

Design (`configs/revision.yaml`): 4 datasets × 2 archs × L = 1..4 × {feature, topo} ×
fracs {5, 10, 15, 20, 30}% = **320 settings**. Interventions sampled exactly as
`exp_scaling` (seed 42), so rows join with `results/scaling`. Parts B, C, E at L = 2.
Run: `scripts/run_revision.sh` (1 h 51 min, M3 Max). Gate: 320 rows, 0 exactness failures,
max abs err 2.68e-5 at ATOL = RTOL = 1e-4.

Findings that changed the plan (full list in `results/revision/README.md`, "Findings"):

1. Measured speedup far below the projection: ours slower than from-scratch in **194 / 320**,
   median 0.90, growing with depth (24/40/57/73 slow settings at L = 1..4). Only RGCN-feature
   (delta path) and SNB consistently faster at L = 2.
2. Cause: the closure is most of the graph at 5 to 30% (closure_frac 0.51 to 1.00;
   Spearman ρ(speedup, closure_frac) = −0.69). Note: **every frac in the design is ≤ 30%**,
   i.e. inside the regime the paper claims; there is no extended sweep to blame.
3. The `edges_touched` proxy misses O(|V|+|E|) framework work per query (edge flips,
   full-length masks, per-relation gathers, whole-tensor clones) → this motivated Phase 2.
4. Cache is 2L tensors per type (z[L-1] aliases h[L]); RGCN lazy extras are 84 MB on FAERS
   vs a 25 MB cache; building the cache costs one forward; retaining a state costs ~a full
   cache (copy-on-write at node-type granularity); peak RSS needs both columns read.
5. Store-backed: one-time cost 2.2 to 93 s, download is 92 to 98% of it; k* = 2 everywhere
   (lower bound: ignores apply/rollback in the DB); paper's 2.33 s for eICU confirmed.

## 3. Phase 2: overhead fix (`incremental_forward(impl="v2")`)

Goal: per-query cost proportional to the work done (affected rows + incident edges), eager
PyTorch, no change to what is computed. v1 untouched and still the default.

Rules the brief imposed (as referenced in the README):
- rule 1: the exactness gate runs both impls; v2 is also checked against v1 (output within
  ATOL/RTOL; `nodes_recomputed`, `edges_touched`, `sign_flips`, `dirty_frontier` equal).
- v1 code paths not modified (except the stat-only `edges_scanned` counter).

Steps: (1) profile (`src/eval/profile_overhead.py`); (2) implement `index.py` (CSR/CSC per
relation, built once per session), `overlay.py` (row-level copy-on-write), `closure_v2.py`,
`delta_v2.py`, `engine.py` dispatch; (3) re-run the revision measurements with `--impl v2`
plus Part F (compiled from-scratch, reference only), and re-time v1 under the same
conditions; (4) `v2_*` figures and tables T-a to T-f (`report_overhead_fix.py`).
Run: `scripts/run_overhead_fix.sh`.

Results (`results/overhead_fix/README.md`, v1 re-timed under the same conditions):
- exactness: pytest 98 passed; gate 1848 checks, max abs err 6.1e-5 (ATOL 1e-4); v2 vs
  from-scratch 2.7e-5; v2 vs v1 bitwise identical on closure, ≤ 3.8e-6 on delta.
- biggest v1 cost was not on the suspect list: `torch.unique` sort at `closure.py:110`
  (38% of wall on FAERS RGCN topo).
- slower than from-scratch: v1 199 → v2 **140 / 320** (all on the closure path);
  median speedup 0.90 → 1.06; at closure ≥ 95%: 0.70 → 0.91 (range 0.55 to 1.16).
- edges scanned / touched 1.96 → 1.60 (delta path 1.00); retained state SNB feature
  358 → 114 MB; index 169 MB and 0.56 s on FAERS (counted in one-time cost).
- SNB not faster than v1 (0.98 to 1.01×): per-relation Python overhead with 20 relations.
- compiled from-scratch is 1.42× faster (median); uncompiled v2 beats it in 101 / 320.
- caveat: the v2 run had BTLEServer at 100% CPU and a neo4r process at 25%.

## 4. Open items, in order

1. **Done 2026-10-30, quiet-machine re-run** of `scripts/run_overhead_fix.sh` via `run_quiet.sh`
   (logs `results/logs/*_20261030_192957.log`, top-5 CPU header). pytest 110 passed; gate 1848
   PASS, max 6.1e-5; Part D 320 rows exact. Slower than scratch: v2 149 of 320 (Oct 28 loaded
   run: 140; v1 199). Oct 28 CSVs archived in `results/revision_v2_oct28/`, not for the paper.
2. **Decided 2026-10-30: v2 only** (no v1 re-timing; see §6). Earlier text: If v2: Figures 3, 4, 5 and Table 4 all regenerate from
   v2; claims re-derived. Mixed is worse than either.
3. **Sentences that cannot survive** (full list in §5 below): "never costs more than from-scratch" (§5.2), "never
   drops below parity" (§5.2.1); and the §5.1 gate (absolute 1e-5) vs a suite passing at
   6.1e-5 under ATOL 1e-4: align the text or find the error source before Table 4.
4. **Framing of §5.2:** work saved is tight (rows/edges identical and minimal, lower-bound
   remark promised in R3.O4); time carries fixed overheads that do not shrink with the update.
   Report work and time side by side.
5. **R3.O4:** re-time Ripple and NeutronRT against v2 under the same conditions; split Fig. 5
   into store-update and inference panels; latency with max deviation in one view; report the
   compiled from-scratch baseline openly (or compile v2 too).
6. **R2.O2:** sessions-at-scale experiment (not started).
7. **SNB:** batch the per-relation operations (19 to 28% Python/dispatch in the profile).
8. Index as int32 (halves 169 MB); full-range slice skip at near-full reach was left out on
   purpose (tunes to that case).
9. Nothing is committed. Containers: `docker compose down` when done.
10. **roundtrip bug (found 2026-10-30, fixed):** `_apply_delta` deleted one edge in total
    instead of one per pair. Every earlier store-backed result that deletes edges
    (`exp_db_pipeline`, `exp_db_roundtrip`, `exp_interactive_snb`, hence the Fig. 5 store-update
    bars) under-applied deletions: **those results are invalid for deletions** and will be
    re-run with the fixed code under R3.O4 (item 5).
    `exp_interactive_eicu` only adds edges: unaffected.
11. **Store-built graph vs in-memory, hGAT:** identical graph, different edge order from the DB,
    output differs by up to 1.5e-4 (RGCN 4e-6). Decision: for store-backed runs the gate is
    graph equality (features bitwise, edge sets equal); the output difference is recorded, not
    gated. State it in §5.1 next to the tolerance.
12. **v2 retained state is dense in the last layer** (full copy of layer L per touched node
    type per retained state: 1.9 MB/state on eICU, ~44 GB for 500 states on SNB). Decide
    before the session sweep whether the last layer becomes a row overlay like the others
    (changes fig_sessions_memory and the "O(L n F) per state" sentence of the R2.O2 answer).
    **Done 2026-10-30:** layer L is a RowOverlay (h and z the same object); the dense output
    is built transiently (`closure_v2` / `delta_v2` return value, `OverlayCache.h_final`,
    not memoized); `RowOverlay.dense()` walks the chain cloning the root once. pytest 110
    passed; gate 1848 checks PASS (max 6.1e-5, unchanged). Same-process A/B on the eICU
    pilot traces: per-state time sparse/dense 0.97-1.05 (noise); retained at k = 30, retain
    all: 64.7 -> 34.8 MB. Pilot (`results/sessions_pilot_sparse/`): retained MB at k = 30
    all / 16 / base = 34.7 / 28.4 / 10.5 (config (2,8,0.3)), 39.3 / 33.8 / 10.5 (2,50,0.1).
    Session reps: 3 for retain-all, 1 for keep-16 and base-only.

## 5. Consolidated results (read this before touching §5 of the paper)

`results/RESULTS_revision_and_overhead_fix.md` is the readable overview of both experiments
(2026-10-28 runs). The facts that change the paper:

- **The submitted Fig. 3/4 plotted a projection**, `latency_best_ms` (incremental edge count
  × from-scratch per-edge time), not wall-clock. Measured time is lower by a median factor
  of 1.7 (max 10.6×). Every §5.2 / §5.2.1 number derived from those figures is affected:
  "1.9–4.7× at L=2", "14–37× at L=1", "36.6× → 3.7 → 2.0 → 1.6 on snb", "1.5–1.6× at L=4",
  and the abstract's "up to 37×". §5.3 (store-backed) and §5.4 (session) were measured and
  stand, subject to re-timing with v2.
- **Measured picture (v2):** faster than from-scratch in 180 / 320 settings, median 1.06×.
  Attribute updates on RGCN (delta path) 2–3.3× at L=2; largest win 79× (FAERS RGCN
  attribute 5%, L=1); structure updates and hGAT win only where the closure stays small
  (SNB, FinBench) and lose on eICU / FAERS (0.7–0.8×). Slow settings grow with depth
  (16 / 26 / 43 / 55 at L=1..4).
- **Closure explains it:** closure_frac 0.5–1.0 at 5–30% and L=2; Spearman ρ = −0.69.
  This is the R1.Q2 / R3.O4 answer, now with data.
- **Exactness:** 1848 checks pass at ATOL = RTOL = 1e-4, max 6.1e-5; v2 closure path
  bitwise equal to from-scratch; delta path ≤ 2.7e-5. The paper's §5.1 gate (abs 1e-5,
  rel 1e-4) and Table 4 must be updated to match.
- **Memory:** cache = 2L tensors per node type (not 2L+1); FAERS index 169 MB > cache 25 MB;
  retained state per query 2–3× smaller with v2 (SNB 358 → 114 MB).
- **Store-backed is the strong case:** one-time cost 2.2–93 s, download 92–98% of it;
  break-even k* = 2 on every dataset (lower bound, rollback cost ignored).
- **Compiled from-scratch** (reference): 1.42× faster; uncompiled v2 still wins 101 / 320.

Claims that cannot stay as written: "never costs more than from-scratch" (§5.2), "never
drops below parity" (§5.2.1), "fastest exact maintainer ... below from-scratch everywhere"
(§5.3), "up to 37×" (abstract), "significant speedups" (conclusion, unqualified).

Decision needed with the co-authors: whether the response letter states explicitly that the
revised figures replace an operation-count projection with measured wall-clock time. The
reviewers will compare the two versions; the numbers drop visibly.

## 6. Decisions taken

- **2026-10-30, paper uses v2 only.** No v1 re-timing on the quiet machine
  (`results/revision_v1_same` stays as is). `run_overhead_fix.sh` archives the Oct 28 v2
  CSVs to `results/revision_v2_oct28/` (once; `store_size*` stay) so every setting is
  re-timed. The sparse last layer (item 12) shrinks Part B retained state; T-e regenerates.
  The quiet run is `scripts/run_quiet.sh` (overhead fix, then sessions, then figures; logs
  with top-5 CPU header in `results/logs/`).

- **2026-10-29, figures.** Keep the original Figure 3 and Figure 4 layouts, with the
  measured v2 metric instead of the `latency_best_ms` projection
  (`figures/v2_scaling_L2_latency.pdf`, `v2_scaling_L2_operations.pdf` labelled as an
  operation-count proxy, `v2_scaling_speedup_flips_{feature,topo}_10pct.pdf`). No new
  figure types beyond what the revision plan promised:
  1. **Figure 3 stays as submitted (two rows: measured latency, operations as a proxy),
     x = update size 5-30%, L = 2.** No memory row and no k on this figure. Extending to
     40-50% is optional (`configs/revision_sweep.yaml`), only if a sparse dataset still shows
     a gap at 30% after the quiet re-run.
     **Memory (R1.O3 / R2.O1) is a table, not a plot** (decided 2026-10-29): the letter
     promises only the cached-tensor size per dataset and the one-time build cost. New
     columns in Table 1 (or a small table in §5.1), per dataset: base graph MB, cache MB at
     L = 2, v2 index MB, cache build time in-memory / store-backed (Parts A and C), per-query
     overlay MB at 10% (Part B `retained_hz_MB` + `retained_graph_MB`). The per-query
     peak-memory row (`fig_v2_scaling_L2_memory`, Part B `peak_over_current_MB`) is
     superseded: Part B has no warm-up and `ru_maxrss` is a lifetime high-water mark, so the
     curves are not interpretable; keep the function out of the paper figure set.
     Plus one row "graph in the store (MB)", Neo4j / Memgraph, per dataset, measured by
     `venv/bin/python -m src.db.store_size` (throwaway containers on ports 17687/17688,
     run on the Mac, ~20-30 min for all four, faers dominates) -> `results/revision_v2/
     store_size.csv`. Its `log_bytes` column (transaction log / WAL written while loading)
     divided by edges gives the per-edge undo cost of a materialised update, which is the
     number Laks asked about; reuse it in the session experiment (item 2).
     Measured 2026-10-29 (sizes valid, load times from an unchecked machine, not for the
     paper): store MB Neo4j / Memgraph = eICU 6.4 / 27.5, FinBench 16.9 / 56.9, SNB 102.2 /
     265.9, FAERS 200.6 / 714.5; log MB Neo4j / Memgraph = 17.7 / 7.8, 69.6 / 21.5, 305.0 /
     112.0, 835.9 / 293.2 (135-236 B/edge Neo4j, 56-82 B/edge Memgraph). Neo4j store is on
     disk after checkpoint, Memgraph is resident memory: label them as different quantities.
     Draft table: paper `sections/07a_memory_table.tex` (not yet \input), row filled.
     The per-query overlay column can be dropped if the session experiment (item 2) shows the
     overlays growing along the session.
  2. **Break-even k (R1.O3) is measured in the session experiment (R2.O2):** cumulative time
     vs number of queries, Inq-GNN (cache build + residual deltas) against from-scratch (one
     full pass per query), in-memory for all datasets and store-backed for eICU and FinBench;
     the crossing is k. Retained memory along the session, retain-all vs bounded, gives the
     promised memory footprint. Step sizes as in the §5.4 use case (small composed deltas),
     not the bulk updates of the sweep. The standalone break-even figure is dropped.
     **DB-side memory counterpart (Laks, decided 2026-10-29 to put it here, not in §5.1):**
     on the same x axis (states visited), the undo state a DB + GNN evaluator must keep so
     that every state stays revisitable: cumulative transaction / WAL growth along the session
     trace (Neo4j: `SHOW TRANSACTIONS YIELD allocatedDirectBytes, estimatedUsedHeapMemory`
     inside the open transaction, or size of `transactions/` after commit; Memgraph:
     `SHOW STORAGE INFO` delta or `wal/` growth), and snapshot-per-state (`graph_MB` x states)
     as a dashed upper-bound reference. No GNN involved, machine load irrelevant. Caveat for
     the text: a single DB holds one state at a time, so sibling branches of the DAG either
     cost a revert plus re-apply per visit (goes into the cumulative-time curve) or a copy per
     branch; per-transaction headers dominate for small deltas, so report as indicative.
  Figure 5 (baselines) waits for the R3.O4 re-timing against v2.

## 7. R2.O2 session experiment (planned 2026-10-30; full sweep run 2026-10-30/31, quiet machine)

**Result (`results/sessions/`, log `results/logs/run_sessions_20261030_192957.log`, 5.3 h ours +
1.4 h scratch + 32 min store).** Exactness 3520/3520 checks pass (max 5.7e-6); store graph
gate 40/40 (output diff vs in-memory max 1.5e-4, recorded, hGAT edge order). 46,736 state rows.
- Configuration (2, 8, 0.3) has at most 510 states (binary, depth 8) and reconvergences use
  branch slots, so the generator ran out on SNB (440 states) and FAERS (352). Other traces 500.
- Cumulative time at the last state, scratch / ours retain-all: eICU 0.64-1.47x, FinBench
  2.4-4.1x, SNB 1.45-2.8x, FAERS 1.26-1.55x. Deep configuration (2, 50, 0.1) is the weakest
  and there keep-16 beats retain-all (eICU 0.64x retain-all, ours slower than scratch).
- k* in memory: eICU 9 (RGCN) / 4 (hGAT), FinBench 2, SNB 2, FAERS 5; Neo4j-backed 1-2.
- Retained MB at the last state, base / keep-16 / retain-all: eICU 10 / 29-33 / 407-475,
  FinBench 66 / 137-170 / 1295-1553, SNB 418 / 1218-1406 / 13.7-14.5 GB, FAERS 194 / 348-448 /
  2.5-3.5 GB. Retain-all grows ~linearly in k; keep-16 is the policy to report as practical.
- Store-backed per state: download dominates (eICU 2.3 s, FinBench 6.5 s); apply 6 ms,
  rollback 1 ms, undo heap 2 MiB (quantum).

### Plan as drafted

**Answers.** R2.O2 (dependence on the DAG; many queries: cumulative time, latency vs path
depth, memory footprint), R1.O3 break-even k (cache + k incremental vs k from-scratch,
in-memory and store-backed), and the DB undo-state counterpart (Laks).

**Workload.** Synthetic DAG traces from `src/explore/trace.py` (exists), deltas as in §5.4:
feature delta on a 32-node cohort or 8 add + 8 del edges on the core relation,
`feature_frac 0.5`, `share_rate 0.25`. One 500-state trace per configuration; the 10 / 50 /
100 / 200 points are its prefixes, so nothing is run twice. Configurations:
`(branching, max_depth, backtrack) in {(2, 8, 0.3), (4, 8, 0.3), (2, 50, 0.1)}`; the third is
the deep one for latency vs depth. Datasets all four in-memory; store-backed (Neo4j) on
eICU and FinBench, configuration (2, 8, 0.3) only, first 100 states.

**Methods.** (a) from-scratch: one full pass per state; store-backed variant = apply delta on
an open transaction + download full graph + rollback + full pass (`src/db/roundtrip.py`, as
in `exp_interactive_eicu`). (b) ours v2 with reuse: resume from the nearest materialised
ancestor with the residual delta (`closure_forward_v2` / `delta_forward_v2` on the ancestor
OverlayCache, residual dirty sets as `src/explore/reuse.py` does for v1), retention
`max_caches in {all, 16, 1}` (1 = base only = no reuse, the ablation for "per-query cost
does not depend on the DAG"). Cache + index build counted once at state 0. No Ripple /
NeutronRT here (they belong to §5.3).

**Per-state record.** wall, depth, ancestor depth, residual size, nodes_recomputed,
edges_touched, sign_flips, retained bytes (base cache + index + sum of materialised overlay
`nbytes()`), edges applied so far (for the DB undo estimate). Exactness vs from-scratch every
10 states, untimed, ATOL = RTOL = 1e-4, stop on failure.

**Timing hygiene.** Quiet machine, v2 only, one warm-up state, ours repeated 3 times (median
cumulative), from-scratch once (500 samples per trace already show its spread).

**Figures / table (all x = number of states visited k, one panel per dataset, RGCN and hGAT
as colours as in Fig. 3).**
- F-a cumulative session time: ours retain-all (starts at the build cost) vs from-scratch;
  store-backed as a second line style on eICU / FinBench; the crossing is k*. Table of k*
  per dataset / arch / backend.
- F-b per-query latency vs path depth (deep configuration), ours median per depth with
  sign_flips or residual size alongside, from-scratch as the flat reference; retain-all vs
  base-only shows the DAG dependence.
- F-c retained memory vs k: ours retain-all, ours 16, base only; DB-side: undo log
  = `log bytes/edge` (store_size.csv: Neo4j 135-236, Memgraph 56-82 B/edge) x edges applied
  on the path, cumulative over retained states; snapshot-per-state = store MB x k as the
  dashed upper bound. Caveat in the text: one DB holds one state, so siblings cost a revert
  plus re-apply per visit or a copy per branch.

**Runtime estimate.** Ours negligible. From-scratch in-memory per state ~ eICU 0.03 s,
FinBench 0.2, SNB 0.6, FAERS 1.0 -> 500 states x 3 configs x 2 archs ~ 1.5 h on FAERS, 50 min
SNB, minutes elsewhere. Store-backed from-scratch ~2.3 s (eICU) / 7 s (FinBench) per state x
100 states x 2 archs ~ 8 + 23 min.

**Code.** `src/eval/exp_sessions.py` + `configs/sessions.yaml` + `scripts/run_sessions.sh`;
extend `run_trace` with an `impl="v2"` branch and retained-bytes accounting; figures
`fig_sessions_{time,depth,memory}` in `src/eval/figures.py`; results in `results/sessions/`.

## 8. Reproduce

```bash
scripts/run_revision.sh                                   # phase 1 (v1)
scripts/run_quiet.sh                                      # paper run: overhead_fix + sessions + figures, logged
scripts/run_sessions.sh                                   # R2.O2 only (resumable; results/sessions/)
venv/bin/python -m src.eval.figures --sessions            # fig_sessions_{time,depth,memory}, kstar.csv
scripts/run_overhead_fix.sh                               # phase 2: pytest, gate, profiles, revision_v2, figures, tables
venv/bin/python -m src.eval.exp_revision --config configs/revision.yaml --impl v2 --exp revision_v2 --parts A C D B E F report
venv/bin/python -m src.eval.figures --revision --v2
```
