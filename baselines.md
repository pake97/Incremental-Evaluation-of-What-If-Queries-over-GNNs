# Forward Counterfactual Query — Baseline Implementation Spec

**Purpose.** Implement fair, exact-comparable baselines for the *forward counterfactual query*
(evaluate `f(G^Δ)` given a cached forward pass on `G` and an intervention `Δ`). All baselines plug
into one shared harness so the only thing measured is the incremental method itself.

**Scope of this document**
- Doc 0 — Shared harness (build first; everything depends on it)
- Doc 1 — Ripple baseline (look-forward delta model, linear aggregators)
- Doc 2 — NeutronRT baseline (decoupled operators; RGCN + hGAT)

**Fixed experimental decisions**
- Backend: **NumPy / CPU** (PyTorch-CPU acceptable; no GPU).
- Models: **2-layer RGCN** and **2-layer heterogeneous GAT (hGAT)**.
- Baselines: **Ripple**, **NeutronRT**, plus **from-scratch RGCN** and **from-scratch hGAT** references.
  **InkStream is not used.**
- Datasets (all heterogeneous): **eICU, FAERS, FinBench, SNB**. Specs are schema-driven — iterate over
  whatever node/relation types the loader exposes; do not hardcode type names.
- Two experiment sets, plotted separately:
  - **Topological**: x-axis = number of edge edits ∈ {1, 100, 1000}.
  - **Feature**: x-axis = cohort size (nodes shifted) ∈ {1, 100, 1000}.
- Per experiment: 4 latency plots (one per dataset) + 4 operation-count plots (one per dataset).
- **Exactness axis**: for every method, report distance from full recomputation
  (`MSE` and `max-abs` of final-layer embeddings). Exact methods → ~1e-12; inexact → nonzero.

**Expected exactness outcomes (sanity targets)**
| Method | RGCN | hGAT |
|---|---|---|
| Our incremental | ~0 (Thm 4.4/4.5) | ~0 |
| NeutronRT | ~0 (equiv. proof) | ~0 |
| Ripple | ~0 (linear, per-relation) | **nonzero** (naive on attention — expected, demonstrates the point) |
| From-scratch | 0 (reference) | 0 (reference) |

---

## Doc 0 — Shared Harness

### 0.1 Cached forward pass

Run the full forward pass on base graph `G` **once** and cache, per layer `k ∈ {0..L-1}`:

- `Z[k]` — post-projection tensor `H[k] @ W[k]` (node × F_{k+1})
- `M[k]` — post-aggregation tensor (node × F_{k+1})
- `H[k]` — post-activation tensor; `H[0] = X` (input features)

For heterogeneous models these are stored **per node type**, and aggregation is stored/keyed **per
relation type**. Keep the raw pieces needed to invert/reapply context (see NeutronRT): for RGCN keep
per-relation degree/count; for hGAT keep per-edge attention logits and per-node attention-sum.

```python
class ForwardCache:
    # all dict-keyed by node_type (str) -> np.ndarray, and by relation_type where noted
    Z: dict[str, list[np.ndarray]]   # [layer] post-projection
    M: dict[str, list[np.ndarray]]   # [layer] post-aggregation
    H: dict[str, list[np.ndarray]]   # [layer] post-activation  (H[0] = X)
    # model-specific side state:
    deg: dict[str, np.ndarray]       # RGCN: per-relation degree/count (keyed rel->array)
    att_logits: dict                 # hGAT: per-edge raw attention logits (keyed rel->array)
    att_sum: dict[str, np.ndarray]   # hGAT: per-node attention normalizer (nbr_ctx)
```

### 0.2 Intervention object (unified update model)

`Δ = (ΔE_plus, ΔE_minus, Δκ)`:

```python
class Delta:
    edges_add:    list[tuple(rel, u, v)]        # topological insertions
    edges_del:    list[tuple(rel, u, v)]        # topological deletions
    feat_updates: list[tuple(ntype, node, k, delta_val)]  # attribute shifts x_k += delta
```

- **Topological experiment** builds `Δ` from `edges_add`/`edges_del` (size = #edits).
- **Feature experiment** builds `Δ` from `feat_updates` over a cohort (size = #nodes in cohort;
  same feature coordinate `k`, same shift `delta_val` under homogeneity).

### 0.3 The single entry point

Every baseline conforms to this interface. The harness — not the baseline — owns rollback and metrics.

```python
def evaluate(method, G, cache, delta, query) -> Result:
    counters.reset()                          # instrumentation on
    snap = snapshot_affected(G, cache, delta) # save only slices method will touch
    t0 = time.perf_counter()
    H_L_hat = method.apply(G, cache, delta)   # method mutates cache in place, returns final-layer emb
    wall = time.perf_counter() - t0
    q_val = query(H_L_hat)

    # exactness check vs from-scratch on G^Δ
    H_L_true = full_recompute(G, delta, model=method.model_ref)
    mse     = float(np.mean((H_L_hat - H_L_true) ** 2))
    max_abs = float(np.max(np.abs(H_L_hat - H_L_true)))

    restore(G, cache, snap)                   # base graph + cache pristine again
    return Result(q_val, wall, counters.edges, counters.verts, counters.macs, mse, max_abs)
```

**Rollback contract.** Baselines are free to mutate `cache`/`G` in place (Ripple and NeutronRT both
do). `snapshot_affected` records the pre-image of every slice the method will write; `restore` puts it
back. This makes the branch-evaluate-rollback loop uniform, so no baseline is penalized differently
for state handling. Snapshot only the affected region (not the whole graph) or snapshotting itself
dominates the timing — but snapshot/restore time is **excluded** from `wall` (it is harness overhead,
identical for all methods).

### 0.4 Instrumentation (the clean signal on CPU)

Wall-clock on CPU is noisy; operation counts are the comparable metric. Implement a global counter
incremented inside every method's aggregation/projection loops:

```python
class Counters:
    edges: int   # += 1 per edge whose message is (re)computed / aggregated
    verts: int   # += 1 per node whose projection or activation is (re)computed
    macs:  int   # += F_in*F_out per projection, += F per aggregation step (optional single number)
```

All three methods (ours, Ripple, NeutronRT) and from-scratch must increment the **same** counters at
the **same** granularity, or the comparison is meaningless. Define the counting points once (in shared
aggregate/project helpers) and have every method call those helpers.

### 0.5 Degeneracy flag

If the L-hop closure of `Δ` reaches ≈ all nodes (dense graph / hub edits), every method degenerates to
full recompute and plots flatten. Detect and log this rather than treating it as a bug:

```python
if affected_node_count > 0.5 * total_nodes:
    result.degenerate = True   # annotate the plot point; do not debug
```

Watch for this on the denser of {eICU, FAERS, FinBench, SNB}. Report it as a finding
(consistent with all three source papers: when updates span the graph, incremental ≈ full).

---

## Doc 1 — Ripple Baseline

**What it is.** Exact incremental inference for **linear aggregators** via a strictly look-forward
delta model with per-hop mailboxes. Native to homogeneous sum/mean/weighted-sum.

**Fit for us.**
- **RGCN** → exact. RGCN aggregation is per-relation (weighted) sum, which is linear. Run Ripple's
  delta model **once per relation type** and sum contributions. Exactness distance ~0.
- **hGAT** → **naive / nonzero distance** (agreed). Attention weights are data-dependent, which
  Ripple does not maintain (its own paper lists attention as future work). Apply the linear delta model
  as if attention weights were frozen at their base-graph values; the resulting distance is nonzero and
  is the point (motivates why hGAT needs NeutronRT / our method).

### 1.1 Data structures

```python
mailbox[node][hop]  # accumulates incremental messages; L mailboxes per node
```

### 1.2 Message construction (sum aggregation; per relation)

For a neighbor `u` of `v` along relation `r`, message into `v`'s hop-(l+1) mailbox:

- **Embedding of u changed** `h^{l-}_u -> h^l_u`:
  `m = W_l[r] @ (h^l_u - h^{l-}_u)`
- **Edge (u,v) added**: treat old embedding as zero → `m = W_l[r] @ h^l_u`
- **Edge (u,v) deleted**: treat new embedding as zero → `m = - W_l[r] @ h^{l-}_u`
- **mean / weighted-sum**: carry the α weight → `m = α (W_l[r] @ h^l_u) - α (W_l[r] @ h^{l-}_u)`
- **Feature update** (hop 0): the shifted node is the origin; emit messages to its out-neighbors.

Messages arriving at the same mailbox are **summed** (permutation-invariance).

### 1.3 Propagation loop (BSP, hop by hop)

```python
def apply(G, cache, delta):
    seed_hop0(delta)                      # feature shifts + edge endpoints -> hop-1 mailboxes
    for l in range(1, L+1):
        for v in affected_at_hop(l):
            agg_delta = sum(mailbox[v][l]) # apply phase
            h_new = cache.H[type(v)][l][v] + combine(agg_delta)   # count edge/vert here
            if h_new != cache.H[type(v)][l][v]:                   # look-forward: only if changed
                emit_messages(v, h_new, to_hop=l+1)               # compute phase
                cache.H[type(v)][l][v] = h_new
    return cache.H[...][L]
```

- **Per-relation for RGCN**: run message construction/aggregation separately per `r`, then sum into
  the node's aggregate before the update/activation.
- **hGAT naive**: reuse cached `att_sum` and per-edge attention from the base graph (do NOT recompute
  them). Aggregate `att * (W h)` deltas with frozen `att`. Distance will be nonzero — log it, don't fix it.
- Count every recomputed message as an **edge** and every recomputed node embedding as a **vertex** via
  the shared counters (Doc 0.4).

### 1.4 Notes

- CPU/NumPy to match Ripple's own implementation.
- No pruning / no selective skip beyond the look-forward "only propagate if embedding changed" test.
- Feature intervention over a cohort: Ripple must process the **entire L-hop closure** of the cohort —
  this is the case our low-rank method beats it on; make sure the counters reflect that full closure.

---

## Doc 2 — NeutronRT Baseline

**What it is.** Exact incremental inference across diverse GNNs via **fine-grained operator decoupling**
+ safe reordering. Handles GCN-style normalization and attention (GAT) exactly. This is the strong,
fair baseline for **both** RGCN and hGAT.

### 2.1 The five decoupled operators

```
ms_local(h_u, h_v)                  # edge-wise local message (independent per edge)
nbr_ctx({mlc_uv})                   # neighbor-wise context (a per-destination scalar/vector)
msg_cbn(nct_v, mlc_uv)              # combine local message with context -> final message
msg_cbn_inv(msg_uv, nct_v)         # inverse: strip context from an aggregated result
aggregate(·) ; update(a_v)          # sum aggregation ; vertex-wise update (linear + activation)
```

**RGCN row** (per relation `r`, degree normalization):
```
ms_local = 1/sqrt(d_u)              # source-side factor
nbr_ctx  = count(neighbors)         # -> d_v
msg_cbn  = mlc * 1/sqrt(nct)        # apply 1/sqrt(d_v)
msg_cbn_inv = msg * sqrt(nct)
aggregate = sum ; f_nn(h_u) = W_r h_u ; update = ReLU(W a)  # per your RGCN definition
```

**hGAT row** (per relation `r`, attention):
```
ms_local = exp(LeakyReLU(a_r^T [W_r h_v || W_r h_u]))   # raw attention logit (exp'd)
nbr_ctx  = sum over neighbors of ms_local               # attention normalizer (per dest)
msg_cbn  = mlc / nct                                     # normalized attention score
msg_cbn_inv = msg * nct
aggregate = sum ; f_nn(h_u) = W_r h_u ; update = ELU(a)
```

### 2.2 Reordered single-layer incremental update (their Algorithm 1)

For destination `v` with affected neighborhood `ΔN(v)`, cached aggregate `a_v`, cached context `nct_v`:

```python
# 1. recompute local messages for affected edges only
for u in dN(v):
    mlc[u,v] = ms_local(h[u], h[v])          # count edge

# 2. partial context update (uses old context + new local msgs)
nct_new = nbr_ctx(nct_v, {mlc[u,v] for u in dN(v)})

# 3. strip old context from cached aggregate
a_hat = msg_cbn_inv(nct_v, a_v)

# 4. partial aggregate: add deltas from affected neighbors only  (reuse unaffected sum)
a_hat = aggregate(a_hat, {mlc[u,v] * f_nn(h[u]) for u in dN(v)})   # count edges

# 5. reapply new context
a_v = msg_cbn(nct_new, a_hat)

# 6. update -> new embedding
h_v_new = update(a_v)                          # count vertex
```

This reuses the unaffected-subgraph aggregation exactly; equivalence to full recompute holds under
their Conditions 1–4 (associativity of `nbr_ctx`/`aggregate`; distributivity of `msg_cbn` over
`aggregate`; monotone `msg_cbn` for invertibility). For sum + the rows above these hold, so distance ~0.

### 2.3 Constrained vertices (hGAT correctness)

hGAT's `ms_local` depends on the **destination** embedding `h_v`. When `h_v` itself changes, all of
`v`'s incoming messages are invalidated and the incremental reuse is unsafe. For such
**destination-affected** vertices, **recompute the full neighborhood** (do not incrementally update).
Detect them during computation-graph construction (2.4) and route them to the full-recompute path.
This keeps hGAT exact (distance ~0) at the cost of extra work on those vertices — expected and
consistent with NeutronRT's reported constrained-model overhead.

### 2.4 Computation-graph construction (BFS from updates)

```python
V_curr = updated_vertices(delta)
for l in 1..L:
    E_curr = updated_edges(delta) ∪ out_edges(V_curr)     # affected edges this hop
    V_dst  = dst(E_curr)
    if constrained_model (hGAT):                          # add incoming edges of dest-affected
        for v in V_dst if v in src(E_curr):
            E_curr ∪= in_edges(v)
    V_curr = V_dst
    record subgraph(V_dst, E_curr) for layer l
```

Heterogeneous handling: process **each relation type independently** and merge per-node results before
`update` (both RGCN and hGAT).

### 2.5 Notes

- NumPy/CPU implementation (NeutronRT's real system is DGL/GPU; here we only need the algorithm,
  single-machine, in-memory — no offloading, no chunked scheduling, no PMA-CSR).
- Count edges/vertices via the shared counters so it's comparable to Ripple and ours.
- This is the baseline to headline for **topological** interventions and for **hGAT**; expect a tighter
  race than Ripple (both local), which makes the comparison honest.

---

## Deliverables checklist

- [ ] Doc 0 harness: `ForwardCache`, `Delta`, `evaluate()`, snapshot/restore, `Counters`, degeneracy flag.
- [ ] From-scratch RGCN and hGAT (reference; distance ≡ 0 by definition, full counters).
- [ ] Our incremental RGCN + hGAT wired into `evaluate()` (distance ~0 expected).
- [ ] Ripple: per-relation RGCN (exact) + naive hGAT (nonzero distance).
- [ ] NeutronRT: RGCN + hGAT with constrained-vertex recompute (exact).
- [ ] Two drivers: topological (x = edge edits) and feature (x = cohort size), each ∈ {1, 100, 1000}.
- [ ] Per dataset {eICU, FAERS, FinBench, SNB}: latency plot + operation-count plot.
- [ ] Exactness table/column: MSE + max-abs vs full recompute for every method.
- [ ] Degeneracy annotations where L-hop closure ≈ full graph.