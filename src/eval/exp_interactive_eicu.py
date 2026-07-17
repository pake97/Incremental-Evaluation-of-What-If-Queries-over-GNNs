"""Interactive what-if session on eICU / hGAT, backed by a real Neo4j store.

Shows, across a COMPOSING what-if session (a DAG with one branch + one backtrack):
  * OURS keeps the base graph + forward cache RESIDENT: reads the graph from
    Neo4j ONCE at S0, then composes each update on the resident cache (reuse:
    resume from the nearest materialized ancestor) with NO write-back and NO
    re-fetch. Exact (== from-scratch) at every state.
  * FROM-SCRATCH (ground truth) and RIPPLE model the naive store-backed tool:
    for every hypothetical state they APPLY the state's updates to Neo4j and
    RE-FETCH the full graph, then compute; the mutation is ROLLED BACK (txn
    rollback) so the next state starts clean.
  * RIPPLE (exact only for linear aggregation; forced to FREEZE attention on
    hGAT) additionally DRIFTS, with query error COMPOUNDING across composed
    steps.

Query Q = high-risk fraction over a cohort = (#cohort stays with mortality
risk >= tau) / |cohort|.  Label node type = stay; risk = sigmoid(node head).

Everything is REUSED, not reimplemented: ForwardCache/build_cache, the
Intervention/compose API, the shared Counters, whatif_forward (from-scratch),
the incremental maintainer (closure/delta via the reuse engine), the Ripple
baseline (frozen attention on hGAT), and the Neo4j BoltStore round-trip.

  venv/bin/python -m src.eval.exp_interactive_eicu
"""
from __future__ import annotations

import csv
import gzip
import json
import os
import time

import torch

from src.data.graph_state import GraphState, Intervention
from src.data.loader import load_graph
from src.db.roundtrip import (BoltStore, _apply_delta, _download_raw,
                              _edit_pairs)
from src.eval.exactness import check
from src.explore.state import Registry, compose
from src.forward.forward import set_determinism, whatif_forward
from src.incremental.cache import build_cache
from src.incremental.closure import closure_forward
from src.incremental.delta import delta_forward
from src.incremental.engine import (feat_dirty_rows, is_topological,
                                    struct_dirty_rows)
from src.baselines.harness import BaselineCache
from src.baselines.methods import NeutronRT, Ripple
from src.models.model import load_model
from src.paths import CHECKPOINTS, EICU_PROC, EICU_RAW, FIGURES, RESULTS

# ---- fixed configuration (reported into results.json) ---------------------
SEED = 42
DATASET, ARCH, TASK, L = "eicu", "hgat", "nodelabel", 2
NEO4J = dict(uri="bolt://localhost:7687", user="neo4j", password="rcfpassword")
PFX = "eicu_"
AGE_CUTOFF = 60.0            # "over-60" cohort (real ages parsed from demo CSVs)
LABEL_NT = "stay"
MED_NT = "medication"
RECV = ("stay", "received", "medication")
HAS_STAY = ("patient", "has_stay", "stay")
Q_THRESHOLD = 0.15          # bonus: target decision "high-risk fraction <= 15%"

RESULTS_DIR = os.path.join(RESULTS, "interactive_eicu")
FIG_DIR = FIGURES


# ---------------------------------------------------------------------------
# cohort selection (real over-60 patients -> their stays -> their medications)
# ---------------------------------------------------------------------------
def parse_age(s):
    if not s:
        return None
    if ">" in s:
        return 90.0
    try:
        return float(s)
    except ValueError:
        return None


def over60_patient_idx():
    """Real over-60 patient node indices, via demo CSV ages + id_maps."""
    raw = os.path.join(EICU_RAW, "patient.csv.gz")
    id_maps = json.load(open(os.path.join(EICU_PROC, "id_maps.json")))
    pat_map = id_maps["patient"]                       # uniquepid -> idx
    first_age = {}
    with gzip.open(raw, "rt", newline="") as fh:
        for r in csv.DictReader(fh):
            p = r["uniquepid"]
            if p not in first_age:
                first_age[p] = parse_age(r["age"])
    idx = sorted(pat_map[p] for p, a in first_age.items()
                 if p in pat_map and a is not None and a > AGE_CUTOFF)
    return torch.tensor(idx, dtype=torch.long)


def stays_of_patients(gs, patient_idx):
    ei = gs.edge_index[HAS_STAY]                       # [2, E] = (patient, stay)
    sel = torch.isin(ei[0], patient_idx)
    return torch.unique(ei[1][sel])


def meds_of_stays(gs, stay_idx):
    ei = gs.edge_index[RECV]                           # [2, E] = (stay, med)
    sel = torch.isin(ei[0], stay_idx)
    return torch.unique(ei[1][sel])


# ---------------------------------------------------------------------------
# intervention builders (node feature shift; edge add) — via the Δ API
# ---------------------------------------------------------------------------
def iv_feat(gs, nt, rows, delta, coord=0):
    d = torch.zeros_like(gs.x[nt])
    d[rows, coord] = delta
    return Intervention(feat_deltas={nt: d})


def iv_add_recv(stay_ids, med_id):
    """Add a 'second drug' edge stay->med (a common medication) for a cohort."""
    src = stay_ids.to(torch.long)
    dst = torch.full((src.numel(),), int(med_id), dtype=torch.long)
    return Intervention(add_edges={RECV: torch.stack([src, dst])})


# ---------------------------------------------------------------------------
# query
# ---------------------------------------------------------------------------
def risk_vec(model, h):
    return torch.sigmoid(model.classify_nodes(h, LABEL_NT))    # [N_stay]


def query_Q(model, h, cohort, tau):
    r = risk_vec(model, h)[cohort]
    return float((r >= tau).float().mean()), r


def emb_err(hb, ha, nt):
    d = (hb[nt] - ha[nt]).abs()
    return float(d.max()), float(d.mean())


def med_time(fn, reps=3):
    """Median wall-time (ms) of a compute call."""
    xs = []
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); xs.append(time.perf_counter() - t0)
    xs.sort()
    return xs[len(xs) // 2] * 1e3


# ---------------------------------------------------------------------------
# DB metrics (baseline naive-tool round-trip): apply Δ + full re-fetch + rollback
# ---------------------------------------------------------------------------
def _bytes_of(raw, schema):
    nodes, edges = raw
    b = 0
    for nt in schema.node_types:
        for _idx, feat in nodes[nt]:
            b += 8 * (1 + (len(feat) if feat is not None else 0))
    for et in schema.edge_types:
        b += 16 * len(edges[tuple(et)])
    return b


def db_roundtrip(store, gs, iv, schema, reps=3):
    """Model the naive tool for one state: apply the state's Δ to Neo4j, RE-FETCH
    the full graph, roll back. Returns write-op count, refetch node/edge counts,
    bytes, and median round-trip wall-time. Rollback = transaction rollback."""
    adds, dels, cohort = _edit_pairs(gs, iv)
    n_writes = (sum(len(p) for p in adds.values())
                + sum(len(p) for p in dels.values())
                + sum(len(v) for v in cohort.values()))
    walls, raw = [], None
    for _ in range(reps):
        with store._session() as s:
            tx = s.begin_transaction()
            t0 = time.perf_counter()
            _apply_delta(tx, adds, dels, cohort, PFX)      # DB writes
            raw = _download_raw(tx, schema, PFX)           # full re-fetch
            tx.rollback()                                  # disposable overlay
            walls.append(time.perf_counter() - t0)
    walls.sort()
    n_nodes = sum(len(raw[0][nt]) for nt in schema.node_types)
    n_edges = sum(len(raw[1][tuple(et)]) for et in schema.edge_types)
    return dict(db_writes=n_writes, refetch_nodes=n_nodes, refetch_edges=n_edges,
                refetch_bytes=_bytes_of(raw, schema),
                roundtrip_ms=round(walls[len(walls) // 2] * 1e3, 2))


def db_download_once(store, schema, reps=3):
    """The single load OURS pays at S0 (and its equality-check payload)."""
    walls, raw = [], None
    for _ in range(reps):
        with store._session() as s:
            tx = s.begin_transaction()
            t0 = time.perf_counter()
            raw = _download_raw(tx, schema, PFX)
            tx.rollback()
            walls.append(time.perf_counter() - t0)
    walls.sort()
    n_nodes = sum(len(raw[0][nt]) for nt in schema.node_types)
    n_edges = sum(len(raw[1][tuple(et)]) for et in schema.edge_types)
    return raw, dict(refetch_nodes=n_nodes, refetch_edges=n_edges,
                     refetch_bytes=_bytes_of(raw, schema),
                     roundtrip_ms=round(walls[len(walls) // 2] * 1e3, 2))


def verify_db_matches_local(raw, schema, gs):
    """All three methods must start from the SAME base graph. Rebuild tensors
    from the Neo4j download and assert they equal the local processed graph."""
    from src.db.roundtrip import _build_tensors
    x_db, ei_db = _build_tensors(raw, schema)
    xmax = max(float((x_db[nt] - gs.x[nt]).abs().max()) for nt in gs.node_types)
    edge_ok = all(ei_db[et].size(1) == gs.edge_index[et].size(1)
                  for et in gs.edge_types)
    return xmax, edge_ok


# ---------------------------------------------------------------------------
# OURS: resident-cache reuse (resume from nearest materialized ancestor)
# ---------------------------------------------------------------------------
def _nearest_ancestor(caches, state):
    best = None
    for s in caches:
        if s <= state and (best is None or len(s) > len(best)):
            best = s
    return best if best is not None else frozenset()


def ours_reuse(model, gs, reg, caches, state):
    """One incremental step on the resident cache; materializes the result.
    Uses the closure path (exact for hGAT). Returns (h, edges_touched, info)."""
    anc = _nearest_ancestor(caches, state)
    residual = state - anc
    iv_full = compose(reg, state)
    iv_res = compose(reg, residual)
    cache = caches[anc]
    view = gs.apply(iv_full)
    h, new_cache, stats = closure_forward(
        model, cache, view,
        feat_dirty=feat_dirty_rows(iv_res),
        struct_dirty=struct_dirty_rows(gs, iv_res))
    caches[state] = new_cache
    return h, stats["edges_touched"], dict(ancestor_depth=len(anc),
                                           residual_size=len(residual))


# ---------------------------------------------------------------------------
# the session
# ---------------------------------------------------------------------------
def build_session(gs, reg, cohort_stays, med_cohort, deltas):
    """Register the deltas and return (states-in-order, backtrack_idx, dag, meta).
    DAG: S0{} -> S1{d1} -> S2{d1,d2};  backtrack to {d1} -> S3{d1,d2p};
         extend S3 branch -> S4{d1,d2p,d3}."""
    g = torch.Generator().manual_seed(SEED)
    C = cohort_stays
    # deterministic sub-cohorts
    perm = C[torch.randperm(C.numel(), generator=g)]
    sub2 = perm[: max(1, C.numel() // 2)]              # d2 sub-cohort
    sub3 = perm[C.numel() // 2:]                        # d3 (extend) sub-cohort
    # a common medication (high in-degree) as the added "second drug"
    ei = gs.edge_index[RECV]
    deg = torch.zeros(gs.num_nodes[MED_NT]).index_add_(
        0, ei[1], torch.ones(ei.size(1)))
    common_med = int(deg.argmax())

    d1 = reg.add(iv_feat(gs, MED_NT, med_cohort, deltas["d1"]))       # attribute
    d2 = reg.add(iv_add_recv(sub2, common_med))                       # structure
    d2p = reg.add(iv_feat(gs, MED_NT, med_cohort, deltas["d2p"]))     # attribute
    d3 = reg.add(iv_add_recv(sub3, common_med))                       # structure

    S = [frozenset(), frozenset({d1}), frozenset({d1, d2}),
         frozenset({d1, d2p}), frozenset({d1, d2p, d3})]
    dag = {"S0": [], "S1": ["d1"], "S2": ["d1", "d2"],
           "S3": ["d1", "d2p (backtrack to {d1})"],
           "S4": ["d1", "d2p", "d3 (extend S3 branch)"]}
    meta = dict(common_med_idx=common_med, sub2_size=int(sub2.numel()),
                sub3_size=int(sub3.numel()), med_cohort_size=int(med_cohort.numel()),
                delta_ids=dict(d1=d1, d2=d2, d2p=d2p, d3=d3))
    return S, 3, dag, meta      # backtrack_idx = 3 (S3)


def choose_tau(model, gs, cohort, lo=0.2, hi=0.5):
    """Pick tau so the S0 baseline Q is non-degenerate (~0.2-0.5)."""
    h0 = whatif_forward(model, gs, Intervention())
    r = risk_vec(model, h0)[cohort]
    # target Q ~ 0.3: tau = the (1-0.3) quantile of cohort risk, clamped
    tau = float(torch.quantile(r, 0.70))
    tau = min(max(tau, 0.05), 0.95)
    q = float((r >= tau).float().mean())
    return round(tau, 4), q


def run_session(deltas, arch=ARCH, session_len=5):
    set_determinism(SEED)
    data, schema = load_graph(DATASET, refresh_card=False)
    gs = GraphState(data)
    model = load_model(os.path.join(CHECKPOINTS, DATASET,
                                    f"{TASK}_{arch}_L{L}.pt"))
    store = BoltStore(NEO4J["uri"], NEO4J["user"], NEO4J["password"])
    with store._session() as s:
        ver = s.run("CALL dbms.components() YIELD name, versions, edition "
                    "RETURN name, versions[0] AS v, edition").single()
        neo4j_info = {"name": ver["name"], "version": ver["v"], "edition": ver["edition"]}

    # S0: OURS reads the base graph from Neo4j ONCE; verify identical start
    raw0, load_metrics = db_download_once(store, schema)
    xmax, edge_ok = verify_db_matches_local(raw0, schema, gs)
    assert edge_ok and xmax < 1e-4, (
        f"Neo4j base graph != local processed graph (xmax={xmax}, edge_ok={edge_ok})")

    # resident caches (ours): forward cache built once from the resident graph.
    # Ripple reads a BaselineCache (base-graph attention) — also built once.
    base_cache = build_cache(model, gs.base_view())
    bcache = BaselineCache(model, gs)
    caches = {frozenset(): base_cache}

    # cohorts
    p60 = over60_patient_idx()
    cohort = stays_of_patients(gs, p60)
    med_cohort = meds_of_stays(gs, cohort)
    tau, baseQ = choose_tau(model, gs, cohort)

    reg = Registry()
    states, backtrack_idx, dag, smeta = build_session(gs, reg, cohort, med_cohort, deltas)
    states = states[:session_len]

    E = sum(t.size(1) for t in gs.edge_index.values())
    scratch_edges_per_state = 2 * E * L

    rows = []
    cum_scratch = cum_ours = 0
    cum_db_bytes_base = cum_db_ms_base = 0.0
    cum_db_bytes_ours = load_metrics["refetch_bytes"]        # one-time at S0
    cum_db_ms_ours = load_metrics["roundtrip_ms"]
    cum_lat_scr = cum_lat_rip = cum_lat_neu = cum_lat_ours = 0.0
    for si, state in enumerate(states):
        iv = compose(reg, state)
        # ours reuse pieces (ancestor captured BEFORE materializing this state)
        anc = _nearest_ancestor(caches, state)
        iv_res = compose(reg, state - anc)
        view = gs.apply(iv)
        fd, sd = feat_dirty_rows(iv_res), struct_dirty_rows(gs, iv_res)

        # ours' best reuse path per arch: low-rank delta for RGCN feature-only
        # residual, exact closure recompute otherwise (both exact).
        use_delta = (arch == "rgcn") and (not is_topological(iv_res))
        def _ours():
            if use_delta:
                return delta_forward(model, caches[anc], iv_res.feat_deltas)
            return closure_forward(model, caches[anc], view,
                                   feat_dirty=fd, struct_dirty=sd)

        # (a) from-scratch ground truth (in-memory tensor == the re-fetched G^Δ)
        h_a = whatif_forward(model, gs, iv)
        # (b) ours: resident-cache incremental composition
        h_b, new_cache, ostats = _ours()
        ours_edges = ostats["edges_touched"]
        # (c) Ripple: frozen base-graph attention/degree, composed from base cache
        h_c = Ripple(model, gs, base_cache).apply(bcache, iv)
        # (d) NeutronRT: EXACT streaming baseline (new-degree delta for RGCN,
        # constrained closure for hGAT), recomputed from base each state.
        h_d = NeutronRT(model, gs, base_cache).apply(bcache, iv)

        # compute wall-times (ms); DB round-trip timed separately below
        t_scr = med_time(lambda: whatif_forward(model, gs, iv))
        t_rip = med_time(lambda: Ripple(model, gs, base_cache).apply(bcache, iv))
        t_neu = med_time(lambda: NeutronRT(model, gs, base_cache).apply(bcache, iv))
        t_our = med_time(_ours)
        caches[state] = new_cache            # materialize AFTER timing

        Qa, _ = query_Q(model, h_a, cohort, tau)
        Qb, _ = query_Q(model, h_b, cohort, tau)
        Qc, _ = query_Q(model, h_c, cohort, tau)
        Qd, _ = query_Q(model, h_d, cohort, tau)
        b_qmax = abs(Qb - Qa)
        c_qmax = abs(Qc - Qa)
        d_qmax = abs(Qd - Qa)
        b_emax, b_emean = emb_err(h_b, h_a, LABEL_NT)
        c_emax, c_emean = emb_err(h_c, h_a, LABEL_NT)
        d_emax, d_emean = emb_err(h_d, h_a, LABEL_NT)

        # exactness gate: ours must match from-scratch
        rep = check(h_b, h_a)
        if not rep.passed:
            store.close()
            raise SystemExit(f"COMPOSITION BUG: ours != from-scratch at S{si} "
                             f"(max_abs={rep.max_abs_err:.2e}). Not plotting.")

        # work proxies
        cum_scratch += scratch_edges_per_state
        cum_ours += ours_edges

        # DB cost: baselines apply+refetch every state; ours 0 after S0
        if si == 0:
            dbm = dict(db_writes=0, refetch_nodes=load_metrics["refetch_nodes"],
                       refetch_edges=load_metrics["refetch_edges"],
                       refetch_bytes=load_metrics["refetch_bytes"],
                       roundtrip_ms=load_metrics["roundtrip_ms"])
        else:
            dbm = db_roundtrip(store, gs, iv, schema)
        cum_db_bytes_base += dbm["refetch_bytes"]
        cum_db_ms_base += dbm["roundtrip_ms"]

        # end-to-end LATENCY (ms): baselines pay DB apply+download+rollback every
        # state + their compute; ours pays only its resident compute per state
        # (+ the one-time S0 graph load, charged once).
        lat_scr = dbm["roundtrip_ms"] + t_scr
        lat_rip = dbm["roundtrip_ms"] + t_rip
        lat_neu = dbm["roundtrip_ms"] + t_neu
        lat_our = t_our + (load_metrics["roundtrip_ms"] if si == 0 else 0.0)
        cum_lat_scr += lat_scr
        cum_lat_rip += lat_rip
        cum_lat_neu += lat_neu
        cum_lat_ours += lat_our

        rows.append(dict(
            step=si, state=sorted(state), depth=len(state),
            is_backtrack=(si == backtrack_idx),
            Q_scratch=round(Qa, 6), Q_ours=round(Qb, 6), Q_ripple=round(Qc, 6),
            Q_neutronrt=round(Qd, 6),
            ours_query_abs_err=b_qmax, ripple_query_abs_err=c_qmax,
            neutronrt_query_abs_err=d_qmax,
            ours_emb_max=b_emax, ours_emb_mean=b_emean,
            ripple_emb_max=c_emax, ripple_emb_mean=c_emean,
            neutronrt_emb_max=d_emax, neutronrt_emb_mean=d_emean,
            edges_scratch_step=scratch_edges_per_state, edges_ours_step=ours_edges,
            cum_edges_scratch=cum_scratch, cum_edges_ours=cum_ours,
            gnn_ms_scratch=round(t_scr, 3), gnn_ms_ripple=round(t_rip, 3),
            gnn_ms_neutronrt=round(t_neu, 3), gnn_ms_ours=round(t_our, 3),
            lat_ms_scratch=round(lat_scr, 2), lat_ms_ripple=round(lat_rip, 2),
            lat_ms_neutronrt=round(lat_neu, 2), lat_ms_ours=round(lat_our, 2),
            cum_lat_ms_scratch=round(cum_lat_scr, 2),
            cum_lat_ms_ripple=round(cum_lat_rip, 2),
            cum_lat_ms_neutronrt=round(cum_lat_neu, 2),
            cum_lat_ms_ours=round(cum_lat_ours, 2),
            **{f"db_{k}": v for k, v in dbm.items()},
            cum_db_bytes_base=cum_db_bytes_base, cum_db_ms_base=round(cum_db_ms_base, 2),
            cum_db_bytes_ours=cum_db_bytes_ours, cum_db_ms_ours=cum_db_ms_ours,
            ancestor_depth=len(anc), residual_size=len(state - anc)))
        print(f"S{si} |D|={len(state)} {'[BACKTRACK]' if si==backtrack_idx else '':11} "
              f"Q: scr={Qa:.4f} ours={Qb:.4f} rip={Qc:.4f} neu={Qd:.4f} | "
              f"rip_err={c_qmax*100:5.2f}pp neu_err={d_qmax*1e3:.1e} ours_err={b_qmax*1e3:.1e} | "
              f"lat(ms) scr={lat_scr:.0f} neu={lat_neu:.0f} ours={lat_our:.0f}", flush=True)

    store.close()
    return rows, dict(arch=arch, tau=tau, baseQ=baseQ, cohort_size=int(cohort.numel()),
                      over60_patients=int(p60.numel()), dag=dag, backtrack_idx=backtrack_idx,
                      neo4j=neo4j_info, db_verify=dict(feat_max_abs=xmax, edges_match=edge_ok),
                      one_time_load=load_metrics, session_meta=smeta,
                      scratch_edges_per_state=scratch_edges_per_state)


# ---------------------------------------------------------------------------
# drift gate + escalation
# ---------------------------------------------------------------------------
def drift_ok(rows):
    e1 = rows[1]["ripple_query_abs_err"]           # S1
    ef = rows[-1]["ripple_query_abs_err"]          # final
    return (ef >= 0.005) and (e1 == 0 or ef >= 5 * e1), e1, ef


def _threshold_flip(rows):
    for r in rows:
        if (r["Q_scratch"] <= Q_THRESHOLD) != (r["Q_ripple"] <= Q_THRESHOLD):
            return dict(step=r["step"], Q_scratch=r["Q_scratch"], Q_ripple=r["Q_ripple"])
    return None


def _arch_section(rows, info):
    ok, e1, ef = drift_ok(rows)
    return dict(
        arch=info["arch"], tau=info["tau"], base_Q=info["baseQ"],
        drift_gate=dict(passed=bool(ok), ripple_err_S1=e1, ripple_err_final=ef,
                        ratio=(ef / e1 if e1 else None), min_abs=0.005),
        threshold_flip=_threshold_flip(rows),
        exactness=dict(ours_vs_scratch_max_query_err=max(r["ours_query_abs_err"] for r in rows),
                       neutronrt_vs_scratch_max_query_err=max(r["neutronrt_query_abs_err"] for r in rows),
                       passed=bool(max(r["ours_query_abs_err"] for r in rows) < 1e-4)),
        totals=dict(
            cum_edges_scratch=rows[-1]["cum_edges_scratch"], cum_edges_ours=rows[-1]["cum_edges_ours"],
            edge_msg_saving=round(rows[-1]["cum_edges_scratch"] / max(rows[-1]["cum_edges_ours"], 1), 1),
            db_bytes_base=rows[-1]["cum_db_bytes_base"], db_bytes_ours=rows[-1]["cum_db_bytes_ours"],
            lat_ms_scratch=rows[-1]["cum_lat_ms_scratch"], lat_ms_ripple=rows[-1]["cum_lat_ms_ripple"],
            lat_ms_neutronrt=rows[-1]["cum_lat_ms_neutronrt"], lat_ms_ours=rows[-1]["cum_lat_ms_ours"],
            latency_speedup=round(rows[-1]["cum_lat_ms_scratch"] / max(rows[-1]["cum_lat_ms_ours"], 1e-9), 1)),
        states=rows)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    # Fix the Δ magnitudes on the hGAT drift gate (escalate if needed), then run
    # the SAME session on RGCN for the second column.
    escalation = "none"
    deltas = dict(d1=1.0, d2p=2.5)
    rows_h, info_h = run_session(deltas, arch="hgat")
    ok, _, _ = drift_ok(rows_h)
    for esc, dv in (("delta x2 (2.0 / 5.0)", dict(d1=2.0, d2p=5.0)),
                    ("delta x4 (4.0 / 10.0)", dict(d1=4.0, d2p=10.0))):
        if ok:
            break
        escalation, deltas = esc, dv
        rows_h, info_h = run_session(deltas, arch="hgat")
        ok, _, _ = drift_ok(rows_h)
    rows_r, info_r = run_session(deltas, arch="rgcn")

    archs = {"hgat": _arch_section(rows_h, info_h), "rgcn": _arch_section(rows_r, info_r)}
    out = dict(
        dataset=DATASET, task=TASK, layers=L, seed=SEED, deltas=deltas, escalation=escalation,
        Q_threshold=Q_THRESHOLD, backtrack_step=info_h["backtrack_idx"], dag=info_h["dag"],
        cohort_stays=info_h["cohort_size"], over60_patients=info_h["over60_patients"],
        cohort_choice=("over-60 patients (real ages, demo CSV) -> their stays "
                       "(query cohort); Δ shifts the MEDICATION node feature for "
                       "medications administered to those stays. eICU stores dosage "
                       "on the received EDGE attr (unused by the encoder); the node "
                       "feature is the node-level dosage/intensity analogue."),
        neo4j=info_h["neo4j"], db_verify=info_h["db_verify"], one_time_load=info_h["one_time_load"],
        rollback_method="transaction rollback (tx.rollback) — disposable overlay",
        session_meta=info_h["session_meta"], archs=archs)
    json.dump(out, open(os.path.join(RESULTS_DIR, "results.json"), "w"), indent=2)

    # stdout summary
    print("\n==== interactive eICU session summary (RGCN + hGAT) ====")
    print(f"Neo4j: {info_h['neo4j']['name']} {info_h['neo4j']['version']} ({info_h['neo4j']['edition']})")
    print(f"base graph verified identical to local: feat_max_abs={info_h['db_verify']['feat_max_abs']:.1e}")
    print(f"cohort = {info_h['cohort_size']} stays of {info_h['over60_patients']} over-60 patients; "
          f"deltas={deltas}; escalation={escalation}")
    for a in ("rgcn", "hgat"):
        s = archs[a]; g = s["drift_gate"]; t = s["totals"]
        print(f"  [{a}] tau={s['tau']} baseQ={s['base_Q']:.3f} | "
              f"Ripple err S1={g['ripple_err_S1']*100:.2f}pp -> final={g['ripple_err_final']*100:.2f}pp "
              f"(gate={g['passed']}) | ours exact={s['exactness']['passed']} "
              f"(max {s['exactness']['ours_vs_scratch_max_query_err']:.1e}) | "
              f"flip={'yes' if s['threshold_flip'] else 'no'} | "
              f"latency ours {t['lat_ms_ours']:.0f}ms vs scratch {t['lat_ms_scratch']:.0f}ms ({t['latency_speedup']}x)")

    from src.eval.figures import fig_interactive_eicu
    fig_interactive_eicu(out)
    print(f"\nwrote {RESULTS_DIR}/results.json + figures/interactive_eicu.(pdf|png)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
