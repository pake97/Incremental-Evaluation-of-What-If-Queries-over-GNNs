"""Interactive what-if session on SNB (LDBC social network) / hGAT + RGCN,
backed by a real Neo4j store. Same design as exp_interactive_eicu, different
dataset + query (a social-network cohort query instead of a clinical one).

Graph: LDBC SNB. Label node type = person; the frozen node-classifier scores
each person for the target class (home-country membership). Query:

  Q = (#cohort persons with class score >= tau) / |cohort|   ("predicted-positive
      fraction over a cohort").

Cohort = the "young-user" cohort (bottom tertile of the age feature). Composing
session (branch + backtrack) mixes ATTRIBUTE Δ (shift the person age feature for
the cohort — a demographic what-if) and STRUCTURE Δ (add `knows` edges from a
sub-cohort to the most-connected person — a "befriend the hub" what-if).

Reuses the shared machinery from exp_interactive_eicu (timing, DB round-trip,
resident-cache reuse, Ripple, exactness, figure). Baselines re-fetch the full
graph from Neo4j every state; ours reads it once at S0 and composes resident.

  venv/bin/python -m src.eval.exp_interactive_snb
"""
from __future__ import annotations

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
from src.paths import CHECKPOINTS, FIGURES, RESULTS
# generic, dataset-independent helpers
from src.eval.exp_interactive_eicu import (med_time, emb_err, drift_ok,
                                          iv_feat, _bytes_of,
                                          _nearest_ancestor,
                                          verify_db_matches_local)

SEED = 42
DATASET, TASK, L = "snb", "nodelabel", 2
NEO4J = dict(uri="bolt://localhost:7687", user="neo4j", password="rcfpassword")
PFX = "snb_"
LABEL_NT = "person"
CORE = ("person", "knows", "person")
AGE_COORD = 0                       # person feat: [age_z, browser 1-hot(5), nlang_z]
Q_THRESHOLD = 0.15
DB_REPS = 1                         # SNB full download ~25s; keep timing reps small

RESULTS_DIR = os.path.join(RESULTS, "interactive_snb")


# ---- query -----------------------------------------------------------------
def risk_vec(model, h):
    return torch.sigmoid(model.classify_nodes(h, LABEL_NT))


def query_Q(model, h, cohort, tau):
    r = risk_vec(model, h)[cohort]
    return float((r >= tau).float().mean()), r


def choose_tau(model, gs, cohort):
    h0 = whatif_forward(model, gs, Intervention())
    r = risk_vec(model, h0)[cohort]
    tau = min(max(float(torch.quantile(r, 0.70)), 0.05), 0.95)
    return round(tau, 4), float((r >= tau).float().mean())


# ---- interventions ---------------------------------------------------------
def iv_add_knows(person_ids, hub):
    src = person_ids.to(torch.long)
    dst = torch.full((src.numel(),), int(hub), dtype=torch.long)
    return Intervention(add_edges={CORE: torch.stack([src, dst])})


# ---- DB round-trip (naive tool: apply Δ + full re-fetch + rollback) --------
def db_roundtrip(store, gs, iv, schema, reps=DB_REPS):
    adds, dels, cohort = _edit_pairs(gs, iv)
    n_writes = (sum(len(p) for p in adds.values()) + sum(len(p) for p in dels.values())
                + sum(len(v) for v in cohort.values()))
    walls, raw = [], None
    for _ in range(reps):
        with store._session() as s:
            tx = s.begin_transaction()
            t0 = time.perf_counter()
            _apply_delta(tx, adds, dels, cohort, PFX)
            raw = _download_raw(tx, schema, PFX)
            tx.rollback()
            walls.append(time.perf_counter() - t0)
    walls.sort()
    n_nodes = sum(len(raw[0][nt]) for nt in schema.node_types)
    n_edges = sum(len(raw[1][tuple(et)]) for et in schema.edge_types)
    return dict(db_writes=n_writes, refetch_nodes=n_nodes, refetch_edges=n_edges,
                refetch_bytes=_bytes_of(raw, schema),
                roundtrip_ms=round(walls[len(walls) // 2] * 1e3, 2))


def db_download_once(store, schema, reps=DB_REPS):
    walls, raw = [], None
    for _ in range(max(reps, 1)):
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


# ---- session ---------------------------------------------------------------
def build_session(gs, reg, cohort, deltas):
    g = torch.Generator().manual_seed(SEED)
    perm = cohort[torch.randperm(cohort.numel(), generator=g)]
    sub2 = perm[: max(1, cohort.numel() // 2)]
    sub3 = perm[cohort.numel() // 2:]
    ei = gs.edge_index[CORE]
    deg = torch.zeros(gs.num_nodes[LABEL_NT]).index_add_(0, ei[1], torch.ones(ei.size(1)))
    hub = int(deg.argmax())                       # most-connected person

    d1 = reg.add(iv_feat(gs, LABEL_NT, cohort, deltas["d1"], coord=AGE_COORD))   # attribute
    d2 = reg.add(iv_add_knows(sub2, hub))                                        # structure
    d2p = reg.add(iv_feat(gs, LABEL_NT, cohort, deltas["d2p"], coord=AGE_COORD)) # attribute
    d3 = reg.add(iv_add_knows(sub3, hub))                                        # structure

    S = [frozenset(), frozenset({d1}), frozenset({d1, d2}),
         frozenset({d1, d2p}), frozenset({d1, d2p, d3})]
    dag = {"S0": [], "S1": ["d1"], "S2": ["d1", "d2"],
           "S3": ["d1", "d2p (backtrack to {d1})"],
           "S4": ["d1", "d2p", "d3 (extend S3 branch)"]}
    meta = dict(hub_person_idx=hub, sub2_size=int(sub2.numel()),
                sub3_size=int(sub3.numel()), cohort_size=int(cohort.numel()),
                delta_ids=dict(d1=d1, d2=d2, d2p=d2p, d3=d3))
    return S, 3, dag, meta


def run_session(deltas, arch, session_len=5):
    set_determinism(SEED)
    data, schema = load_graph(DATASET, refresh_card=False)
    gs = GraphState(data)
    model = load_model(os.path.join(CHECKPOINTS, DATASET, f"{TASK}_{arch}_L{L}.pt"))
    store = BoltStore(NEO4J["uri"], NEO4J["user"], NEO4J["password"])
    with store._session() as s:
        ver = s.run("CALL dbms.components() YIELD name, versions, edition "
                    "RETURN name, versions[0] AS v, edition").single()
        neo4j_info = {"name": ver["name"], "version": ver["v"], "edition": ver["edition"]}

    raw0, load_metrics = db_download_once(store, schema)
    xmax, edge_ok = verify_db_matches_local(raw0, schema, gs)
    assert edge_ok and xmax < 1e-4, f"Neo4j != local (xmax={xmax}, edges={edge_ok})"

    base_cache = build_cache(model, gs.base_view())
    bcache = BaselineCache(model, gs)
    caches = {frozenset(): base_cache}

    # cohort = young-user tertile (bottom third of the age feature)
    age = gs.x[LABEL_NT][:, AGE_COORD]
    cutoff = torch.quantile(age, 1.0 / 3.0)
    cohort = (age <= cutoff).nonzero(as_tuple=False).squeeze(1)
    tau, baseQ = choose_tau(model, gs, cohort)

    reg = Registry()
    states, backtrack_idx, dag, smeta = build_session(gs, reg, cohort, deltas)
    states = states[:session_len]

    E = sum(t.size(1) for t in gs.edge_index.values())
    scratch_edges = 2 * E * L

    rows = []
    cum_scr = cum_our = 0
    cum_lat_scr = cum_lat_rip = cum_lat_neu = cum_lat_ours = 0.0
    cum_db_bytes_base = cum_db_ms_base = 0.0
    cum_db_bytes_ours = load_metrics["refetch_bytes"]
    cum_db_ms_ours = load_metrics["roundtrip_ms"]
    for si, state in enumerate(states):
        iv = compose(reg, state)
        anc = _nearest_ancestor(caches, state)
        iv_res = compose(reg, state - anc)
        view = gs.apply(iv)
        fd, sd = feat_dirty_rows(iv_res), struct_dirty_rows(gs, iv_res)
        use_delta = (arch == "rgcn") and (not is_topological(iv_res))

        def _ours():
            if use_delta:
                return delta_forward(model, caches[anc], iv_res.feat_deltas)
            return closure_forward(model, caches[anc], view, feat_dirty=fd, struct_dirty=sd)

        h_a = whatif_forward(model, gs, iv)
        h_b, new_cache, ostats = _ours()
        h_c = Ripple(model, gs, base_cache).apply(bcache, iv)
        h_d = NeutronRT(model, gs, base_cache).apply(bcache, iv)   # exact streaming baseline

        t_scr = med_time(lambda: whatif_forward(model, gs, iv), reps=2)
        t_rip = med_time(lambda: Ripple(model, gs, base_cache).apply(bcache, iv), reps=2)
        t_neu = med_time(lambda: NeutronRT(model, gs, base_cache).apply(bcache, iv), reps=2)
        t_our = med_time(_ours, reps=2)
        caches[state] = new_cache

        Qa, _ = query_Q(model, h_a, cohort, tau)
        Qb, _ = query_Q(model, h_b, cohort, tau)
        Qc, _ = query_Q(model, h_c, cohort, tau)
        Qd, _ = query_Q(model, h_d, cohort, tau)
        b_qmax, c_qmax, d_qmax = abs(Qb - Qa), abs(Qc - Qa), abs(Qd - Qa)
        b_emax, b_emean = emb_err(h_b, h_a, LABEL_NT)
        c_emax, c_emean = emb_err(h_c, h_a, LABEL_NT)
        d_emax, d_emean = emb_err(h_d, h_a, LABEL_NT)

        if not check(h_b, h_a).passed:
            store.close()
            raise SystemExit(f"COMPOSITION BUG: ours != from-scratch at S{si}")

        cum_scr += scratch_edges
        cum_our += ostats["edges_touched"]

        if si == 0:
            dbm = dict(db_writes=0, refetch_nodes=load_metrics["refetch_nodes"],
                       refetch_edges=load_metrics["refetch_edges"],
                       refetch_bytes=load_metrics["refetch_bytes"],
                       roundtrip_ms=load_metrics["roundtrip_ms"])
        else:
            dbm = db_roundtrip(store, gs, iv, schema)
        cum_db_bytes_base += dbm["refetch_bytes"]
        cum_db_ms_base += dbm["roundtrip_ms"]

        lat_scr = dbm["roundtrip_ms"] + t_scr
        lat_rip = dbm["roundtrip_ms"] + t_rip
        lat_neu = dbm["roundtrip_ms"] + t_neu
        lat_our = t_our + (load_metrics["roundtrip_ms"] if si == 0 else 0.0)
        cum_lat_scr += lat_scr; cum_lat_rip += lat_rip
        cum_lat_neu += lat_neu; cum_lat_ours += lat_our

        rows.append(dict(
            step=si, state=sorted(state), depth=len(state), is_backtrack=(si == backtrack_idx),
            Q_scratch=round(Qa, 6), Q_ours=round(Qb, 6), Q_ripple=round(Qc, 6), Q_neutronrt=round(Qd, 6),
            ours_query_abs_err=b_qmax, ripple_query_abs_err=c_qmax, neutronrt_query_abs_err=d_qmax,
            ours_emb_max=b_emax, ours_emb_mean=b_emean,
            ripple_emb_max=c_emax, ripple_emb_mean=c_emean,
            neutronrt_emb_max=d_emax, neutronrt_emb_mean=d_emean,
            edges_scratch_step=scratch_edges, edges_ours_step=ostats["edges_touched"],
            cum_edges_scratch=cum_scr, cum_edges_ours=cum_our,
            gnn_ms_scratch=round(t_scr, 3), gnn_ms_ripple=round(t_rip, 3),
            gnn_ms_neutronrt=round(t_neu, 3), gnn_ms_ours=round(t_our, 3),
            lat_ms_scratch=round(lat_scr, 2), lat_ms_ripple=round(lat_rip, 2),
            lat_ms_neutronrt=round(lat_neu, 2), lat_ms_ours=round(lat_our, 2),
            cum_lat_ms_scratch=round(cum_lat_scr, 2), cum_lat_ms_ripple=round(cum_lat_rip, 2),
            cum_lat_ms_neutronrt=round(cum_lat_neu, 2), cum_lat_ms_ours=round(cum_lat_ours, 2),
            **{f"db_{k}": v for k, v in dbm.items()},
            cum_db_bytes_base=cum_db_bytes_base, cum_db_ms_base=round(cum_db_ms_base, 2),
            cum_db_bytes_ours=cum_db_bytes_ours, cum_db_ms_ours=cum_db_ms_ours,
            ancestor_depth=len(anc), residual_size=len(state - anc)))
        print(f"[{arch}] S{si} |D|={len(state)} {'[BACKTRACK]' if si==backtrack_idx else '':11} "
              f"Q: scr={Qa:.4f} ours={Qb:.4f} rip={Qc:.4f} neu={Qd:.4f} | rip_err={c_qmax*100:5.2f}pp "
              f"neu_err={d_qmax*1e3:.1e} ours_err={b_qmax*1e3:.1e} | "
              f"lat(ms) scr={lat_scr:.0f} neu={lat_neu:.0f} ours={lat_our:.0f}", flush=True)

    store.close()
    return rows, dict(arch=arch, tau=tau, baseQ=baseQ, cohort_size=int(cohort.numel()),
                      dag=dag, backtrack_idx=backtrack_idx, neo4j=neo4j_info,
                      db_verify=dict(feat_max_abs=xmax, edges_match=edge_ok),
                      one_time_load=load_metrics, session_meta=smeta)


def _threshold_flip(rows):
    for r in rows:
        if (r["Q_scratch"] <= Q_THRESHOLD) != (r["Q_ripple"] <= Q_THRESHOLD):
            return dict(step=r["step"], Q_scratch=r["Q_scratch"], Q_ripple=r["Q_ripple"])
    return None


def _arch_section(rows, info):
    ok, e1, ef = drift_ok(rows)
    mq = max(r["ours_query_abs_err"] for r in rows)
    return dict(
        arch=info["arch"], tau=info["tau"], base_Q=info["baseQ"],
        drift_gate=dict(passed=bool(ok), ripple_err_S1=e1, ripple_err_final=ef,
                        ratio=(ef / e1 if e1 else None), min_abs=0.005),
        threshold_flip=_threshold_flip(rows),
        exactness=dict(ours_vs_scratch_max_query_err=mq,
                       neutronrt_vs_scratch_max_query_err=max(r["neutronrt_query_abs_err"] for r in rows),
                       passed=bool(mq < 1e-4)),
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
    os.makedirs(FIGURES, exist_ok=True)

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
        cohort_persons=info_h["cohort_size"],
        cohort_choice=("young-user cohort = persons in the bottom tertile of the age "
                       "feature (query cohort). Attribute Δ shifts the person age feature "
                       "for the cohort; structure Δ adds `knows` edges from a sub-cohort to "
                       "the most-connected person (social hub). Label = home-country class; "
                       "score = sigmoid(person head)."),
        neo4j=info_h["neo4j"], db_verify=info_h["db_verify"], one_time_load=info_h["one_time_load"],
        rollback_method="transaction rollback (tx.rollback) — disposable overlay",
        session_meta=info_h["session_meta"], archs=archs,
        fig_title="Interactive what-if session — SNB (Neo4j-backed)",
        fig_name="interactive_snb",
        q_ylabel="home-country\nfraction Q")
    json.dump(out, open(os.path.join(RESULTS_DIR, "results.json"), "w"), indent=2)

    print("\n==== interactive SNB session summary (RGCN + hGAT) ====")
    print(f"Neo4j: {info_h['neo4j']['name']} {info_h['neo4j']['version']} ({info_h['neo4j']['edition']})")
    print(f"base graph verified identical to local: feat_max_abs={info_h['db_verify']['feat_max_abs']:.1e}")
    print(f"cohort = {info_h['cohort_size']} young-user persons; deltas={deltas}; escalation={escalation}")
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
    print(f"\nwrote {RESULTS_DIR}/results.json + figures/interactive_snb.(pdf|png)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
