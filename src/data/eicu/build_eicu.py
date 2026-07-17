"""Build the eICU heterogeneous graph from the PhysioNet eICU-CRD demo CSVs.

Mirrors src/data/finbench/prepare_finbench.py and src/data/faers/build_faers.py:
same processed contract (hetero .pt + 3 splits + id_maps + meta), consumed by
the same loader, models, and maintenance code.

Graph (5 node types):
  patient     uniquepid              feat: gender one-hot, age (z)
  stay        patientunitstayid      feat: unittype one-hot, unitstaytype
                                     one-hot, admissionweight/height (z)
                                     label: in-hospital mortality
  diagnosis   first 2 levels of diagnosisstring   feat: const
  treatment   first 3 levels of treatmentstring   feat: const
  medication  drughiclseqno (fallback: cleaned drugname)  feat: const

Edges (edge_time = minutes from unit admission where the event is dated):
  patient -has_stay->  stay        (static; no edge_time)
  stay -diagnosed->    diagnosis   (diagnosisoffset;  value = priority ord)
  stay -treated->      treatment   (treatmentoffset)
  stay -received->     medication  (drugstartoffset;  value = log1p(dose))
                                   [CORE relation for the link tasks]

Tasks (master seed 42, temporal protocol as in the other datasets):
  node label : stay mortality (hospitaldischargestatus == 'Expired'),
               stratified 70/10/20 masks.
  link pred  : stay -received-> medication appearing after the time cut;
               cuts at q60/q70/q80 of drugstartoffset; 1:1 negatives over
               (stay, medication) pairs never linked at any time.
  link reg   : log1p(parsed dose) on received edges (positives only).

Run:  python -m src.data.eicu.build_eicu
"""
from __future__ import annotations

import argparse, csv, gzip, json, os, re

import numpy as np
import torch
from torch_geometric.data import HeteroData

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
# (raw dir, processed dir) per release; identical schema, so one builder serves both.
RELEASES = {
    "demo": (os.path.join(ROOT, "dataset", "physionet.org", "files",
                          "eicu-crd-demo", "2.0.1"),
             os.path.join(ROOT, "dataset", "eicu", "processed")),
    "full": (os.path.join(ROOT, "dataset", "physionet.org", "files",
                          "eicu-crd", "2.0"),
             os.path.join(ROOT, "dataset", "eicu_full", "processed")),
}
RAW, OUT = RELEASES["demo"]          # rebound in main() from --release
SEED = 42
Q_MSG, Q_TRAIN, Q_VAL = 0.60, 0.70, 0.80
PRIORITY = {"Primary": 2.0, "Major": 1.0, "Other": 0.0}
RE_DOSE = re.compile(r"[-+]?\d*\.?\d+")


def rows(name):
    with gzip.open(os.path.join(RAW, name), "rt", newline="") as fh:
        yield from csv.DictReader(fh)


def zscore(v):
    v = np.asarray(v, dtype=np.float32)
    med = np.nanmedian(v)
    v = np.where(np.isnan(v), med, v)
    sd = v.std()
    return (v - v.mean()) / (sd if sd > 1e-8 else 1.0)


def onehot(values, vocab):
    m = np.zeros((len(values), len(vocab)), dtype=np.float32)
    idx = {c: i for i, c in enumerate(vocab)}
    for r, val in enumerate(values):
        if val in idx:
            m[r, idx[val]] = 1.0
    return m


def parse_age(s):
    if not s:
        return np.nan
    if ">" in s:
        return 90.0
    try:
        return float(s)
    except ValueError:
        return np.nan


def parse_dose(s):
    m = RE_DOSE.search(s or "")
    return float(m.group()) if m else np.nan


def main():
    global RAW, OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--release", choices=sorted(RELEASES), default="demo",
                    help="'demo' (eicu-crd-demo 2.0.1) or 'full' (credentialed "
                         "eicu-crd 2.0); identical schema, different scale.")
    a = ap.parse_args()
    RAW, OUT = RELEASES[a.release]
    if not os.path.isdir(RAW):
        raise SystemExit(
            f"raw eICU '{a.release}' not found at {RAW}\n"
            "The full release is credentialed — download it into that path with "
            "your PhysioNet account (see src/data/eicu/README.md).")
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(SEED)

    # ---- patients & stays --------------------------------------------------
    pats = list(rows("patient.csv.gz"))
    stay_ids = [r["patientunitstayid"] for r in pats]
    stay_id = {s: i for i, s in enumerate(stay_ids)}
    pids = sorted({r["uniquepid"] for r in pats})
    pat_id = {p: i for i, p in enumerate(pids)}

    genders = sorted({r["gender"] for r in pats if r["gender"]})
    # patient features come from their first stay's demographics
    first = {}
    for r in pats:
        first.setdefault(r["uniquepid"], r)
    pat_x = np.concatenate([
        onehot([first[p]["gender"] for p in pids], genders),
        zscore([parse_age(first[p]["age"]) for p in pids])[:, None],
    ], axis=1)

    unittypes = sorted({r["unittype"] for r in pats if r["unittype"]})
    staytypes = sorted({r["unitstaytype"] for r in pats if r["unitstaytype"]})

    def fnum(r, k):
        try:
            return float(r[k]) if r[k] else np.nan
        except ValueError:
            return np.nan

    stay_x = np.concatenate([
        onehot([r["unittype"] for r in pats], unittypes),
        onehot([r["unitstaytype"] for r in pats], staytypes),
        zscore([fnum(r, "admissionweight") for r in pats])[:, None],
        zscore([fnum(r, "admissionheight") for r in pats])[:, None],
    ], axis=1)
    stay_y = np.array([1 if r["hospitaldischargestatus"] == "Expired" else 0
                       for r in pats], dtype=np.int64)

    has_stay = np.array([[pat_id[r["uniquepid"]], stay_id[r["patientunitstayid"]]]
                         for r in pats], dtype=np.int64).T

    # ---- event tables -> concept nodes + dated edges -----------------------
    def concept_edges(table, sid_col, off_col, concept_fn, value_fn):
        cid, edges = {}, []
        for r in rows(table):
            sid = r.get(sid_col)
            if sid not in stay_id:
                continue
            c = concept_fn(r)
            if not c:
                continue
            try:
                t = float(r[off_col])
            except (ValueError, TypeError):
                continue
            j = cid.setdefault(c, len(cid))
            edges.append((stay_id[sid], j, t, value_fn(r)))
        return cid, edges

    dx_id, dx_edges = concept_edges(
        "diagnosis.csv.gz", "patientunitstayid", "diagnosisoffset",
        lambda r: "|".join(r["diagnosisstring"].split("|")[:2]),
        lambda r: PRIORITY.get(r["diagnosispriority"], 0.0))
    tr_id, tr_edges = concept_edges(
        "treatment.csv.gz", "patientunitstayid", "treatmentoffset",
        lambda r: "|".join(r["treatmentstring"].split("|")[:3]),
        lambda r: 0.0)
    med_id, med_edges = concept_edges(
        "medication.csv.gz", "patientunitstayid", "drugstartoffset",
        lambda r: (None if r["drugordercancelled"] == "Yes"
                   else (f"hicl:{r['drughiclseqno']}" if r["drughiclseqno"]
                         else re.sub(r"[\d.]+|\s+", " ", r["drugname"]).strip().upper() or None)),
        lambda r: parse_dose(r["dosage"]))

    def pack(edges, with_value=True):
        ei = torch.tensor([[e[0] for e in edges], [e[1] for e in edges]], dtype=torch.long)
        t = torch.tensor([e[2] for e in edges], dtype=torch.float32)
        v = np.array([e[3] for e in edges], dtype=np.float32)
        return ei, t, v

    data = HeteroData()
    data["patient"].x = torch.from_numpy(pat_x)
    data["stay"].x = torch.from_numpy(stay_x)
    for nt, cid in (("diagnosis", dx_id), ("treatment", tr_id), ("medication", med_id)):
        data[nt].x = torch.ones(len(cid), 1)

    data["patient", "has_stay", "stay"].edge_index = torch.from_numpy(has_stay)

    for (s, r_, d), edges, vfun in (
            (("stay", "diagnosed", "diagnosis"), dx_edges, lambda v: v),
            (("stay", "treated", "treatment"), tr_edges, lambda v: v),
            (("stay", "received", "medication"), med_edges,
             lambda v: np.log1p(np.where(np.isnan(v) | (v < 0), 0.0, v)))):
        ei, t, v = pack(edges)
        val = vfun(v).astype(np.float32)
        data[s, r_, d].edge_index = ei
        data[s, r_, d].edge_time = t
        data[s, r_, d].edge_attr = torch.from_numpy(
            np.stack([val, zscore(t.numpy())], axis=1))

    # ---- temporal cuts on the core relation --------------------------------
    RECV = ("stay", "received", "medication")
    times = data[RECV].edge_time.numpy()
    q60, q70, q80 = np.quantile(times, [Q_MSG, Q_TRAIN, Q_VAL])

    # ---- link prediction: 1:1 negatives over never-linked pairs ------------
    ei = data[RECV].edge_index.numpy()
    linked = set(zip(ei[0].tolist(), ei[1].tolist()))
    n_stay, n_med = stay_x.shape[0], len(med_id)

    def negatives(k):
        out = []
        while len(out) < k:
            s = rng.integers(0, n_stay, size=2 * k)
            m = rng.integers(0, n_med, size=2 * k)
            for a, b in zip(s.tolist(), m.tolist()):
                if (a, b) not in linked:
                    out.append((a, b))
                    if len(out) == k:
                        break
        return np.array(out, dtype=np.int64).T

    def window(lo, hi):
        mask = (times >= lo) & (times < hi)
        return ei[:, mask]

    linkpred, linkreg = {}, {}
    for split_name, (lo, hi, cut) in {
            "train": (q60, q70, q60), "val": (q70, q80, q70),
            "test": (q80, np.inf, q80)}.items():
        pos = window(lo, hi)
        # dedupe repeated (stay, med) pairs inside the window
        pos = np.unique(pos.T, axis=0).T
        neg = negatives(pos.shape[1])
        eli = np.concatenate([pos, neg], axis=1)
        lbl = np.concatenate([np.ones(pos.shape[1]), np.zeros(neg.shape[1])])
        perm = rng.permutation(eli.shape[1])
        linkpred[split_name] = {
            "edge_label_index": torch.from_numpy(eli[:, perm]),
            "edge_label": torch.from_numpy(lbl[perm]).float(),
            "mp_time_cut": float(cut),
        }
        mask = (times >= lo) & (times < hi)
        linkreg[split_name] = {
            "edge_index": torch.from_numpy(ei[:, mask]),
            "y_log_dose": data[RECV].edge_attr[torch.from_numpy(mask), 0].clone(),
            "mp_time_cut": float(cut),
        }

    # ---- node labelling: stratified 70/10/20 over stays ---------------------
    idx = rng.permutation(n_stay)
    pos_idx = idx[stay_y[idx] == 1]
    neg_idx = idx[stay_y[idx] == 0]
    masks = {k: np.zeros(n_stay, dtype=bool) for k in ("train", "val", "test")}
    for part in (pos_idx, neg_idx):
        a, b = int(0.7 * len(part)), int(0.8 * len(part))
        masks["train"][part[:a]] = True
        masks["val"][part[a:b]] = True
        masks["test"][part[b:]] = True
    nodelabel = {"y": torch.from_numpy(stay_y),
                 "train_mask": torch.from_numpy(masks["train"]),
                 "val_mask": torch.from_numpy(masks["val"]),
                 "test_mask": torch.from_numpy(masks["test"])}

    # ---- write --------------------------------------------------------------
    torch.save(data, os.path.join(OUT, "eicu_hetero.pt"))
    torch.save(linkpred, os.path.join(OUT, "split_linkpred.pt"))
    torch.save(linkreg, os.path.join(OUT, "split_linkreg.pt"))
    torch.save(nodelabel, os.path.join(OUT, "split_nodelabel.pt"))
    json.dump({"patient": pat_id, "stay": stay_id,
               "diagnosis": dx_id, "treatment": tr_id, "medication": med_id},
              open(os.path.join(OUT, "id_maps.json"), "w"))
    meta = {
        "seed": SEED,
        "node_counts": {nt: int(data[nt].num_nodes) for nt in data.node_types},
        "edge_counts": {"__".join(et): int(data[et].edge_index.size(1))
                        for et in data.edge_types},
        "mortality": {"pos": int(stay_y.sum()), "n": int(n_stay)},
        "cuts_minutes": {"q60": float(q60), "q70": float(q70), "q80": float(q80)},
        "linkpred_sizes": {k: int(v["edge_label"].numel()) for k, v in linkpred.items()},
        "linkreg_sizes": {k: int(v["y_log_dose"].numel()) for k, v in linkreg.items()},
    }
    json.dump(meta, open(os.path.join(OUT, "meta.json"), "w"), indent=2)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
