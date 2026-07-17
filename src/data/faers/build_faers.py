"""Build the FAERS Drug-AdverseEvent heterogeneous graph from aggregates.pkl.

Mirrors src/data/finbench/prepare_finbench.py.  Reuses the generic
src/models/model.py and src/data/graph_state.py downstream.

Graph (7 node types, full ontology for message passing):
  drug            DrugBank ID            feat: ATC_Class + Subclass_3 one-hot
  term            MedDRA PT (AE U indic.) feat: SOC one-hot
  atc_subclass1   ATC level-1 subclass   feat: const
  atc_class       ATC top class          feat: const
  hlt, hlgt, soc  MedDRA hierarchy        feat: const

Edges:
  drug -causes->        term   (AE co-report count, dated)   [CORE: link tasks]
  drug -indicated_for-> term   (indication count)
  drug -has_atc1->      atc_subclass1 ; atc_subclass1 -in_class-> atc_class
  term -in_hlt-> hlt ; hlt -in_hlgt-> hlgt ; hlgt -in_soc-> soc

Tasks:
  node label : drug 'serious-prone' = serious-fraction > 0.8 (support >= 5)
  link pred  : drug -causes-> term     (temporal split, 1:1 negatives)
  link reg   : co-report count on drug -causes-> term (log1p target)
"""
from __future__ import annotations
import csv, json, os, pickle
import numpy as np
import torch
from torch_geometric.data import HeteroData

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
CSVD = os.path.join(ROOT, "dataset", "faers", "CSV")
OUT = os.path.join(ROOT, "dataset", "faers", "processed")
SEED = 42
SERIOUS_TH = 0.8
SUPPORT_MIN = 5
Q_MSG, Q_TRAIN, Q_VAL = 0.60, 0.70, 0.80
rng = np.random.default_rng(SEED)


def onehot(values, vocab):
    idx = {v: i for i, v in enumerate(vocab)}
    out = np.zeros((len(values), len(vocab)), dtype=np.float32)
    for r, v in enumerate(values):
        if v in idx:
            out[r, idx[v]] = 1.0
    return out


def zscore(x):
    x = np.asarray(x, np.float64); mu, sd = x.mean(), x.std() or 1.0
    return ((x - mu) / sd).astype(np.float32)


def main():
    agg = pickle.load(open(os.path.join(OUT, "aggregates.pkl"), "rb"))
    da, da_date = agg["da_count"], agg["da_date"]
    di = agg["di_count"]
    drug_n, drug_ser = agg["drug_n"], agg["drug_ser"]

    # ---- ontology dictionaries ----
    atc = {r["DBID"]: r for r in csv.DictReader(open(
        os.path.join(CSVD, "ATC_ontology_DBID_stringname_FAERSCode.csv")))}
    md_rows = list(csv.DictReader(open(os.path.join(CSVD, "MedDRA_dic_all.csv"))))
    pt_key = list(md_rows[0].keys())[0]  # BOM on 'PT'
    med = {r[pt_key]: r for r in md_rows}

    # ---- vocabularies ----
    drugs = sorted(drug_n)
    terms = sorted({a for _, a in da} | {t for _, t in di})
    d_ix = {d: i for i, d in enumerate(drugs)}
    t_ix = {t: i for i, t in enumerate(terms)}

    sub1 = sorted({atc[d]["ATC_Subclass_1"] for d in drugs if d in atc})
    cls = sorted({atc[d]["ATC_Class"] for d in drugs if d in atc})
    hlt = sorted({med[t]["HLT"] for t in terms if t in med})
    hlgt = sorted({med[t]["HLGT"] for t in terms if t in med})
    soc = sorted({med[t]["SOC"] for t in terms if t in med})
    s1_ix = {v: i for i, v in enumerate(sub1)}; cl_ix = {v: i for i, v in enumerate(cls)}
    hlt_ix = {v: i for i, v in enumerate(hlt)}; hlgt_ix = {v: i for i, v in enumerate(hlgt)}
    soc_ix = {v: i for i, v in enumerate(soc)}

    data = HeteroData()

    # ---- node features ----
    cls3 = sorted({atc[d]["ATC_Subclass_3"] for d in drugs if d in atc})
    data["drug"].x = torch.tensor(np.hstack([
        onehot([atc.get(d, {}).get("ATC_Class", "?") for d in drugs], cls),
        onehot([atc.get(d, {}).get("ATC_Subclass_3", "?") for d in drugs], cls3),
    ]), dtype=torch.float32)
    data["term"].x = torch.tensor(
        onehot([med.get(t, {}).get("SOC", "?") for t in terms], soc), dtype=torch.float32)
    for nt, vocab in [("atc_subclass1", sub1), ("atc_class", cls),
                      ("hlt", hlt), ("hlgt", hlgt), ("soc", soc)]:
        data[nt].x = torch.ones((len(vocab), 1), dtype=torch.float32)
    for nt in ["drug", "term", "atc_subclass1", "atc_class", "hlt", "hlgt", "soc"]:
        data[nt].num_nodes = data[nt].x.size(0)

    # ---- drug serious-prone label (on drug) ----
    frac = np.array([drug_ser[d] / drug_n[d] for d in drugs])
    sup = np.array([drug_n[d] for d in drugs])
    y = (frac > SERIOUS_TH).astype(np.int64)
    labelable = sup >= SUPPORT_MIN
    data["drug"].y = torch.tensor(y)
    data["drug"].serious_frac = torch.tensor(frac, dtype=torch.float32)
    data["drug"].support = torch.tensor(sup)

    # ---- core edges: drug -causes-> term (dated, weighted) ----
    src, dst, cnt, tim = [], [], [], []
    for (d, a), c in da.items():
        src.append(d_ix[d]); dst.append(t_ix[a]); cnt.append(c)
        tim.append(da_date.get((d, a), 0))
    cnt = np.array(cnt, np.float64); tim = np.array(tim, np.float64)
    tim[tim == 0] = tim[tim > 0].min()  # undated -> earliest (stays in train/msg)
    ei = torch.tensor([src, dst], dtype=torch.long)
    data["drug", "causes", "term"].edge_index = ei
    data["drug", "causes", "term"].edge_attr = torch.tensor(
        np.stack([np.log1p(cnt), zscore(tim)], 1), dtype=torch.float32)
    data["drug", "causes", "term"].edge_time = torch.tensor(tim)
    data["drug", "causes", "term"].count = torch.tensor(cnt)

    # ---- drug -indicated_for-> term ----
    si, ti, ci = [], [], []
    for (d, t), c in di.items():
        if t in t_ix:
            si.append(d_ix[d]); ti.append(t_ix[t]); ci.append(c)
    data["drug", "indicated_for", "term"].edge_index = torch.tensor([si, ti], dtype=torch.long)
    data["drug", "indicated_for", "term"].edge_attr = torch.tensor(
        np.stack([np.log1p(np.array(ci, np.float64)), np.zeros(len(ci))], 1), dtype=torch.float32)

    # ---- ontology edges ----
    def onto(et, pairs):
        if pairs:
            data[et].edge_index = torch.tensor(list(zip(*pairs)), dtype=torch.long)
        else:
            data[et].edge_index = torch.zeros((2, 0), dtype=torch.long)

    onto(("drug", "has_atc1", "atc_subclass1"),
         [(d_ix[d], s1_ix[atc[d]["ATC_Subclass_1"]]) for d in drugs if d in atc])
    onto(("atc_subclass1", "in_class", "atc_class"),
         list({(s1_ix[atc[d]["ATC_Subclass_1"]], cl_ix[atc[d]["ATC_Class"]]) for d in drugs if d in atc}))
    onto(("term", "in_hlt", "hlt"),
         [(t_ix[t], hlt_ix[med[t]["HLT"]]) for t in terms if t in med])
    onto(("hlt", "in_hlgt", "hlgt"),
         list({(hlt_ix[med[t]["HLT"]], hlgt_ix[med[t]["HLGT"]]) for t in terms if t in med}))
    onto(("hlgt", "in_soc", "soc"),
         list({(hlgt_ix[med[t]["HLGT"]], soc_ix[med[t]["SOC"]]) for t in terms if t in med}))

    # ---- temporal link splits on drug-causes-term ----
    t = tim; q60, q70, q80 = np.quantile(t[t > t.min()], [Q_MSG, Q_TRAIN, Q_VAL]) \
        if (t > t.min()).any() else np.quantile(t, [Q_MSG, Q_TRAIN, Q_VAL])
    eia = ei.numpy()
    def win(lo, hi):
        m = (t >= lo) & (t < hi) if hi is not None else (t >= lo)
        return eia[:, m], cnt[m]
    pos = {"train": win(q60, q70), "val": win(q70, q80), "test": win(q80, None)}
    mp_cut = {"train": q60, "val": q70, "test": q80}
    existing = set(zip(eia[0].tolist(), eia[1].tolist()))
    n_drug, n_term = len(drugs), len(terms)

    def neg(k):
        out = []
        while len(out) < k:
            s = rng.integers(0, n_drug, k); d = rng.integers(0, n_term, k)
            for ss, dd in zip(s.tolist(), d.tolist()):
                if (ss, dd) not in existing:
                    out.append((ss, dd))
                    if len(out) == k: break
        return np.array(out).T

    linkpred, linkreg = {}, {}
    for sp in ("train", "val", "test"):
        p_ei, p_c = pos[sp]; np_ = p_ei.shape[1]
        ng = neg(np_)
        linkpred[sp] = {
            "edge_label_index": torch.tensor(np.hstack([p_ei, ng]), dtype=torch.long),
            "edge_label": torch.tensor(np.hstack([np.ones(np_), np.zeros(np_)]), dtype=torch.float32),
            "mp_time_cut": float(mp_cut[sp])}
        linkreg[sp] = {
            "edge_index": torch.tensor(p_ei, dtype=torch.long),
            "y_log_count": torch.tensor(np.log1p(p_c), dtype=torch.float32),
            "y_count": torch.tensor(p_c),
            "mp_time_cut": float(mp_cut[sp])}

    # ---- node-label split (drug, stratified over labelable) ----
    train = np.zeros(n_drug, bool); val = np.zeros(n_drug, bool); test = np.zeros(n_drug, bool)
    for c in (0, 1):
        idx = np.where((y == c) & labelable)[0]; rng.shuffle(idx)
        a, b = int(.7 * len(idx)), int(.8 * len(idx))
        train[idx[:a]] = True; val[idx[a:b]] = True; test[idx[b:]] = True
    nodelabel = {"y": data["drug"].y, "train_mask": torch.tensor(train),
                 "val_mask": torch.tensor(val), "test_mask": torch.tensor(test)}

    os.makedirs(OUT, exist_ok=True)
    torch.save(data, os.path.join(OUT, "faers_hetero.pt"))
    torch.save(linkpred, os.path.join(OUT, "split_linkpred.pt"))
    torch.save(linkreg, os.path.join(OUT, "split_linkreg.pt"))
    torch.save(nodelabel, os.path.join(OUT, "split_nodelabel.pt"))
    json.dump({"drug": {d: i for d, i in d_ix.items()}, "term": t_ix},
              open(os.path.join(OUT, "id_maps.json"), "w"))

    meta = {
        "seed": SEED, "serious_threshold": SERIOUS_TH, "support_min": SUPPORT_MIN,
        "node_counts": {nt: int(data[nt].num_nodes) for nt in data.node_types},
        "feat_dims": {nt: int(data[nt].x.size(1)) for nt in data.node_types},
        "edge_counts": {f"{a}-{r}-{b}": int(data[a, r, b].edge_index.size(1))
                        for (a, r, b) in data.edge_types},
        "drug_label": {"pos": int(((y == 1) & labelable).sum()),
                       "neg": int(((y == 0) & labelable).sum()),
                       "labelable": int(labelable.sum()), "unlabeled": int((~labelable).sum())},
        "split_dates": {"q60": float(q60), "q70": float(q70), "q80": float(q80)},
        "linkpred_sizes": {sp: int(linkpred[sp]["edge_label"].numel()) for sp in linkpred},
        "linkreg_sizes": {sp: int(linkreg[sp]["y_count"].numel()) for sp in linkreg},
    }
    json.dump(meta, open(os.path.join(OUT, "meta.json"), "w"), indent=2)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
