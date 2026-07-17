"""Prepare the LDBC FinBench graph for GNN training.

Reads the raw pipe-delimited CSVs in ``raw/`` and writes, to ``processed/``:

  finbench_hetero.pt   a PyG HeteroData object (5 node types, 13 relations)
  split_linkpred.pt    temporal link-prediction split (transfer edges + negs)
  split_linkreg.pt     temporal link-regression split (transfer amount target)
  split_nodelabel.pt   stratified account fraud (isBlocked) split
  id_maps.json         raw id <-> contiguous index, per node type
  meta.json            dims, counts, split times, normalisation stats

Design notes
------------
* Three tasks, all Account-centric:
    - link prediction : will an ``account -transfer-> account`` edge form in a
      future time window?  (negatives sampled over account pairs, 1:1)
    - link regression : the transfer ``amount`` on such an edge (log1p target)
    - node labelling  : Account.isBlocked (5.1% positive).  Company / Person
      carry no positive labels in this dump, so node labelling is on Account.
* Temporal protocol (no future leakage).  Transfers are ordered by time and
  cut at the 0.60 / 0.70 / 0.80 quantiles into four windows:
      msg  (<q60)  ->  supervise [q60,q70)   = TRAIN
      observe(<q70)->  supervise [q70,q80)   = VAL
      observe(<q80)->  supervise [q80, end]  = TEST
  Every timestamped edge is masked to the message window when building the
  message-passing graph for a given split (see ``edge_time`` on each relation).
* Feature encoding is deliberately simple (one-hot for low-card categoricals,
  z-scored epochs / log-amounts).  High-card free-text / PII fields are dropped.

Run:  python dataset/finbench01/prepare_finbench.py
"""
from __future__ import annotations

import csv
import json
import os
from datetime import datetime

import numpy as np
import torch
from torch_geometric.data import HeteroData

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
RAW = os.path.join(ROOT, "dataset", "finbench01", "raw")
OUT = os.path.join(ROOT, "dataset", "finbench01", "processed")
SEED = 42

# Temporal split quantiles over transfer createTime.
Q_MSG, Q_TRAIN, Q_VAL = 0.60, 0.70, 0.80

# Ordinal scales for the two naturally-ordered categoricals.
ACCOUNT_LEVEL_ORDER = [
    "Basic level", "Silver level", "Gold level",
    "Platinum level", "Diamond level", "Elite level",
]
RISK_LEVEL_ORDER = [
    "Minimal risk", "Low risk", "Moderate risk", "Significant risk",
    "High risk", "Very high risk", "Severe risk", "Critical risk", "Extreme risk",
]
EMAIL_TOP = ["gmail.com", "gmx.com", "yahoo.com", "hotmail.com", "zoho.com"]

rng = np.random.default_rng(SEED)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def load(name):
    with open(os.path.join(RAW, name), newline="") as fh:
        return list(csv.DictReader(fh, delimiter="|"))


def parse_time(s):
    """Parse 'YYYY-MM-DD HH:MM:SS.fff' (or epoch-millis ints) -> unix seconds."""
    s = s.strip()
    if s.isdigit():  # epochMillis (lastLoginTime)
        return int(s) / 1000.0
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(s, fmt).timestamp()
        except ValueError:
            continue
    return float("nan")


def zscore(x):
    x = np.asarray(x, dtype=np.float64)
    mu, sd = np.nanmean(x), np.nanstd(x) or 1.0
    return (np.nan_to_num(x, nan=mu) - mu) / sd, (float(mu), float(sd))


def onehot(values, vocab):
    idx = {v: i for i, v in enumerate(vocab)}
    out = np.zeros((len(values), len(vocab)), dtype=np.float32)
    for r, v in enumerate(values):
        if v in idx:
            out[r, idx[v]] = 1.0
    return out


def ordinal(values, order):
    idx = {v: i for i, v in enumerate(order)}
    n = max(len(order) - 1, 1)
    return np.array([[idx.get(v, 0) / n] for v in values], dtype=np.float32)


def freq_encode(values):
    from collections import Counter
    c = Counter(values)
    tot = len(values)
    return np.array([[c[v] / tot] for v in values], dtype=np.float32)


# ---------------------------------------------------------------------------
# Node tables -> id maps, features, labels
# ---------------------------------------------------------------------------
def build_nodes():
    feats, idmaps, labels, raw_time = {}, {}, {}, {}

    # ---- Account ----------------------------------------------------------
    # Kept: isBlocked (label), accoutType, freqLoginType, accountLevel.
    a = load("Account.csv")
    idmaps["account"] = {int(r["accountId"]): i for i, r in enumerate(a)}
    feats["account"] = np.hstack([
        onehot([r["accoutType"] for r in a], sorted({r["accoutType"] for r in a})),
        onehot([r["freqLoginType"] for r in a], sorted({r["freqLoginType"] for r in a})),
        ordinal([r["accountLevel"] for r in a], ACCOUNT_LEVEL_ORDER),
    ]).astype(np.float32)
    labels["account"] = np.array(
        [1 if r["isBlocked"] == "true" else 0 for r in a], dtype=np.int64)
    raw_time["account"] = np.array([parse_time(r["createTime"]) for r in a])

    # ---- Company ----------------------------------------------------------
    # Kept: country, business (frequency-encoded).
    c = load("Company.csv")
    idmaps["company"] = {int(r["companyId"]): i for i, r in enumerate(c)}
    feats["company"] = np.hstack([
        freq_encode([r["country"] for r in c]),
        freq_encode([r["business"] for r in c]),
    ]).astype(np.float32)

    # ---- Person -----------------------------------------------------------
    # Kept: gender, birthday (as z-scored age), country.
    p = load("Person.csv")
    idmaps["person"] = {int(r["personId"]): i for i, r in enumerate(p)}
    age, age_s = zscore([parse_time(r["birthday"]) for r in p])
    feats["person"] = np.hstack([
        age[:, None],
        onehot([r["gender"] for r in p], ["male", "female"]),
        freq_encode([r["country"] for r in p]),
    ]).astype(np.float32)

    # ---- Loan -------------------------------------------------------------
    # Kept: loanAmount, balance, createTime, loanUsage, interestRate.
    ln = load("Loan.csv")
    idmaps["loan"] = {int(r["loanId"]): i for i, r in enumerate(ln)}
    la, la_s = zscore([np.log1p(float(r["loanAmount"])) for r in ln])
    ba, ba_s = zscore([np.log1p(float(r["balance"])) for r in ln])
    ir, ir_s = zscore([float(r["interestRate"]) for r in ln])
    lct, lct_s = zscore([parse_time(r["createTime"]) for r in ln])
    feats["loan"] = np.hstack([
        la[:, None], ba[:, None], ir[:, None], lct[:, None],
        onehot([r["loanUsage"] for r in ln], sorted({r["loanUsage"] for r in ln})),
    ]).astype(np.float32)

    # ---- Medium -----------------------------------------------------------
    # Kept: mediumType, riskLevel.
    m = load("Medium.csv")
    idmaps["medium"] = {int(r["mediumId"]): i for i, r in enumerate(m)}
    feats["medium"] = np.hstack([
        onehot([r["mediumType"] for r in m], sorted({r["mediumType"] for r in m})),
        ordinal([r["riskLevel"] for r in m], RISK_LEVEL_ORDER),
    ]).astype(np.float32)

    norm_stats = {
        "person.age": age_s,
        "loan.logAmount": la_s, "loan.logBalance": ba_s, "loan.interestRate": ir_s,
    }
    return feats, idmaps, labels, raw_time, norm_stats


# ---------------------------------------------------------------------------
# Edge tables
# ---------------------------------------------------------------------------
# (csv, src_type, src_col, dst_type, dst_col, relation, amount_col)
EDGE_SPECS = [
    ("AccountTransferAccount.csv", "account", "fromId", "account", "toId", "transfer", "amount"),
    ("AccountWithdrawAccount.csv", "account", "fromId", "account", "toId", "withdraw", "amount"),
    ("AccountRepayLoan.csv", "account", "accountId", "loan", "loanId", "repay", "amount"),
    ("LoanDepositAccount.csv", "loan", "loanId", "account", "accountId", "deposit", "amount"),
    ("CompanyApplyLoan.csv", "company", "companyId", "loan", "loanId", "apply", None),
    ("PersonApplyLoan.csv", "person", "personId", "loan", "loanId", "apply", None),
    ("CompanyGuaranteeCompany.csv", "company", "fromId", "company", "toId", "guarantee", None),
    ("PersonGuaranteePerson.csv", "person", "fromId", "person", "toId", "guarantee", None),
    ("CompanyInvestCompany.csv", "company", "investorId", "company", "companyId", "invest", "ratio"),
    ("PersonInvestCompany.csv", "person", "investorId", "company", "companyId", "invest", "ratio"),
    ("CompanyOwnAccount.csv", "company", "companyId", "account", "accountId", "own", None),
    ("PersonOwnAccount.csv", "person", "personId", "account", "accountId", "own", None),
    ("MediumSignInAccount.csv", "medium", "mediumId", "account", "accountId", "signin", None),
]


def build_edges(idmaps):
    """Return {rel: dict(edge_index, edge_attr, edge_time, amount)}."""
    edges = {}
    for csv_name, st, sc, dt, dc, rel, amt_col in EDGE_SPECS:
        rows = load(csv_name)
        smap, dmap = idmaps[st], idmaps[dt]
        src, dst, attr, times, amts = [], [], [], [], []
        for r in rows:
            si, di = smap.get(int(r[sc])), dmap.get(int(r[dc]))
            if si is None or di is None:
                continue
            t = parse_time(r["createTime"])
            src.append(si); dst.append(di); times.append(t)
            a = float(r[amt_col]) if amt_col else 0.0
            amts.append(a)
            attr.append([np.log1p(a) if amt_col == "amount" else a, t])
        key = (st, rel, dt)
        ei = torch.tensor([src, dst], dtype=torch.long)
        attr = np.asarray(attr, dtype=np.float64)
        # z-score the time column of edge_attr (keep amount/ratio as-is/log)
        if attr.size:
            tcol = attr[:, 1]
            attr[:, 1] = (tcol - np.nanmean(tcol)) / (np.nanstd(tcol) or 1.0)
        edges[key] = {
            "edge_index": ei,
            "edge_attr": torch.tensor(attr, dtype=torch.float32) if attr.size
                          else torch.zeros((0, 2)),
            "edge_time": torch.tensor(times, dtype=torch.float64),
            "amount": torch.tensor(amts, dtype=torch.float64),
        }
    return edges


# ---------------------------------------------------------------------------
# Assemble HeteroData
# ---------------------------------------------------------------------------
def build_hetero(feats, labels, edges):
    data = HeteroData()
    for nt, x in feats.items():
        data[nt].x = torch.tensor(x, dtype=torch.float32)
        data[nt].num_nodes = x.shape[0]
    for nt, y in labels.items():
        data[nt].y = torch.tensor(y, dtype=torch.long)
    for key, e in edges.items():
        data[key].edge_index = e["edge_index"]
        data[key].edge_attr = e["edge_attr"]
        data[key].edge_time = e["edge_time"]
    return data


# ---------------------------------------------------------------------------
# Temporal splits for transfer edges (link prediction + regression)
# ---------------------------------------------------------------------------
def temporal_transfer_splits(edges, n_accounts):
    e = edges[("account", "transfer", "account")]
    ei = e["edge_index"].numpy()
    t = e["edge_time"].numpy()
    amt = e["amount"].numpy()
    q60, q70, q80 = np.quantile(t, [Q_MSG, Q_TRAIN, Q_VAL])

    def window(lo, hi):
        m = (t >= lo) & (t < hi) if hi is not None else (t >= lo)
        return ei[:, m], amt[m]

    pos = {
        "train": window(q60, q70),
        "val": window(q70, q80),
        "test": window(q80, None),
    }
    # message-passing edge masks (everything strictly before the window start)
    mp_cut = {"train": q60, "val": q70, "test": q80}

    # existing transfer pairs at ANY time -> forbidden negatives
    existing = set(map(tuple, ei.T.tolist()))
    senders = np.unique(ei[0])

    def sample_neg(n):
        out = []
        while len(out) < n:
            s = senders[rng.integers(0, len(senders), size=n)]
            d = rng.integers(0, n_accounts, size=n)
            for ss, dd in zip(s.tolist(), d.tolist()):
                if ss != dd and (ss, dd) not in existing:
                    out.append((ss, dd))
                    if len(out) == n:
                        break
        return np.array(out, dtype=np.int64).T

    linkpred, linkreg = {}, {}
    for sp in ("train", "val", "test"):
        p_ei, p_amt = pos[sp]
        npos = p_ei.shape[1]
        neg = sample_neg(npos)
        label_index = np.hstack([p_ei, neg])
        label = np.hstack([np.ones(npos), np.zeros(npos)]).astype(np.float32)
        linkpred[sp] = {
            "edge_label_index": torch.tensor(label_index, dtype=torch.long),
            "edge_label": torch.tensor(label, dtype=torch.float32),
            "mp_time_cut": float(mp_cut[sp]),
        }
        linkreg[sp] = {  # regression only over real (positive) edges
            "edge_index": torch.tensor(p_ei, dtype=torch.long),
            "y_amount": torch.tensor(p_amt, dtype=torch.float64),
            "y_log_amount": torch.tensor(np.log1p(p_amt), dtype=torch.float32),
            "mp_time_cut": float(mp_cut[sp]),
        }
    return linkpred, linkreg, (float(q60), float(q70), float(q80))


# ---------------------------------------------------------------------------
# Stratified node-label split (Account.isBlocked)
# ---------------------------------------------------------------------------
def node_label_split(y, frac=(0.7, 0.1, 0.2)):
    n = len(y)
    train = np.zeros(n, bool); val = np.zeros(n, bool); test = np.zeros(n, bool)
    for cls in (0, 1):
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        a, b = int(frac[0] * len(idx)), int((frac[0] + frac[1]) * len(idx))
        train[idx[:a]] = True; val[idx[a:b]] = True; test[idx[b:]] = True
    return {
        "y": torch.tensor(y, dtype=torch.long),
        "train_mask": torch.tensor(train),
        "val_mask": torch.tensor(val),
        "test_mask": torch.tensor(test),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    os.makedirs(OUT, exist_ok=True)
    print("Loading nodes ...")
    feats, idmaps, labels, raw_time, norm_stats = build_nodes()
    print("Loading edges ...")
    edges = build_edges(idmaps)
    data = build_hetero(feats, labels, edges)

    print("Building temporal link splits ...")
    n_acc = feats["account"].shape[0]
    linkpred, linkreg, qtimes = temporal_transfer_splits(edges, n_acc)
    print("Building node-label split ...")
    nodelabel = node_label_split(labels["account"])

    torch.save(data, os.path.join(OUT, "finbench_hetero.pt"))
    torch.save(linkpred, os.path.join(OUT, "split_linkpred.pt"))
    torch.save(linkreg, os.path.join(OUT, "split_linkreg.pt"))
    torch.save(nodelabel, os.path.join(OUT, "split_nodelabel.pt"))
    with open(os.path.join(OUT, "id_maps.json"), "w") as fh:
        json.dump({k: {str(a): b for a, b in v.items()} for k, v in idmaps.items()}, fh)

    meta = {
        "seed": SEED,
        "node_types": {nt: int(feats[nt].shape[0]) for nt in feats},
        "feat_dims": {nt: int(feats[nt].shape[1]) for nt in feats},
        "relations": [list(k) for k in edges],
        "edge_counts": {f"{k[0]}-{k[1]}-{k[2]}": int(e["edge_index"].shape[1])
                        for k, e in edges.items()},
        "account_label_balance": {
            "pos": int(labels["account"].sum()),
            "neg": int((labels["account"] == 0).sum()),
        },
        "split_quantiles": {"q60": qtimes[0], "q70": qtimes[1], "q80": qtimes[2]},
        "linkpred_sizes": {sp: int(linkpred[sp]["edge_label"].numel()) for sp in linkpred},
        "linkreg_sizes": {sp: int(linkreg[sp]["y_amount"].numel()) for sp in linkreg},
        "nodelabel_sizes": {
            "train": int(nodelabel["train_mask"].sum()),
            "val": int(nodelabel["val_mask"].sum()),
            "test": int(nodelabel["test_mask"].sum()),
        },
        "norm_stats": norm_stats,
    }
    with open(os.path.join(OUT, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)

    print("\n=== DONE ===")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
