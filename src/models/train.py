"""Train the frozen single-task models for one dataset.

  architectures : rgcn, hgat
  depths        : 1, 2, 3, 4 layers
  tasks         : from the dataset's TaskSpec (linkpred BCE, linkreg MSE on the
                  log target, nodelabel BCE + pos_weight)

Each model is the shared encoder + only its task head.  Light, full-batch,
Adam 1e-3, early stop on the validation metric.  Link tasks encode on a
time-masked message-passing graph (edges strictly before the supervised
window); node labelling is transductive on the full graph.

All what-if experiments load these checkpoints and never retrain: f_theta is
fixed, only the input graph is hypothetically perturbed.

Outputs:  checkpoints/<dataset>/<task>_<arch>_L<n>.pt
          results/train/<dataset>/metrics.csv
Run:      python -m src.models.train --dataset finbench
          python -m src.models.train --dataset faers --max-epochs 60
          python -m src.models.train --dataset finbench --quick   (sanity)
"""
from __future__ import annotations

import argparse, csv, os, random, time
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score

from src.data.loader import load_graph, load_split
from src.models.model import build_model, save_model
from src.models.tasks import spec
from src.paths import CHECKPOINTS, RESULTS

SEED, HIDDEN, LR = 42, 64, 1e-3


def seed_all(s=SEED):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    torch.use_deterministic_algorithms(True)


def mp_edges(data, cut):
    """Edge dict keeping only edges strictly before `cut`; relations without
    an edge_time (static structure, e.g. ontology hierarchies) are kept whole."""
    out = {}
    for et in data.edge_types:
        if hasattr(data[et], "edge_time"):
            keep = data[et].edge_time < cut
            out[et] = data[et].edge_index[:, keep]
        else:
            out[et] = data[et].edge_index
    return out


# ---------------------------------------------------------------------------
# Task trainers
# ---------------------------------------------------------------------------
def train_linkpred(data, sp, split, mt, L, max_epochs, patience):
    seed_all()
    st, dt_ = sp.core_rel[0], sp.core_rel[2]
    model = build_model(data, mt, L, HIDDEN)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    cut = {s: mp_edges(data, split[s]["mp_time_cut"]) for s in ("train", "val", "test")}
    y = {s: split[s]["edge_label"] for s in split}
    idx = {s: split[s]["edge_label_index"] for s in split}

    best, best_state, wait = -1.0, None, 0
    for ep in range(max_epochs):
        model.train(); opt.zero_grad()
        h = model.encode(data.x_dict, cut["train"])
        score = model.link_score(h, idx["train"], src_type=st, dst_type=dt_)
        loss = F.binary_cross_entropy_with_logits(score, y["train"])
        loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            hv = model.encode(data.x_dict, cut["val"])
            s = model.link_score(hv, idx["val"], src_type=st, dst_type=dt_).sigmoid().numpy()
        auc = roc_auc_score(y["val"].numpy(), s)
        if auc > best:
            best, best_state, wait = auc, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(best_state); model.eval()
    with torch.no_grad():
        ht = model.encode(data.x_dict, cut["test"])
        stc = model.link_score(ht, idx["test"], src_type=st, dst_type=dt_).sigmoid().numpy()
    yt = y["test"].numpy()
    return model, {"val_auc": best, "test_auc": roc_auc_score(yt, stc),
                   "test_ap": average_precision_score(yt, stc), "epochs": ep + 1}


def train_linkreg(data, sp, split, mt, L, max_epochs, patience):
    seed_all()
    st, dt_ = sp.core_rel[0], sp.core_rel[2]
    model = build_model(data, mt, L, HIDDEN)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    cut = {s: mp_edges(data, split[s]["mp_time_cut"]) for s in ("train", "val", "test")}
    y = {s: split[s][sp.linkreg_target] for s in split}
    idx = {s: split[s]["edge_index"] for s in split}

    best, best_state, wait = 1e18, None, 0
    for ep in range(max_epochs):
        model.train(); opt.zero_grad()
        h = model.encode(data.x_dict, cut["train"])
        pred = model.link_regress(h, idx["train"], src_type=st, dst_type=dt_)
        loss = F.mse_loss(pred, y["train"])
        loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            hv = model.encode(data.x_dict, cut["val"])
            pv = model.link_regress(hv, idx["val"], src_type=st, dst_type=dt_)
            vrmse = F.mse_loss(pv, y["val"]).sqrt().item()
        if vrmse < best:
            best, best_state, wait = vrmse, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(best_state); model.eval()
    with torch.no_grad():
        ht = model.encode(data.x_dict, cut["test"])
        pred = model.link_regress(ht, idx["test"], src_type=st, dst_type=dt_)
    yt = y["test"]
    return model, {"val_rmse_log": best,
                   "test_rmse_log": F.mse_loss(pred, yt).sqrt().item(),
                   "test_mae_log": (pred - yt).abs().mean().item(), "epochs": ep + 1}


def train_nodelabel(data, sp, split, mt, L, max_epochs, patience):
    seed_all()
    nt = sp.label_nt
    model = build_model(data, mt, L, HIDDEN)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    y = split["y"].float()
    trm, vam, tem = split["train_mask"], split["val_mask"], split["test_mask"]
    npos, nneg = int(y[trm].sum()), int((y[trm] == 0).sum())
    pos_weight = torch.tensor([nneg / max(npos, 1)])

    best, best_state, wait = -1.0, None, 0
    for ep in range(max_epochs):
        model.train(); opt.zero_grad()
        h = model.encode(data.x_dict, data.edge_index_dict)
        logit = model.classify_nodes(h, node_type=nt)
        loss = F.binary_cross_entropy_with_logits(logit[trm], y[trm], pos_weight=pos_weight)
        loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            s = model.classify_nodes(model.encode(data.x_dict, data.edge_index_dict),
                                     node_type=nt).sigmoid().numpy()
        auc = roc_auc_score(y[vam].numpy(), s[vam.numpy()])
        if auc > best:
            best, best_state, wait = auc, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(best_state); model.eval()
    with torch.no_grad():
        s = model.classify_nodes(model.encode(data.x_dict, data.edge_index_dict),
                                 node_type=nt).sigmoid().numpy()
    yt = y[tem].numpy()
    return model, {"val_auc": best, "test_auc": roc_auc_score(yt, s[tem.numpy()]),
                   "test_ap": average_precision_score(yt, s[tem.numpy()]), "epochs": ep + 1}


TRAINERS = {"linkpred": train_linkpred, "linkreg": train_linkreg, "nodelabel": train_nodelabel}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--archs", nargs="+", default=["rgcn", "hgat"])
    ap.add_argument("--layers", nargs="+", type=int, default=[1, 2, 3, 4])
    ap.add_argument("--tasks", nargs="+", default=None)
    ap.add_argument("--max-epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=25)
    ap.add_argument("--quick", action="store_true", help="1 epoch each (sanity)")
    args = ap.parse_args()

    sp = spec(args.dataset)
    tasks = args.tasks or sp.tasks
    max_epochs = 1 if args.quick else args.max_epochs

    models_dir = os.path.join(CHECKPOINTS, args.dataset)
    results_dir = os.path.join(RESULTS, "train", args.dataset)
    os.makedirs(models_dir, exist_ok=True); os.makedirs(results_dir, exist_ok=True)

    data, _ = load_graph(args.dataset)
    rows = []
    for mt in args.archs:
        for L in args.layers:
            for task in tasks:
                split = load_split(args.dataset, task)
                t0 = time.perf_counter()
                model, metrics = TRAINERS[task](data, sp, split, mt, L, max_epochs, args.patience)
                dt = time.perf_counter() - t0
                save_model(model, os.path.join(models_dir, f"{task}_{mt}_L{L}.pt"))
                rows.append({"task": task, "arch": mt, "layers": L, "wall_s": round(dt, 1),
                             **{k: round(v, 4) for k, v in metrics.items()}})
                print(f"{args.dataset:9} {task:10} {mt}-L{L}  "
                      + "  ".join(f"{k}={v}" for k, v in metrics.items()) + f"  ({dt:.0f}s)",
                      flush=True)

    keys = sorted({k for r in rows for k in r})
    lead = ["task", "arch", "layers", "wall_s"]
    with open(os.path.join(results_dir, "metrics.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=lead + [k for k in keys if k not in lead])
        w.writeheader(); w.writerows(rows)
    print(f"\nWrote {len(rows)} rows -> {results_dir}/metrics.csv")


if __name__ == "__main__":
    main()
