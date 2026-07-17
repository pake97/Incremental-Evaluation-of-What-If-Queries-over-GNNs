"""Shared experiment plumbing: configs, manifests, tidy CSVs, checkpoints.

Every experiment is idempotent, resumable, and self-describing: it writes
results/<exp>/manifest.json (config, seed, versions, dataset cards, git commit)
plus tidy CSVs, and re-running with the same config reproduces the numbers.
"""
from __future__ import annotations

import csv
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from typing import List, Optional

import torch
import yaml

from src.paths import CHECKPOINTS, CONFIGS, RESULTS, ROOT


def load_config(path: str) -> dict:
    with open(path) as fh:
        cfg = yaml.safe_load(fh)
    with open(os.path.join(CONFIGS, "common.yaml")) as fh:
        common = yaml.safe_load(fh)
    return {**common, **cfg}


def git_commit() -> Optional[str]:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() if out.returncode == 0 else None
    except OSError:
        return None


def write_manifest(exp: str, cfg: dict, dataset_cards: dict) -> str:
    import torch_geometric
    outdir = os.path.join(RESULTS, exp)
    os.makedirs(outdir, exist_ok=True)
    manifest = {
        "experiment": exp,
        "config": cfg,
        "seed": cfg.get("seed"),
        "versions": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "torch_geometric": torch_geometric.__version__,
            "platform": platform.platform(),
        },
        "git_commit": git_commit(),
        "dataset_cards": dataset_cards,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    path = os.path.join(outdir, "manifest.json")
    with open(path, "w") as fh:
        json.dump(manifest, fh, indent=2)
    return path


def write_csv(exp: str, rows: List[dict], name: str = "results.csv") -> str:
    outdir = os.path.join(RESULTS, exp)
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, name)
    keys: List[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    return path


def dataset_card(dataset: str) -> dict:
    from src.data.loader import GRAPHS
    path = os.path.join(GRAPHS[dataset][0], "dataset_card.json")
    return json.load(open(path)) if os.path.exists(path) else {}


def ckpt_path(dataset: str, task: str, arch: str, L: int) -> str:
    return os.path.join(CHECKPOINTS, dataset, f"{task}_{arch}_L{L}.pt")


def missing_checkpoints(dataset: str, archs, layers, tasks) -> List[tuple]:
    return [(dataset, t, a, L) for a in archs for L in layers for t in tasks
            if not os.path.exists(ckpt_path(dataset, t, a, L))]


def summary_line(exp: str, ok: bool, detail: str) -> int:
    """Print the one-line PASS/FAIL summary; return the process exit code."""
    print(f"{'PASS' if ok else 'FAIL'} {exp}: {detail}")
    return 0 if ok else 1
