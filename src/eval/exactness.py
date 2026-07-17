"""The exactness gate (Golden rule).

``check(a, b)`` compares an incremental output against the from-scratch ground
truth and reports max/mean absolute error plus pass/fail under the configured
ATOL/RTOL. Every incremental code path must go through this before any of its
timings are reported.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Union

import torch
import yaml

from src.paths import CONFIGS

TensorOrDict = Union[torch.Tensor, Dict[str, torch.Tensor]]


def tolerances() -> tuple[float, float]:
    with open(os.path.join(CONFIGS, "common.yaml")) as fh:
        cfg = yaml.safe_load(fh)
    return float(cfg["atol"]), float(cfg["rtol"])


@dataclass
class ExactnessReport:
    max_abs_err: float
    mean_abs_err: float
    passed: bool
    atol: float
    rtol: float

    def __str__(self):
        s = "PASS" if self.passed else "FAIL"
        return (f"exactness {s}: max_abs={self.max_abs_err:.3e} "
                f"mean_abs={self.mean_abs_err:.3e} (atol={self.atol}, rtol={self.rtol})")


def check(incremental: TensorOrDict, ground_truth: TensorOrDict) -> ExactnessReport:
    atol, rtol = tolerances()
    if isinstance(incremental, dict):
        keys = set(incremental) | set(ground_truth)
        pairs = [(incremental[k], ground_truth[k]) for k in sorted(keys)]
    else:
        pairs = [(incremental, ground_truth)]

    max_err, sum_err, n, ok = 0.0, 0.0, 0, True
    for a, b in pairs:
        if a.shape != b.shape:
            raise AssertionError(f"shape mismatch: {tuple(a.shape)} vs {tuple(b.shape)}")
        d = (a - b).abs()
        max_err = max(max_err, float(d.max()) if d.numel() else 0.0)
        sum_err += float(d.sum())
        n += d.numel()
        ok = ok and bool(torch.allclose(a, b, atol=atol, rtol=rtol))
    return ExactnessReport(max_err, sum_err / max(n, 1), ok, atol, rtol)


def assert_exact(incremental: TensorOrDict, ground_truth: TensorOrDict) -> ExactnessReport:
    rep = check(incremental, ground_truth)
    if not rep.passed:
        raise AssertionError(str(rep))
    return rep
