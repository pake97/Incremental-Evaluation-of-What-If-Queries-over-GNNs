"""Canonical repository paths. Everything path-like goes through here."""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASET = os.path.join(ROOT, "dataset")
FINBENCH_RAW = os.path.join(DATASET, "finbench01", "raw")
FINBENCH_PROC = os.path.join(DATASET, "finbench01", "processed")
FAERS_CSV = os.path.join(DATASET, "faers", "CSV")
FAERS_PROC = os.path.join(DATASET, "faers", "processed")
EICU_RAW = os.path.join(DATASET, "physionet.org", "files", "eicu-crd-demo", "2.0.1")
EICU_PROC = os.path.join(DATASET, "eicu", "processed")
EICU_FULL_RAW = os.path.join(DATASET, "physionet.org", "files", "eicu-crd", "2.0")
EICU_FULL_PROC = os.path.join(DATASET, "eicu_full", "processed")
SNB_RAW = os.path.join(DATASET, "snb")
SNB_PROC = os.path.join(DATASET, "snb", "processed")

CHECKPOINTS = os.path.join(ROOT, "checkpoints")
RESULTS = os.path.join(ROOT, "results")
FIGURES = os.path.join(ROOT, "figures")
CONFIGS = os.path.join(ROOT, "configs")
