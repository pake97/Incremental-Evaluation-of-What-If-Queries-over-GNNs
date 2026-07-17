# FAERS → property graph construction

FAERS is relational (drug / adverse-event report tables); this construction into
a heterogeneous property graph is a **methods decision of the paper**, not an
implementation detail. The pipeline is two scripts, run in order:

1. `scan_faers.py` — one streaming pass over `dataset/faers/CSV/patient_safety.csv`
   (the raw per-report table). Extracts, per (drug, MedDRA term) pair: co-report
   counts, first/last report dates, and the fraction of reports flagged serious.
   Writes `dataset/faers/processed/aggregates.pkl`.
2. `build_faers.py` — builds the PyG `HeteroData` from the aggregates plus the
   ATC ontology and MedDRA dictionary CSVs. Writes `faers_hetero.pt`, the three
   task splits, `id_maps.json`, `meta.json` to `dataset/faers/processed/`.

## Modeling choices

- **Node types (7)** — `drug` (DrugBank id; ATC class/subclass one-hots, 28 dims),
  `term` (MedDRA preferred term, covering both adverse events and indications;
  SOC one-hot, 27 dims), and the ontology levels `atc_subclass1`, `atc_class`,
  `hlt`, `hlgt`, `soc` (constant 1-dim features — they exist to carry message
  passing through the hierarchies, not to be classified).
- **Individual reports are NOT nodes.** Reports are aggregated into weighted
  drug–term edges (co-report count, dated by report dates). This keeps the graph
  at ~3.6k drugs / ~20k terms with ~3.6M `causes` edges instead of tens of
  millions of report nodes, and makes the link tasks well-posed.
- **Edge types** — `drug -causes-> term` (adverse-event co-reports; the core
  relation for the link tasks), `drug -indicated_for-> term`,
  `drug -has_atc1-> atc_subclass1 -in_class-> atc_class`, and the MedDRA chain
  `term -in_hlt-> hlt -in_hlgt-> hlgt -in_soc-> soc`. Money-style
  `edge_attr = [log1p(count), z(time)]` on the report-derived edges mirrors the
  FinBench convention so downstream code is dataset-agnostic.

## Tasks

- **Node labelling** — drug is "serious-prone" if its serious-report fraction
  > 0.8 with support ≥ 5 co-reports (1,698 pos / 3,140 labelable).
- **Link prediction** — will `drug -causes-> term` be reported in a future
  window; temporal split at report-date quantiles q60/q70/q80, 1:1 negatives.
- **Link regression** — log1p co-report count on positive `causes` edges.

Master seed 42 everywhere; split sizes and counts in
`dataset/faers/processed/meta.json`.
