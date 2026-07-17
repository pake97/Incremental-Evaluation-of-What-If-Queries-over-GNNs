# eICU → property graph construction

Raw data: PhysioNet **eICU-CRD demo 2.0.1** at
`dataset/physionet.org/files/eicu-crd-demo/2.0.1/`. Built by `build_eicu.py`
into `dataset/eicu/processed/` under the same contract as FinBench/FAERS.

## Two releases, one builder

`build_eicu.py --release {demo,full}` handles both (identical schema, different
scale). `demo` (default) → dataset key `eicu`. `full` → key `eicu_full`,
reading `dataset/physionet.org/files/eicu-crd/2.0/` and writing
`dataset/eicu_full/processed/`.

The **full release (~200k unit stays / ~139k patients, ~11M edges) is
credentialed** — it is NOT downloadable non-interactively and must never be
committed. Obtain it once with your own PhysioNet account (requires CITI
training + signed DUA on https://physionet.org/content/eicu-crd/2.0/):

```bash
cd dataset/physionet.org/files
wget -r -N -c -np --user YOUR_PHYSIONET_USER --ask-password \
    https://physionet.org/files/eicu-crd/2.0/
# then, from the repo root:
venv/bin/python -m src.data.eicu.build_eicu --release full
venv/bin/python -m src.models.train --dataset eicu_full --max-epochs 40
```

Everything downstream (loader, tasks, all four experiments) already recognizes
`eicu_full`; add it to an experiment config's `datasets:` list to include it. Like
FAERS, this relational→graph construction is a **methods decision of the
paper**; the choices and their rationale:

## Modeling choices

- **Node types (5)** — `patient` (uniquepid; gender one-hot + age z, with
  `> 89` coded as 90), `stay` (ICU unit stay = patientunitstayid; unittype +
  unitstaytype one-hots, admission weight/height z), and three **concept**
  types: `diagnosis` (first 2 levels of the `diagnosisstring` path, 110
  concepts), `treatment` (first 3 levels of `treatmentstring`, 414 concepts),
  `medication` (HICL code `drughiclseqno` when present, else the drugname with
  numbers stripped; 1,437 concepts). Concept nodes carry a constant feature —
  they exist to route message passing, not to be classified.
- **Individual event rows are NOT nodes** (same decision as FAERS reports):
  each diagnosis/treatment/medication event becomes a dated edge
  `stay → concept`, keeping the graph small and the link tasks well-posed.
- **Edges** — `patient -has_stay-> stay` (static, no `edge_time` — the trainer
  keeps untimed relations whole when time-masking), and dated
  `stay -diagnosed-> diagnosis` (value = priority ordinal Primary 2 / Major 1 /
  Other 0), `stay -treated-> treatment`, `stay -received-> medication`
  (value = log1p(dose), dose parsed as the leading number of the free-text
  `dosage` field; unparsable → 0). `edge_time` = event offset in **minutes
  from unit admission**; `edge_attr = [value, z(time)]` mirrors the other
  datasets. Cancelled medication orders are dropped.

## Tasks (all seeded 42)

- **Node labelling** — in-hospital mortality on `stay`
  (`hospitaldischargestatus == 'Expired'`, 212/2,520 ≈ 8.4% positive),
  stratified 70/10/20 masks.
- **Link prediction** — will `stay -received-> medication` occur later in the
  stay: temporal cuts at q60/q70/q80 of `drugstartoffset` (offsets are
  per-stay clocks, so a global cut means "predict later-course medications
  from the early course"); 1:1 negatives over (stay, medication) pairs never
  linked.
- **Link regression** — log1p(parsed dose) on `received` edges (positives
  only, same cuts). **Caveat**: the dose comes from a free-text field; the
  leading-number parse is documented lossy (unit heterogeneity mg vs mL is
  not resolved). Treat linkreg on eICU as a secondary task.

Counts and split sizes: `dataset/eicu/processed/meta.json`.
