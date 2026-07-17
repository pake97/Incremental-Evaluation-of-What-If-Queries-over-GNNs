# LDBC SNB → property graph

Raw data: **LDBC Social Network Benchmark** CSV export at `dataset/snb/`
(pipe-delimited, one file per node type and per relation). Unlike FAERS/eICU
this is *already* a labeled property graph, so `build_snb.py` is a direct schema
mapping, not a relational construction — the same situation as FinBench. Built
into `dataset/snb/processed/` under the common contract.

Scale: **|V| ≈ 367k, |E| ≈ 1.44M, 8 node types, 20 relations** — the largest
graph in the suite by node count and the richest by relation count.

## Modeling choices

- **Node types (8)**: `person`, `comment`, `post`, `forum`, `tag`, `tagclass`,
  `organisation` (Company + University, distinguished by a type feature),
  `place` (City/Country/Continent). SNB ids are large int64 keys; each type gets
  a contiguous id map (`id_maps.json`) so interventions reference positions.
- **Features** (PII and free text dropped, per the FinBench convention —
  names, email, IP, message `content` are excluded): `person` = age (z, from
  birthday vs a 2013 reference), browser 1-hot, #languages (z); `comment`/`post`
  = length (z) + browser 1-hot (+ has-image bit for posts); `organisation` /
  `place` = type 1-hot; `forum` / `tag` / `tagclass` = constant (structural
  carriers for message passing).
- **Relations (20)**: the full SNB schema — social (`knows`), engagement
  (`likes_post`, `likes_comment`, `hasInterest`, `hasMember`), tagging
  (`post/comment/forum hasTag tag`), authorship (`hasCreator`, `hasModerator`,
  `inForum`), threading (`replyOf_post`, `replyOf_comment`), affiliation
  (`studyAt`, `workAt`), and the geo/tag ontologies (`locatedIn`, `isPartOf`,
  `hasType`, `subclassOf`). `edge_time` (unix sec from `creationDate`) is
  attached wherever a timestamp exists, so the temporal message-passing cut
  applies; the static ontology relations carry no time and are kept whole.

## Tasks (master seed 42)

- **Link prediction** — `person knows person`, the canonical SNB
  friend-recommendation task. Temporal split at `creationDate` quantiles
  q60/q70/q80, 1:1 negatives over person pairs never linked. (knows is stored
  once per undirected pair; the encoder derives the reverse relation.)
- **Node labelling** — person **gender** (834 female / 805 male), a genuine
  node attribute independent of the graph structure and therefore a clean
  transductive target. **Dropped from the feature matrix** to prevent leakage.
  Stratified 70/10/20 masks.
- **No link regression** — SNB has no meaningful continuous edge target, so it
  is deliberately not declared (a task is not forced where
  it is meaningless). `src/models/tasks.py` lists `tasks = [linkpred, nodelabel]`.
