"""Build the LDBC SNB heterogeneous graph from the raw pipe-delimited CSVs.

SNB is already a labeled property graph (like FinBench), so this is a direct
schema mapping, not a relational→graph construction. Mirrors the other builders:
same processed contract (hetero .pt + splits + id_maps + meta + dataset card),
consumed by the same loader, models, and maintenance code.

Node types (8):
  person   label = home country (binary: most-common country vs rest), resolved
           from LocationCityId up the Place hierarchy; NOT in features (no
           leakage).  feat: age (z from birthday), browserUsed 1-hot,
           #languages (z)
  comment  feat: length (z), browserUsed 1-hot
  post     feat: length (z), has-image (bin), browserUsed 1-hot
  forum    feat: const                organisation feat: type 1-hot (Company/University)
  tag      feat: const                place        feat: type 1-hot (City/Country/Continent)
  tagclass feat: const

Relations (schema-valid triples). edge_time (unix sec from creationDate) is set
where a creation timestamp exists, so the temporal message-passing cut works:
  person knows person            [CORE — link prediction, temporal split]
  person likes post / comment
  person hasInterest tag ; forum hasMember person ; forum hasTag tag
  post hasTag tag ; comment hasTag tag
  person studyAt organisation ; person workAt organisation
  post hasCreator person ; comment hasCreator person ; forum hasModerator person
  post inForum forum ; comment replyOf post ; comment replyOf comment
  organisation locatedIn place ; place isPartOf place
  tag hasType tagclass ; tagclass subclassOf tagclass

Tasks (master seed 42, temporal protocol as elsewhere):
  node label : person home country (most-common vs rest), stratified 70/10/20.
  link pred  : person knows person; cuts at q60/q70/q80 of creationDate;
               1:1 negatives over never-linked person pairs.
  (no link regression — SNB has no meaningful continuous edge target.)

Run:  python -m src.data.snb.build_snb
"""
from __future__ import annotations

import csv, json, os
from datetime import datetime, timezone

import numpy as np
import torch
from torch_geometric.data import HeteroData

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
RAW = os.path.join(ROOT, "dataset", "snb")
OUT = os.path.join(ROOT, "dataset", "snb", "processed")
SEED = 42
Q_MSG, Q_TRAIN, Q_VAL = 0.60, 0.70, 0.80
AGE_REF_YEAR = 2013  # network's end-of-life; age = ref - birth year


def rows(name):
    with open(os.path.join(RAW, name + ".csv"), newline="") as fh:
        yield from csv.DictReader(fh, delimiter="|")


def to_unix(s):
    if not s:
        return np.nan
    try:
        return datetime.fromisoformat(s).timestamp()
    except ValueError:
        return np.nan


def zscore(v):
    v = np.asarray(v, dtype=np.float32)
    med = np.nanmedian(v)
    v = np.where(np.isnan(v), med, v)
    sd = v.std()
    return (v - v.mean()) / (sd if sd > 1e-8 else 1.0)


def onehot(values, vocab=None):
    vocab = vocab or sorted({v for v in values if v})
    idx = {c: i for i, c in enumerate(vocab)}
    m = np.zeros((len(values), len(vocab)), dtype=np.float32)
    for r, v in enumerate(values):
        if v in idx:
            m[r, idx[v]] = 1.0
    return m, vocab


def main():
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(SEED)

    # ---- nodes: read tables, build per-type id maps (SNB ids -> [0,N)) -----
    def load_nodes(name, key="id"):
        rs = list(rows(name))
        idmap = {r[key]: i for i, r in enumerate(rs)}
        return rs, idmap

    person, pid = load_nodes("Person")
    comment, cid = load_nodes("Comment")
    post, poid = load_nodes("Post")
    forum, fid = load_nodes("Forum")
    tag, tid = load_nodes("Tag")
    tagclass, tcid = load_nodes("TagClass")
    org, oid = load_nodes("Organisation")
    place, plid = load_nodes("Place")

    # ---- person features (+ gender label, dropped from X) ------------------
    def birth_year(s):
        try:
            return float(s.split("-")[0])
        except (ValueError, AttributeError, IndexError):
            return np.nan
    p_age = zscore([AGE_REF_YEAR - birth_year(r["birthday"]) for r in person])
    p_brow, brow_vocab = onehot([r["browserUsed"] for r in person])
    p_nlang = zscore([len(r["language"].split(";")) if r["language"] else 0
                      for r in person])
    person_x = np.concatenate([p_age[:, None], p_brow, p_nlang[:, None]], axis=1)

    # label: home country, resolved city -> ... -> country via Place hierarchy.
    # Derived from LocationCityId, which is NOT a person feature, so no leakage.
    place_type = {r["id"]: r["type"] for r in place}
    place_parent = {r["id"]: (r["PartOfPlaceId"].split(".")[0]
                              if r["PartOfPlaceId"] else None) for r in place}

    def country_of(city_id):
        seen, cur = 0, city_id
        while cur in place_type and seen < 8:
            if place_type[cur] == "Country":
                return cur
            cur = place_parent.get(cur)
            seen += 1
        return None

    countries = [country_of(r["LocationCityId"]) for r in person]
    from collections import Counter
    top_country = Counter(c for c in countries if c).most_common(1)[0][0]
    country_label = np.array([1 if c == top_country else 0 for c in countries],
                             dtype=np.int64)

    c_len = zscore([float(r["length"] or 0) for r in comment])
    c_brow, _ = onehot([r["browserUsed"] for r in comment], brow_vocab)
    comment_x = np.concatenate([c_len[:, None], c_brow], axis=1)

    po_len = zscore([float(r["length"] or 0) for r in post])
    po_img = np.array([[1.0 if r["imageFile"] else 0.0] for r in post], dtype=np.float32)
    po_brow, _ = onehot([r["browserUsed"] for r in post], brow_vocab)
    post_x = np.concatenate([po_len[:, None], po_img, po_brow], axis=1)

    org_x, org_types = onehot([r["type"] for r in org])
    place_x, place_types = onehot([r["type"] for r in place])

    data = HeteroData()
    data["person"].x = torch.from_numpy(person_x.astype(np.float32))
    data["comment"].x = torch.from_numpy(comment_x.astype(np.float32))
    data["post"].x = torch.from_numpy(post_x.astype(np.float32))
    data["forum"].x = torch.ones(len(forum), 1)
    data["tag"].x = torch.ones(len(tag), 1)
    data["tagclass"].x = torch.ones(len(tagclass), 1)
    data["organisation"].x = torch.from_numpy(org_x)
    data["place"].x = torch.from_numpy(place_x)

    # ---- edges -------------------------------------------------------------
    def edge_from_file(name, s_col, d_col, s_map, d_map, time_col="creationDate"):
        src, dst, t = [], [], []
        for r in rows(name):
            a, b = r.get(s_col), r.get(d_col)
            # ids may carry a trailing .0 (float-formatted foreign keys)
            a = a.split(".")[0] if a else a
            b = b.split(".")[0] if b else b
            if a in s_map and b in d_map:
                src.append(s_map[a]); dst.append(d_map[b])
                t.append(to_unix(r.get(time_col, "")) if time_col else np.nan)
        return src, dst, np.array(t, dtype=np.float32)

    def node_derived(recs, node_map, id_key, s_map_ref, ref_key, time_key="creationDate"):
        """Edge from a node's foreign key (e.g. Post.CreatorPersonId)."""
        src, dst, t = [], [], []
        for r in recs:
            b = r.get(ref_key)
            b = b.split(".")[0] if b else b
            if r[id_key] in node_map and b in s_map_ref:
                src.append(node_map[r[id_key]]); dst.append(s_map_ref[b])
                t.append(to_unix(r.get(time_key, "")))
        return src, dst, np.array(t, dtype=np.float32)

    def add(et, src, dst, t=None):
        ei = torch.tensor([src, dst], dtype=torch.long)
        data[et].edge_index = ei
        val = np.zeros(len(src), dtype=np.float32)
        if t is not None and np.isfinite(t).any():
            data[et].edge_time = torch.from_numpy(np.nan_to_num(t, nan=0.0))
            zt = zscore(t)
        else:
            zt = np.zeros(len(src), dtype=np.float32)
        data[et].edge_attr = torch.from_numpy(np.stack([val, zt], axis=1))

    # from edge CSVs (timed)
    add(("person", "knows", "person"),
        *edge_from_file("Person_knows_Person", "Person1Id", "Person2Id", pid, pid))
    add(("person", "likes_post", "post"),
        *edge_from_file("Person_likes_Post", "PersonId", "PostId", pid, poid))
    add(("person", "likes_comment", "comment"),
        *edge_from_file("Person_likes_Comment", "PersonId", "CommentId", pid, cid))
    add(("person", "hasInterest", "tag"),
        *edge_from_file("Person_hasInterest_Tag", "PersonId", "TagId", pid, tid))
    add(("forum", "hasMember", "person"),
        *edge_from_file("Forum_hasMember_Person", "ForumId", "PersonId", fid, pid))
    add(("forum", "hasTag", "tag"),
        *edge_from_file("Forum_hasTag_Tag", "ForumId", "TagId", fid, tid))
    add(("post", "hasTag", "tag"),
        *edge_from_file("Post_hasTag_Tag", "PostId", "TagId", poid, tid))
    add(("comment", "hasTag", "tag"),
        *edge_from_file("Comment_hasTag_Tag", "CommentId", "TagId", cid, tid))
    add(("person", "studyAt", "organisation"),
        *edge_from_file("Person_studyAt_University", "PersonId", "UniversityId",
                        pid, oid, time_col=None))
    add(("person", "workAt", "organisation"),
        *edge_from_file("Person_workAt_Company", "PersonId", "CompanyId",
                        pid, oid, time_col=None))

    # node-derived (from foreign keys on Post/Comment/Forum/Tag/TagClass/Org/Place)
    add(("post", "hasCreator", "person"),
        *node_derived(post, poid, "id", pid, "CreatorPersonId"))
    add(("comment", "hasCreator", "person"),
        *node_derived(comment, cid, "id", pid, "CreatorPersonId"))
    add(("forum", "hasModerator", "person"),
        *node_derived(forum, fid, "id", pid, "ModeratorPersonId"))
    add(("post", "inForum", "forum"),
        *node_derived(post, poid, "id", fid, "ContainerForumId"))
    add(("comment", "replyOf_post", "post"),
        *node_derived(comment, cid, "id", poid, "ParentPostId"))
    add(("comment", "replyOf_comment", "comment"),
        *node_derived(comment, cid, "id", cid, "ParentCommentId"))
    # static structure (no time)
    add(("organisation", "locatedIn", "place"),
        *node_derived(org, oid, "id", plid, "LocationPlaceId", time_key="")[:2])
    add(("place", "isPartOf", "place"),
        *node_derived(place, plid, "id", plid, "PartOfPlaceId", time_key="")[:2])
    add(("tag", "hasType", "tagclass"),
        *node_derived(tag, tid, "id", tcid, "TypeTagClassId", time_key="")[:2])
    add(("tagclass", "subclassOf", "tagclass"),
        *node_derived(tagclass, tcid, "id", tcid, "SubclassOfTagClassId", time_key="")[:2])

    # ---- link prediction split on knows (temporal, 1:1 negatives) ----------
    KNOWS = ("person", "knows", "person")
    ei = data[KNOWS].edge_index.numpy()
    times = data[KNOWS].edge_time.numpy()
    q60, q70, q80 = np.quantile(times[times > 0], [Q_MSG, Q_TRAIN, Q_VAL])
    linked = set(map(tuple, ei.T.tolist()))
    n_p = len(person)

    def negatives(k):
        out = []
        while len(out) < k:
            a = rng.integers(0, n_p, 2 * k); b = rng.integers(0, n_p, 2 * k)
            for x, y in zip(a.tolist(), b.tolist()):
                if x != y and (x, y) not in linked:
                    out.append((x, y))
                    if len(out) == k:
                        break
        return np.array(out, dtype=np.int64).T

    linkpred = {}
    for name, (lo, hi, cut) in {"train": (q60, q70, q60), "val": (q70, q80, q70),
                                "test": (q80, np.inf, q80)}.items():
        mask = (times >= lo) & (times < hi)
        pos = np.unique(ei[:, mask].T, axis=0).T
        neg = negatives(pos.shape[1])
        eli = np.concatenate([pos, neg], axis=1)
        lbl = np.concatenate([np.ones(pos.shape[1]), np.zeros(neg.shape[1])])
        perm = rng.permutation(eli.shape[1])
        linkpred[name] = {"edge_label_index": torch.from_numpy(eli[:, perm]),
                          "edge_label": torch.from_numpy(lbl[perm]).float(),
                          "mp_time_cut": float(cut)}

    # ---- node labelling: stratified 70/10/20 over persons by gender --------
    idx = rng.permutation(n_p)
    masks = {k: np.zeros(n_p, dtype=bool) for k in ("train", "val", "test")}
    for g in (0, 1):
        part = idx[country_label[idx] == g]
        a, b = int(0.7 * len(part)), int(0.8 * len(part))
        masks["train"][part[:a]] = True
        masks["val"][part[a:b]] = True
        masks["test"][part[b:]] = True
    nodelabel = {"y": torch.from_numpy(country_label),
                 "train_mask": torch.from_numpy(masks["train"]),
                 "val_mask": torch.from_numpy(masks["val"]),
                 "test_mask": torch.from_numpy(masks["test"])}

    # ---- write -------------------------------------------------------------
    torch.save(data, os.path.join(OUT, "snb_hetero.pt"))
    torch.save(linkpred, os.path.join(OUT, "split_linkpred.pt"))
    torch.save(nodelabel, os.path.join(OUT, "split_nodelabel.pt"))
    json.dump({"person": pid, "comment": cid, "post": poid, "forum": fid,
               "tag": tid, "tagclass": tcid, "organisation": oid, "place": plid},
              open(os.path.join(OUT, "id_maps.json"), "w"))
    meta = {
        "seed": SEED,
        "node_counts": {nt: int(data[nt].num_nodes) for nt in data.node_types},
        "edge_counts": {"__".join(et): int(data[et].edge_index.size(1))
                        for et in data.edge_types},
        "country_label": {"top_country_id": top_country,
                          "positive": int(country_label.sum()),
                          "total": int(country_label.size)},
        "browser_vocab": brow_vocab, "org_types": org_types, "place_types": place_types,
        "cuts_unix": {"q60": float(q60), "q70": float(q70), "q80": float(q80)},
        "linkpred_sizes": {k: int(v["edge_label"].numel()) for k, v in linkpred.items()},
    }
    json.dump(meta, open(os.path.join(OUT, "meta.json"), "w"), indent=2)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
