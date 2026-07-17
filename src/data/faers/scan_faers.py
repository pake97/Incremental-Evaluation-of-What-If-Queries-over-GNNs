"""Single streaming pass over patient_safety.csv -> aggregated Drug-AE graph.

Collapses ~9.3M adverse-event reports into:
  * drug -causes-> AE       co-report counts (+ first-seen date for temporal split)
  * drug -indicated-> term  indication counts
  * per-drug / per-AE seriousness (n_reports, n_serious)  -> 'serious-prone' label
  * vocabularies of drugs / AEs / indications

Reports with empty drugs OR empty SE are skipped (the first ~5-10% of the file
has no drug data).  Codes are parsed with regex (fast):
  drugs -> DrugBank IDs (DB\\d+);  AEs / indications -> MedDRA codes (\\d{7,8}).

Saves processed/aggregates.pkl and prints stats.
"""
from __future__ import annotations
import csv, os, pickle, re, time
from collections import Counter, defaultdict

csv.field_size_limit(10**9)
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
SRC = os.path.join(ROOT, "dataset", "faers", "CSV", "patient_safety.csv")
OUT = os.path.join(ROOT, "dataset", "faers", "processed")
os.makedirs(OUT, exist_ok=True)

RE_DB = re.compile(r"DB\d+")
RE_MED = re.compile(r"\d{7,8}")


def to_yyyymmdd(s):
    # 'YYYY-MM-DD' -> int YYYYMMDD ; cheap, orderable
    if len(s) >= 10 and s[4] == "-":
        return int(s[:4] + s[5:7] + s[8:10])
    return 0


def main():
    da_count = Counter()                 # (drug, ae) -> co-reports
    da_date = {}                         # (drug, ae) -> min yyyymmdd
    di_count = Counter()                 # (drug, indication) -> count
    drug_n, drug_ser = Counter(), Counter()
    ae_n, ae_ser = Counter(), Counter()
    inds = set()
    n_total = n_used = 0

    t0 = time.time()
    with open(SRC) as fh:
        r = csv.reader(fh)
        cols = next(r); idx = {c: i for i, c in enumerate(cols)}
        di, si, sei, ii, dti = (idx["drugs"], idx["serious"], idx["SE"],
                                idx["indications"], idx["date"])
        for row in r:
            n_total += 1
            if n_total % 1_000_000 == 0:
                print(f"  {n_total:,} rows ({time.time()-t0:.0f}s)  used={n_used:,}  pairs={len(da_count):,}")
            drugs = RE_DB.findall(row[di])
            if not drugs:
                continue
            aes = RE_MED.findall(row[sei])
            if not aes:
                continue
            n_used += 1
            ser = 1 if row[si] == "1" else 0
            date = to_yyyymmdd(row[dti])
            indications = RE_MED.findall(row[ii])
            ds = set(drugs); aeset = set(aes)
            for d in ds:
                drug_n[d] += 1; drug_ser[d] += ser
            for a in aeset:
                ae_n[a] += 1; ae_ser[a] += ser
            for d in ds:
                for a in aeset:
                    key = (d, a)
                    da_count[key] += 1
                    if date and (key not in da_date or date < da_date[key]):
                        da_date[key] = date
                for ind in indications:
                    di_count[(d, ind)] += 1
                    inds.add(ind)

    agg = dict(da_count=dict(da_count), da_date=da_date, di_count=dict(di_count),
               drug_n=dict(drug_n), drug_ser=dict(drug_ser),
               ae_n=dict(ae_n), ae_ser=dict(ae_ser), inds=inds,
               n_total=n_total, n_used=n_used)
    with open(os.path.join(OUT, "aggregates.pkl"), "wb") as f:
        pickle.dump(agg, f, protocol=4)

    print(f"\n=== scan done in {time.time()-t0:.0f}s ===")
    print(f"reports total={n_total:,}  used(drugs&AE)={n_used:,} ({n_used/n_total:.1%})")
    print(f"distinct drugs={len(drug_n):,}  AEs={len(ae_n):,}  indications={len(inds):,}")
    print(f"drug-AE pairs={len(da_count):,}  drug-indication pairs={len(di_count):,}")
    import numpy as np
    cc = np.array(list(da_count.values()))
    print(f"co-report count: median={np.median(cc):.0f} mean={cc.mean():.1f} max={cc.max()} "
          f">=5:{(cc>=5).sum():,} >=10:{(cc>=10).sum():,}")
    # serious-prone label preview (support>=5)
    for name, n, ser in [("AE", ae_n, ae_ser), ("drug", drug_n, drug_ser)]:
        sup = {k: v for k, v in n.items() if v >= 5}
        frac = {k: ser[k] / n[k] for k in sup}
        pos = sum(1 for v in frac.values() if v > 0.5)
        print(f"serious-prone {name} (support>=5): {pos:,}/{len(sup):,} = {pos/len(sup):.1%} positive")


if __name__ == "__main__":
    main()
