"""Exact challenge metric: macro-averaged F0.5 per Source 1 entity.

Singletons: empty prediction -> 1.0, any prediction -> 0.0.
Non-singletons with empty prediction -> 0.0.
"""
import argparse
from collections import defaultdict

import numpy as np


def f05_entity(pred, true, beta=0.5):
    pred, true = set(pred), set(true)
    if not true:
        return 1.0 if not pred else 0.0
    if not pred:
        return 0.0
    tp = len(pred & true)
    if tp == 0:
        return 0.0
    p, r = tp / len(pred), tp / len(true)
    b2 = beta * beta
    return (1 + b2) * p * r / (b2 * p + r)


def macro_f05(preds, gt, s1_ids=None):
    s1_ids = list(gt) if s1_ids is None else list(s1_ids)
    return float(np.mean([f05_entity(preds.get(s, ()), gt.get(s, ())) for s in s1_ids]))


def report(preds, gt, s1_ids, country_of=None, cands=None, title="eval"):
    """Print full diagnostics; return macro F0.5."""
    scores = {s: f05_entity(preds.get(s, ()), gt.get(s, ())) for s in s1_ids}
    single = [s for s in s1_ids if not gt.get(s)]
    multi = [s for s in s1_ids if gt.get(s)]
    tp = sum(len(set(preds.get(s, ())) & gt.get(s, set())) for s in s1_ids)
    npred = sum(len(set(preds.get(s, ()))) for s in s1_ids)
    ntrue = sum(len(gt.get(s, ())) for s in s1_ids)
    print(f"===== {title} =====")
    if cands is not None:
        ctp = sum(len(set(cands.get(s, ())) & gt.get(s, set())) for s in s1_ids)
        ncand = sum(len(cands.get(s, ())) for s in s1_ids)
        print(f"blocking recall      : {ctp / max(ntrue, 1):.5f}  ({ctp}/{ntrue})")
        print(f"avg candidates / S1  : {ncand / max(len(s1_ids), 1):.2f}")
    print(f"pair precision       : {tp / max(npred, 1):.5f}  ({tp}/{npred})")
    print(f"pair recall          : {tp / max(ntrue, 1):.5f}  ({tp}/{ntrue})")
    mf = float(np.mean(list(scores.values())))
    print(f"MACRO F0.5           : {mf:.5f}  (n={len(s1_ids)})")
    if single:
        print(f"  singletons         : {np.mean([scores[s] for s in single]):.5f}  (n={len(single)})")
    if multi:
        print(f"  non-singletons     : {np.mean([scores[s] for s in multi]):.5f}  (n={len(multi)})")
    if country_of is not None:
        byc = defaultdict(list)
        for s in s1_ids:
            byc[country_of.get(s, "")].append(scores[s])
        for c, v in sorted(byc.items()):
            print(f"  country={c!r:12s}: {np.mean(v):.5f}  (n={len(v)})")
    return mf


def main():
    from io_utils import read_ground_truth
    import pandas as pd
    ap = argparse.ArgumentParser(description="Score a matching_results.tsv against ground truth")
    ap.add_argument("--pred", required=True)
    ap.add_argument("--gt", required=True)
    a = ap.parse_args()
    gt = read_ground_truth(a.gt)
    df = pd.read_csv(a.pred, sep="\t", dtype=str, keep_default_na=False)
    preds = {s: [x for x in m.split(",") if x] for s, m in zip(df.iloc[:, 0], df.iloc[:, 1])}
    report(preds, gt, list(gt))


if __name__ == "__main__":
    main()
