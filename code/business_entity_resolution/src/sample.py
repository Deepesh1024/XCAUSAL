"""Build a development sample from the training data.

Picks N Source 1 entities whose ground-truth matches are ALL present in the
loaded S2/S3 files (works even on partially downloaded files), stratified so
the distribution of match counts follows the full ground truth. Adds random
other S2/S3 records as distractors (by default ~26% of the sampled S2/S3
records, the unmatched share observed in the full training data).
"""
from collections import Counter

import numpy as np
import pandas as pd


def build_sample(s1, other, gt, n, seed=0, distractor_frac=0.26):
    rng = np.random.default_rng(seed)
    have_other = set(other["entity_id"].values)
    s1_ids = set(s1["entity_id"].values)
    # target distribution of match counts (from the full GT we have)
    dist = Counter(min(len(v), 8) for v in gt.values())
    tot = sum(dist.values())
    eligible = {}
    for k, v in gt.items():
        if k in s1_ids and all(m in have_other for m in v):
            eligible.setdefault(min(len(v), 8), []).append(k)
    chosen = []
    for b, c in dist.items():
        pool = eligible.get(b, [])
        want = int(round(n * c / tot))
        take = min(want, len(pool))
        if take < want:
            print(f"  [sample] bin {b}: only {len(pool)} eligible, wanted {want}")
        chosen.extend(rng.choice(pool, size=take, replace=False).tolist() if take else [])
    chosen_set = set(chosen)
    matched = {m for k in chosen for m in gt[k]}
    n_dis = int(len(matched) * distractor_frac / (1 - distractor_frac))
    # distractors: loaded S2/S3 records not matched to any chosen S1
    cand = other["entity_id"].values
    mask = ~other["entity_id"].isin(matched).values
    dis = rng.choice(cand[mask], size=min(n_dis, mask.sum()), replace=False)
    keep = matched | set(dis.tolist())
    s1_s = s1[s1["entity_id"].isin(chosen_set)].reset_index(drop=True)
    o_s = other[other["entity_id"].isin(keep)].reset_index(drop=True)
    gt_s = {k: gt[k] for k in chosen}
    sizes = Counter(len(v) for v in gt_s.values())
    print(f"  [sample] S1={len(s1_s):,}  S2/S3={len(o_s):,} (true={len(matched):,}, "
          f"distractors={len(dis):,})  singleton rate={sizes[0] / max(len(gt_s), 1):.2%}")
    return s1_s, o_s, gt_s
