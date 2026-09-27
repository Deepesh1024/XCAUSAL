"""Quick dataset profile: sizes, singleton rate, match-count distribution,
S2/S3 split, missing fields, countries, Unicode scripts, 1-to-many check.

Usage: python src/inspect_data.py --data-dir ../../dataset
"""
import argparse
import os
import unicodedata
from collections import Counter

from io_utils import read_ground_truth, read_source


def script_of(text):
    """Dominant non-Latin Unicode script in text, else 'Latin' (or 'empty')."""
    c = Counter()
    for ch in text:
        if ch.isalpha():
            try:
                c[unicodedata.name(ch).split()[0]] += 1
            except ValueError:
                pass
    if not c:
        return "empty/non-alpha"
    non_latin = {k: v for k, v in c.items() if k != "LATIN"}
    return max(non_latin, key=non_latin.get) if non_latin else "LATIN"


def profile_source(name, df):
    n = len(df)
    print(f"\n--- {name}: {n:,} rows")
    for col in ("business_name", "business_address", "country"):
        miss = (df[col].str.strip() == "").mean()
        print(f"  missing {col:17s}: {miss:.2%}")
    print("  countries:", dict(Counter(df["country"]).most_common(8)))
    samp = df.sample(min(n, 50000), random_state=0)
    for col in ("business_name", "business_address"):
        sc = Counter(script_of(t) for t in samp[col])
        tot = sum(sc.values())
        print(f"  scripts in {col}: " + ", ".join(f"{k}={v / tot:.1%}" for k, v in sc.most_common(8)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="../../dataset")
    ap.add_argument("--splits", default="train,test")
    a = ap.parse_args()

    for split in a.splits.split(","):
        for i in (1, 2, 3):
            p = os.path.join(a.data_dir, split, f"{split}_source{i}.tsv")
            if os.path.exists(p):
                profile_source(f"{split}_source{i}", read_source(p))

    gp = os.path.join(a.data_dir, "train", "train_ground_truth.tsv")
    if not os.path.exists(gp):
        return
    gt = read_ground_truth(gp)
    n = len(gt)
    sizes = Counter(len(v) for v in gt.values())
    print(f"\n--- ground truth: {n:,} S1 entities")
    print(f"  singleton rate: {sizes[0] / n:.2%}")
    print("  matches per S1:", {k: f"{v / n:.1%}" for k, v in sorted(sizes.items())})
    all_m = [m for v in gt.values() for m in v]
    src = Counter(m[:2] for m in all_m)
    print(f"  total matched ids: {len(all_m):,}  split: {dict(src)}")
    per_src = Counter((sum(m.startswith("S2") for m in v), sum(m.startswith("S3") for m in v))
                      for v in gt.values() if v)
    print("  (nS2, nS3) per non-singleton, top:", per_src.most_common(10))
    cnt = Counter(all_m)
    multi = sum(1 for c in cnt.values() if c > 1)
    print(f"  S2/S3 ids matched to >1 S1: {multi:,} of {len(cnt):,} "
          f"({multi / max(len(cnt), 1):.4%})")


if __name__ == "__main__":
    main()
