"""Candidate generation by key blocking (never all-pairs).

Every key is prefixed by the record's country string, so blocking happens
within a country (ground truth: 0 cross-country matches). Country is used
only as an opaque label, so unseen countries (France) work unchanged.

Key families (each with a cap on bucket size n_s1 * n_other):
  A  core-name token                 (rare tokens only, via cap)
  B  name-token skeleton + address number   (very selective, scales well)
  C  7-char skeleton prefix of the concatenated core name (domains, spacing)
  D  address number + address word   (name-independent rescue)
Pairs from all families are unioned; `hits` = number of distinct keys shared.
Then per S1 we keep the top `pre_k` by hits, score them with a cheap fuzzy
name/address similarity and keep the final top `top_k`.
"""
import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.process import cpdist

CONFIGS = {
    # normal: best recall
    "full": dict(cap_A=400, cap_B=2000, cap_C=400, cap_D=2000, cap_E=400, cap_F=1000,
                 n_nums=2, n_addr_tok=3, pre_k=60, top_k=25, families="ABCDEF"),
    # fast fallback: fewer keys, tighter caps, smaller top-k
    "fast": dict(cap_A=150, cap_B=800, cap_C=150, cap_D=800, cap_E=150, cap_F=300,
                 n_nums=1, n_addr_tok=2, pre_k=30, top_k=12, families="ABDEF"),
}


def _hash(keys):
    return pd.util.hash_array(np.asarray(keys, dtype=object)).astype(np.uint64)


def _explode(df, fn):
    """fn(row-tuple) -> list of key strings. Returns (hash array, row index array)."""
    keys, idx = [], []
    for i, ks in enumerate(fn(df)):
        if ks:
            keys.extend(ks)
            idx.extend([i] * len(ks))
    if not keys:
        return np.zeros(0, np.uint64), np.zeros(0, np.int32)
    return _hash(keys), np.asarray(idx, dtype=np.int32)


def _filter_keys(k1, i1, k2, i2, cap):
    """Keep only key rows whose bucket product n_s1 * n_other <= cap
    (counts are global, so chunked joins give identical results)."""
    a = pd.DataFrame({"k": k1, "i1": i1}).drop_duplicates()
    b = pd.DataFrame({"k": k2, "i2": i2}).drop_duplicates()
    c1 = a["k"].value_counts()
    c2 = b["k"].value_counts()
    common = c1.index.intersection(c2.index)
    prod = c1.loc[common].values.astype(np.int64) * c2.loc[common].values
    ok = common[prod <= cap]
    a = a[a["k"].isin(ok)].sort_values("i1", kind="stable").reset_index(drop=True)
    b = b[b["k"].isin(ok)].reset_index(drop=True)
    return a, b


# ---------------------------------------------------------------- key functions
def keys_A(df, cfg):
    for c, core in zip(df["country"].values, df["core"].values):
        yield [f"A{c}|{t}" for t in set(core.split()) if len(t) > 1]


def keys_B(df, cfg):
    n = cfg["n_nums"]
    for c, sk, nums in zip(df["country"].values, df["core_skel"].values, df["nums"].values):
        ns = nums.split()[:n]
        if not ns:
            yield []
            continue
        toks = {t for t in sk.split() if len(t) > 1}
        yield [f"B{c}|{t}|{x}" for t in toks for x in ns]


def keys_C(df, cfg):
    for c, s in zip(df["country"].values, df["core_cat_skel"].values):
        yield [f"C{c}|{s[:7]}", f"c{c}|{s[:12]}"] if len(s) >= 4 else []


def keys_E(df, cfg):
    """whole name skeleton, word order ignored (transpositions, translit.)"""
    for c, sk in zip(df["country"].values, df["core_skel"].values):
        toks = sorted(t for t in sk.split() if t)
        yield [f"E{c}|{' '.join(toks)}"] if toks else []


def keys_F(df, cfg):
    """name skeleton token + address word: number-free (numbers get truncated)"""
    m = cfg["n_addr_tok"]
    for c, sk, aa in zip(df["country"].values, df["core_skel"].values, df["addr_alpha"].values):
        toks = {t for t in sk.split() if len(t) > 1}
        ws = aa.split()[:m]
        yield [f"F{c}|{t}|{w}" for t in toks for w in ws]


def keys_D(df, cfg):
    n, m = cfg["n_nums"], cfg["n_addr_tok"]
    for c, nums, aa in zip(df["country"].values, df["nums"].values, df["addr_alpha"].values):
        ns = nums.split()[:n]
        ts = aa.split()[:m]
        yield [f"D{c}|{x}|{t}" for x in ns for t in ts]


KEYFN = {"A": keys_A, "B": keys_B, "C": keys_C, "D": keys_D, "E": keys_E, "F": keys_F}


def generate_candidates(s1, other, mode="full", verbose=True, chunk=150000):
    """Return DataFrame [i1, i2, hits] of candidate pairs (row indices).
    S1 is processed in chunks of `chunk` rows so peak memory stays bounded."""
    cfg = CONFIGS[mode]
    n1, n2 = len(s1), len(other)
    fams = []
    for fam in cfg["families"]:
        fn = KEYFN[fam]
        k1, i1 = _explode(s1, lambda d: fn(d, cfg))
        k2, i2 = _explode(other, lambda d: fn(d, cfg))
        a, b = _filter_keys(k1, i1, k2, i2, cfg[f"cap_{fam}"])
        del k1, i1, k2, i2
        fams.append((fam, a, b, a["i1"].values))
    out, n_union = [], 0
    raw = {f: 0 for f, *_ in fams}
    for lo in range(0, n1, chunk):
        hi = min(lo + chunk, n1)
        codes = []
        for fam, a, b, ai in fams:
            sl = slice(np.searchsorted(ai, lo), np.searchsorted(ai, hi))
            m = a.iloc[sl].merge(b, on="k")
            raw[fam] += len(m)
            codes.append(m["i1"].values.astype(np.int64) * n2 + m["i2"].values)
        allc = np.concatenate(codes) if codes else np.zeros(0, np.int64)
        del codes
        uniq, hits = np.unique(allc, return_counts=True)
        del allc
        n_union += len(uniq)
        cand = pd.DataFrame({"i1": (uniq // n2).astype(np.int64),
                             "i2": (uniq % n2).astype(np.int64),
                             "hits": np.minimum(hits, 32767).astype(np.int16)})
        del uniq, hits
        out.append(_prune(cand, s1, other, cfg))
        if verbose and n1 > chunk:
            print(f"  [block] S1 {hi:,}/{n1:,}", flush=True)
    cand = pd.concat(out, ignore_index=True) if out else pd.DataFrame(
        {"i1": np.zeros(0, np.int64), "i2": np.zeros(0, np.int64), "hits": np.zeros(0, np.int16)})
    if verbose:
        print("  [block] raw pairs per family: " + ", ".join(f"{f}={v:,}" for f, v in raw.items()))
        print(f"  [block] union: {n_union:,} pairs ({n_union / max(n1, 1):.1f} per S1)")
        print(f"  [block] final: {len(cand):,} pairs ({len(cand) / max(n1, 1):.1f} per S1)",
              flush=True)
    return cand


def _prune(cand, s1, other, cfg):
    """top pre_k by hits, then cheap fuzzy score -> top_k per S1."""
    cand = _topk(cand, "hits", cfg["pre_k"])
    a_n = s1["core"].values[cand["i1"].values]
    b_n = other["core"].values[cand["i2"].values]
    a_a = s1["addr_n"].values[cand["i1"].values]
    b_a = other["addr_n"].values[cand["i2"].values]
    cheap = (cpdist(a_n, b_n, scorer=fuzz.token_set_ratio, workers=-1).astype(np.float32)
             + 0.5 * cpdist(a_a, b_a, scorer=fuzz.token_set_ratio, workers=-1).astype(np.float32))
    cand["cheap"] = cheap + cand["hits"].values.astype(np.float32)
    cand = _topk(cand, "cheap", cfg["top_k"])
    return cand.drop(columns=["cheap"])


def _topk(df, col, k):
    df = df.sort_values(["i1", col], ascending=[True, False], kind="stable")
    r = df.groupby("i1", sort=False).cumcount().values
    return df[r < k]
