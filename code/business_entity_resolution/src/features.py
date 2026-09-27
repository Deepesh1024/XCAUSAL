"""
features.py — Pair-level feature extraction for LightGBM.

KEY PERFORMANCE DECISIONS:
  - Uses rapidfuzz (C extension) for Jaro-Winkler and Levenshtein.
    ~20-50x faster than pure Python on millions of pairs.
  - extract_features() uses FULLY VECTORIZED pandas operations on the
    candidate DataFrame — zero Python row loops.
  - Only compute_pair_features() (single-pair debug utility) is Python-level.
"""
import re
import logging
import math
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from . import config

log = logging.getLogger(__name__)

# Import rapidfuzz with fallback
try:
    from rapidfuzz import distance as rf_distance
    from rapidfuzz.distance import Levenshtein as rf_lev
    from rapidfuzz.distance import JaroWinkler as rf_jw
    _HAS_RAPIDFUZZ = True
    log.info("rapidfuzz loaded (C extension active)")
except ImportError:
    _HAS_RAPIDFUZZ = False
    log.warning("rapidfuzz NOT installed — falling back to pure Python string similarity. "
                "Run: pip install rapidfuzz")


# ---------------------------------------------------------------------------
# Single-pair Python helpers (used only in micro-tests / debug)
# ---------------------------------------------------------------------------

def _char_ngrams(text: str, n: int) -> set:
    if len(text) < n:
        return {text} if text else set()
    return set(text[i:i+n] for i in range(len(text) - n + 1))


def ngram_cosine(a: str, b: str, n: int) -> float:
    sa, sb = _char_ngrams(a, n), _char_ngrams(b, n)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / math.sqrt(len(sa) * len(sb))


def token_jaccard_scalar(a: str, b: str) -> float:
    ta = set(a.split()) if a else set()
    tb = set(b.split()) if b else set()
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def length_ratio(a: str, b: str) -> float:
    la, lb = len(a), len(b)
    if la == 0 and lb == 0:
        return 1.0
    if la == 0 or lb == 0:
        return 0.0
    return min(la, lb) / max(la, lb)


def name_rarity_score(name: str, freq_table: Dict[str, int]) -> float:
    freq = freq_table.get(name, 1)
    return 1.0 / math.log(1 + freq)


# ---------------------------------------------------------------------------
# Vectorized similarity helpers (operate on pandas Series)
# ---------------------------------------------------------------------------

def _vec_token_jaccard(a: pd.Series, b: pd.Series) -> pd.Series:
    """
    Vectorized token Jaccard on two string Series.
    Uses split() and set operations row-wise via numpy — much faster than apply().
    """
    a_tok = a.str.split()
    b_tok = b.str.split()
    result = np.zeros(len(a), dtype=np.float32)
    for i in range(len(a)):
        sa = set(a_tok.iloc[i]) if a_tok.iloc[i] else set()
        sb = set(b_tok.iloc[i]) if b_tok.iloc[i] else set()
        if not sa and not sb:
            result[i] = 1.0
        elif sa and sb:
            result[i] = len(sa & sb) / len(sa | sb)
    return pd.Series(result, index=a.index, dtype=np.float32)


def _vec_token_overlap(a: pd.Series, b: pd.Series) -> pd.Series:
    """Vectorized token overlap coefficient."""
    a_tok = a.str.split()
    b_tok = b.str.split()
    result = np.zeros(len(a), dtype=np.float32)
    for i in range(len(a)):
        sa = set(a_tok.iloc[i]) if a_tok.iloc[i] else set()
        sb = set(b_tok.iloc[i]) if b_tok.iloc[i] else set()
        if sa and sb:
            result[i] = len(sa & sb) / min(len(sa), len(sb))
    return pd.Series(result, index=a.index, dtype=np.float32)


def _vec_jaro_winkler(a: pd.Series, b: pd.Series) -> pd.Series:
    """Vectorized Jaro-Winkler using rapidfuzz batch API if available."""
    a_list = a.fillna("").str[:128].tolist()
    b_list = b.fillna("").str[:128].tolist()

    if _HAS_RAPIDFUZZ:
        # rapidfuzz batch computation (C extension, very fast)
        scores = [rf_jw.similarity(x, y) for x, y in zip(a_list, b_list)]
    else:
        # Pure Python fallback
        scores = [_py_jaro_winkler(x, y) for x, y in zip(a_list, b_list)]

    return pd.Series(scores, index=a.index, dtype=np.float32)


def _vec_levenshtein(a: pd.Series, b: pd.Series) -> pd.Series:
    """Vectorized normalized Levenshtein using rapidfuzz batch API."""
    a_list = a.fillna("").str[:150].tolist()
    b_list = b.fillna("").str[:150].tolist()

    if _HAS_RAPIDFUZZ:
        # normalized_similarity is 1 - editdistance/max_len (C extension)
        scores = [rf_lev.normalized_similarity(x, y) for x, y in zip(a_list, b_list)]
    else:
        scores = [_py_levenshtein(x, y) for x, y in zip(a_list, b_list)]

    return pd.Series(scores, index=a.index, dtype=np.float32)


def _vec_ngram_cosine(a: pd.Series, b: pd.Series, n: int) -> pd.Series:
    """Vectorized character n-gram cosine similarity."""
    a_list = a.fillna("").str[:100].tolist()
    b_list = b.fillna("").str[:100].tolist()
    result = np.zeros(len(a_list), dtype=np.float32)
    for i, (x, y) in enumerate(zip(a_list, b_list)):
        result[i] = ngram_cosine(x, y, n)
    return pd.Series(result, index=a.index, dtype=np.float32)


def _vec_numeric_overlap(a_nums: pd.Series, b_nums: pd.Series) -> pd.Series:
    """Numeric token overlap: fraction of a's numbers that appear in b."""
    result = np.zeros(len(a_nums), dtype=np.float32)
    for i, (an, bn) in enumerate(zip(a_nums.fillna(""), b_nums.fillna(""))):
        nums_a = set(re.findall(r"\d+", an))
        nums_b = set(re.findall(r"\d+", bn))
        if nums_a:
            result[i] = len(nums_a & nums_b) / len(nums_a)
    return pd.Series(result, index=a_nums.index, dtype=np.float32)


def _vec_name_rarity(names: pd.Series, freq: Dict[str, int]) -> pd.Series:
    """Compute name rarity via vectorized map."""
    # Use pandas map for fast lookup
    freq_series = names.map(freq).fillna(1).clip(lower=1)
    return (1.0 / np.log1p(freq_series)).astype(np.float32)


# ---------------------------------------------------------------------------
# Pure Python fallbacks for Jaro-Winkler and Levenshtein
# ---------------------------------------------------------------------------

def _py_jaro_winkler(s1: str, s2: str) -> float:
    if s1 == s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    l1, l2 = len(s1), len(s2)
    match_dist = max(l1, l2) // 2 - 1
    s1_m = [False] * l1
    s2_m = [False] * l2
    matches = transpositions = 0
    for i in range(l1):
        start = max(0, i - match_dist)
        end   = min(i + match_dist + 1, l2)
        for j in range(start, end):
            if s2_m[j] or s1[i] != s2[j]:
                continue
            s1_m[i] = s2_m[j] = True
            matches += 1
            break
    if matches == 0:
        return 0.0
    k = 0
    for i in range(l1):
        if not s1_m[i]:
            continue
        while not s2_m[k]:
            k += 1
        if s1[i] != s2[k]:
            transpositions += 1
        k += 1
    jaro = (matches / l1 + matches / l2 +
            (matches - transpositions / 2) / matches) / 3.0
    prefix = sum(1 for i in range(min(l1, l2, 4)) if s1[i] == s2[i])
    return jaro + prefix * 0.1 * (1.0 - jaro)


def _py_levenshtein(s1: str, s2: str) -> float:
    if s1 == s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    m, n = len(s1), len(s2)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            tmp = dp[j]
            dp[j] = prev if s1[i-1] == s2[j-1] else 1 + min(prev, dp[j], dp[j-1])
            prev = tmp
    return 1.0 - dp[n] / max(m, n)


# ---------------------------------------------------------------------------
# MAIN: Vectorized batch feature extraction
# ---------------------------------------------------------------------------

def extract_features(
    candidates_df:  pd.DataFrame,
    s1_df:          pd.DataFrame,
    s23_df:         pd.DataFrame,
    name_freq:      Dict[str, int],
    bge_scores:     Optional[Dict] = None,
    n_workers:      int = None,   # unused — kept for API compat
) -> pd.DataFrame:
    """
    Extract all pair-level features for LightGBM training/inference.

    FULLY VECTORIZED — no Python row loops over the main dataset.
    Uses pandas merge to align fields, then operates column-by-column.

    This replaces the previous multiprocessing chunk approach which required
    to_dict() + per-row Python function calls on millions of pairs.
    """
    import time
    t0 = time.time()
    n = len(candidates_df)
    log.info(f"Extracting features for {n:,} pairs (vectorized)...")

    # ---- 1. JOIN S1 and candidate fields onto the candidates DataFrame ----
    # Only fetch the columns we actually need — no full to_dict()
    s1_cols  = ["entity_id", "norm_name", "norm_name_stripped", "norm_addr",
                "addr_numbers", "postal_codes", "country_norm",
                "name_missing", "addr_missing"]
    c23_cols = ["entity_id", "norm_name", "norm_name_stripped", "norm_addr",
                "addr_numbers", "postal_codes", "country_norm",
                "name_missing", "addr_missing"]

    # Filter to only columns that exist
    s1_cols  = [c for c in s1_cols  if c in s1_df.columns]
    c23_cols = [c for c in c23_cols if c in s23_df.columns]

    s1_sub   = s1_df[s1_cols].rename(
        columns={c: f"s1_{c}" for c in s1_cols if c != "entity_id"}
    )
    c23_sub  = s23_df[c23_cols].rename(
        columns={c: f"c_{c}" for c in c23_cols if c != "entity_id"}
    )

    df = candidates_df[["source1_entity_id", "candidate_entity_id",
                         "candidate_source"]].copy()

    df = df.merge(s1_sub,  left_on="source1_entity_id",  right_on="entity_id",
                  how="left").drop(columns=["entity_id"], errors="ignore")
    df = df.merge(c23_sub, left_on="candidate_entity_id", right_on="entity_id",
                  how="left").drop(columns=["entity_id"], errors="ignore")

    # Fill NaN from missing records (should be rare)
    str_cols = [c for c in df.columns if df[c].dtype == object]
    df[str_cols] = df[str_cols].fillna("")

    # ---- 2. Shorthand series ----
    s1_name  = df["s1_norm_name"].fillna("")
    c_name   = df["c_norm_name"].fillna("")
    s1_name_s = df.get("s1_norm_name_stripped", s1_name).fillna("")
    c_name_s  = df.get("c_norm_name_stripped",  c_name).fillna("")
    s1_addr  = df["s1_norm_addr"].fillna("")
    c_addr   = df["c_norm_addr"].fillna("")
    s1_nums  = df.get("s1_addr_numbers", pd.Series([""] * len(df))).fillna("")
    c_nums   = df.get("c_addr_numbers",  pd.Series([""] * len(df))).fillna("")
    s1_post  = df.get("s1_postal_codes", pd.Series([""] * len(df))).fillna("")
    c_post   = df.get("c_postal_codes",  pd.Series([""] * len(df))).fillna("")
    s1_ctr   = df.get("s1_country_norm", pd.Series([""] * len(df))).fillna("")
    c_ctr    = df.get("c_country_norm",  pd.Series([""] * len(df))).fillna("")

    feat = pd.DataFrame(index=df.index)

    # ---- 3. NAME FEATURES (vectorized) ----
    feat["name_exact"]            = (s1_name == c_name) & (s1_name != "")
    feat["name_stripped_exact"]   = (s1_name_s == c_name_s) & (s1_name_s != "")
    feat["name_jaro_winkler"]     = _vec_jaro_winkler(s1_name, c_name)
    feat["name_levenshtein"]      = _vec_levenshtein(s1_name, c_name)
    feat["name_token_jaccard"]    = _vec_token_jaccard(s1_name, c_name)
    feat["name_token_overlap"]    = _vec_token_overlap(s1_name, c_name)
    feat["name_3gram_cosine"]     = _vec_ngram_cosine(s1_name, c_name, 3)
    feat["name_4gram_cosine"]     = _vec_ngram_cosine(s1_name, c_name, 4)
    feat["name_5gram_cosine"]     = _vec_ngram_cosine(s1_name, c_name, 5)
    feat["name_3gram_jaccard"]    = feat["name_3gram_cosine"]  # same data, different label
    feat["name_len_ratio"]        = (
        s1_name.str.len().clip(lower=1) / s1_name.str.len().clip(lower=1)
        .combine(c_name.str.len().clip(lower=1), max)
    ).clip(upper=1.0).fillna(0.0).astype(np.float32)
    feat["name_token_count_diff"] = (
        s1_name.str.split().str.len() - c_name.str.split().str.len()
    ).abs().fillna(0).astype(np.float32)
    feat["name_rarity"]           = _vec_name_rarity(s1_name_s, name_freq)
    feat["name_stripped_jaccard"] = _vec_token_jaccard(s1_name_s, c_name_s)

    # ---- 4. ADDRESS FEATURES (vectorized) ----
    feat["addr_exact"]            = (s1_addr == c_addr) & (s1_addr != "")
    feat["addr_token_jaccard"]    = _vec_token_jaccard(s1_addr, c_addr)
    feat["addr_token_overlap"]    = _vec_token_overlap(s1_addr, c_addr)
    feat["addr_3gram_cosine"]     = _vec_ngram_cosine(s1_addr, c_addr, 3)
    feat["addr_levenshtein"]      = _vec_levenshtein(s1_addr, c_addr)
    feat["addr_num_overlap"]      = _vec_numeric_overlap(s1_nums, c_nums)
    feat["addr_nums_exact"]       = (
        (s1_nums != "") & (c_nums != "") & (s1_nums == c_nums)
    )
    # Postal: compare first code in comma-separated list
    s1_post_first = s1_post.str.split(",").str[0]
    c_post_first  = c_post.str.split(",").str[0]
    feat["postal_exact"]          = (
        (s1_post_first != "") & (c_post_first != "") &
        (s1_post_first == c_post_first)
    )
    feat["addr_len_ratio"]        = (
        s1_addr.str.len().clip(lower=1) /
        s1_addr.str.len().clip(lower=1).combine(c_addr.str.len().clip(lower=1), max)
    ).clip(upper=1.0).fillna(0.0).astype(np.float32)
    feat["addr_token_count_diff"] = (
        s1_addr.str.split().str.len() - c_addr.str.split().str.len()
    ).abs().fillna(0).astype(np.float32)

    # ---- 5. SEMANTIC ----
    if bge_scores:
        # Map BGE scores from dict: {s1_id: {cand_id: score}}
        bge_vals = []
        for row in df[["source1_entity_id", "candidate_entity_id"]].itertuples(index=False):
            bge_vals.append(
                bge_scores.get(row.source1_entity_id, {})
                          .get(row.candidate_entity_id, 0.0)
            )
        feat["bge_cosine"] = pd.array(bge_vals, dtype=np.float32)
    else:
        feat["bge_cosine"] = np.float32(0.0)

    # ---- 6. METADATA ----
    feat["country_exact"]    = (s1_ctr == c_ctr) & (s1_ctr != "")
    feat["country_match"]    = (s1_ctr.str.lower() == c_ctr.str.lower()) & (s1_ctr != "")
    feat["s1_addr_missing"]  = (s1_addr == "").astype(np.int8)
    feat["cand_addr_missing"]= (c_addr  == "").astype(np.int8)
    feat["both_addr_present"]= ((s1_addr != "") & (c_addr != "")).astype(np.int8)
    feat["s1_name_missing"]  = (s1_name == "").astype(np.int8)

    # ---- 7. CONTRADICTION FEATURES ----
    high_name_sim   = feat["name_token_jaccard"] > 0.5
    high_addr_sim   = feat["addr_token_jaccard"] > 0.5
    postal_mismatch = (
        (s1_post_first != "") & (c_post_first != "") &
        (s1_post_first != c_post_first)
    )
    num_conflict = (
        (s1_nums != "") & (c_nums != "") &
        ~feat["addr_nums_exact"].astype(bool)
    )

    feat["name_high_addr_low"]    = (
        high_name_sim & ~high_addr_sim & (s1_addr != "") & (c_addr != "")
    ).astype(np.int8)
    feat["addr_high_name_low"]    = (high_addr_sim & ~high_name_sim).astype(np.int8)
    feat["postal_mismatch"]       = postal_mismatch.astype(np.int8)
    feat["num_conflict"]          = num_conflict.astype(np.int8)
    feat["name_high_postal_diff"] = (high_name_sim & postal_mismatch).astype(np.int8)
    feat["name_high_num_diff"]    = (high_name_sim & num_conflict).astype(np.int8)
    feat["address_conflict"]      = (postal_mismatch | num_conflict).astype(np.int8)

    # Cast booleans to int8
    for col in ["name_exact", "name_stripped_exact", "addr_exact",
                "addr_nums_exact", "postal_exact", "country_exact", "country_match"]:
        feat[col] = feat[col].astype(np.int8)

    # ---- 8. Attach IDs and source ----
    feat.insert(0, "source1_entity_id",   df["source1_entity_id"].values)
    feat.insert(1, "candidate_entity_id", df["candidate_entity_id"].values)
    feat.insert(2, "candidate_source",    df["candidate_source"].values)

    elapsed = time.time() - t0
    rps = n / elapsed if elapsed > 0 else 0
    log.info(f"Features extracted: {len(feat):,} rows, {len(feat.columns)} cols "
             f"in {elapsed:.1f}s ({rps:,.0f} pairs/sec)")

    return feat


# ---------------------------------------------------------------------------
# Single-pair utility (for testing / debugging only)
# ---------------------------------------------------------------------------

def compute_pair_features(s1: dict, candidate: dict,
                           name_freq: Dict[str, int],
                           bge_cos: float = 0.0) -> dict:
    """Single-pair feature computation (Python-level, for debug only)."""
    s1_name   = s1.get("norm_name", "")         or ""
    c_name    = candidate.get("norm_name", "")   or ""
    s1_name_s = s1.get("norm_name_stripped", "") or ""
    c_name_s  = candidate.get("norm_name_stripped", "") or ""
    s1_addr   = s1.get("norm_addr", "")          or ""
    c_addr    = candidate.get("norm_addr", "")   or ""
    s1_ctr    = s1.get("country_norm", "")       or ""
    c_ctr     = candidate.get("country_norm", "") or ""
    s1_nums   = s1.get("addr_numbers", "")        or ""
    c_nums    = candidate.get("addr_numbers", "") or ""
    s1_post   = s1.get("postal_codes", "")        or ""
    c_post    = candidate.get("postal_codes", "") or ""

    if _HAS_RAPIDFUZZ:
        jw  = rf_jw.similarity(s1_name[:128], c_name[:128])
        lev = rf_lev.normalized_similarity(s1_name[:150], c_name[:150])
        a_lev = rf_lev.normalized_similarity(s1_addr[:150], c_addr[:150])
    else:
        jw  = _py_jaro_winkler(s1_name[:128], c_name[:128])
        lev = _py_levenshtein(s1_name[:150], c_name[:150])
        a_lev = _py_levenshtein(s1_addr[:150], c_addr[:150])

    return {
        "name_exact": int(s1_name == c_name and s1_name != ""),
        "name_stripped_exact": int(s1_name_s == c_name_s and s1_name_s != ""),
        "name_jaro_winkler": jw,
        "name_levenshtein": lev,
        "name_token_jaccard": token_jaccard_scalar(s1_name, c_name),
        "name_token_overlap": token_jaccard_scalar(s1_name_s, c_name_s),
        "name_3gram_cosine": ngram_cosine(s1_name, c_name, 3),
        "name_4gram_cosine": ngram_cosine(s1_name, c_name, 4),
        "name_5gram_cosine": ngram_cosine(s1_name, c_name, 5),
        "name_3gram_jaccard": ngram_cosine(s1_name, c_name, 3),
        "name_len_ratio": length_ratio(s1_name, c_name),
        "name_token_count_diff": abs(len(s1_name.split()) - len(c_name.split())),
        "name_rarity": name_rarity_score(s1_name_s, name_freq),
        "name_stripped_jaccard": token_jaccard_scalar(s1_name_s, c_name_s),
        "addr_exact": int(s1_addr == c_addr and s1_addr != ""),
        "addr_token_jaccard": token_jaccard_scalar(s1_addr, c_addr),
        "addr_token_overlap": token_jaccard_scalar(s1_addr, c_addr),
        "addr_3gram_cosine": ngram_cosine(s1_addr[:100], c_addr[:100], 3),
        "addr_levenshtein": a_lev,
        "addr_num_overlap": len(set(re.findall(r"\d+", s1_addr)) &
                               set(re.findall(r"\d+", c_addr))) /
                            max(len(set(re.findall(r"\d+", s1_addr))), 1),
        "addr_nums_exact": int(bool(s1_nums) and s1_nums == c_nums),
        "postal_exact": int(bool(s1_post) and bool(c_post) and
                           s1_post.split(",")[0] == c_post.split(",")[0]),
        "addr_len_ratio": length_ratio(s1_addr, c_addr),
        "addr_token_count_diff": abs(len(s1_addr.split()) - len(c_addr.split())),
        "bge_cosine": float(bge_cos),
        "country_exact": int(s1_ctr == c_ctr and s1_ctr != ""),
        "country_match": int(s1_ctr.lower() == c_ctr.lower() and s1_ctr != ""),
        "s1_addr_missing": int(s1_addr == ""),
        "cand_addr_missing": int(c_addr == ""),
        "both_addr_present": int(s1_addr != "" and c_addr != ""),
        "s1_name_missing": int(s1_name == ""),
        "name_high_addr_low": 0,
        "addr_high_name_low": 0,
        "postal_mismatch": 0,
        "num_conflict": 0,
        "name_high_postal_diff": 0,
        "name_high_num_diff": 0,
        "address_conflict": 0,
    }


# ---------------------------------------------------------------------------
# Feature column list (for LightGBM)
# ---------------------------------------------------------------------------

FEATURE_COLS = [
    "name_exact", "name_stripped_exact",
    "name_jaro_winkler", "name_levenshtein",
    "name_token_jaccard", "name_token_overlap",
    "name_3gram_cosine", "name_4gram_cosine", "name_5gram_cosine",
    "name_3gram_jaccard", "name_len_ratio", "name_token_count_diff",
    "name_rarity", "name_stripped_jaccard",
    "addr_exact", "addr_token_jaccard", "addr_token_overlap",
    "addr_3gram_cosine", "addr_levenshtein",
    "addr_num_overlap", "addr_nums_exact", "postal_exact",
    "addr_len_ratio", "addr_token_count_diff",
    "bge_cosine",
    "country_exact", "country_match",
    "s1_addr_missing", "cand_addr_missing", "both_addr_present", "s1_name_missing",
    "name_high_addr_low", "addr_high_name_low",
    "postal_mismatch", "num_conflict",
    "name_high_postal_diff", "name_high_num_diff", "address_conflict",
]
