"""
features.py — Pair-level feature extraction for LightGBM.
All features computed on CPU (per pair).
"""
import re
import logging
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from multiprocessing import Pool
from functools import partial

from . import config
from .normalize import tokenize, get_tokens

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# String similarity helpers
# ---------------------------------------------------------------------------

def _char_ngrams(text: str, n: int) -> set:
    if len(text) < n:
        return {text} if text else set()
    return set(text[i:i+n] for i in range(len(text) - n + 1))


def ngram_jaccard(a: str, b: str, n: int) -> float:
    sa, sb = _char_ngrams(a, n), _char_ngrams(b, n)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def ngram_cosine(a: str, b: str, n: int) -> float:
    """Approximate cosine via n-gram overlap coefficient."""
    sa, sb = _char_ngrams(a, n), _char_ngrams(b, n)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / math.sqrt(len(sa) * len(sb))


def token_jaccard(a: str, b: str) -> float:
    ta = set(a.split()) if a else set()
    tb = set(b.split()) if b else set()
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def token_overlap_coeff(a: str, b: str) -> float:
    ta = set(a.split()) if a else set()
    tb = set(b.split()) if b else set()
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / min(len(ta), len(tb))


def jaro_winkler(s1: str, s2: str) -> float:
    """Jaro-Winkler distance."""
    if s1 == s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    l1, l2 = len(s1), len(s2)
    match_dist = max(l1, l2) // 2 - 1

    s1_matches = [False] * l1
    s2_matches = [False] * l2
    matches = transpositions = 0

    for i in range(l1):
        start = max(0, i - match_dist)
        end   = min(i + match_dist + 1, l2)
        for j in range(start, end):
            if s2_matches[j] or s1[i] != s2[j]:
                continue
            s1_matches[i] = s2_matches[j] = True
            matches += 1
            break

    if matches == 0:
        return 0.0

    k = 0
    for i in range(l1):
        if not s1_matches[i]:
            continue
        while not s2_matches[k]:
            k += 1
        if s1[i] != s2[k]:
            transpositions += 1
        k += 1

    jaro = (matches / l1 + matches / l2 + (matches - transpositions / 2) / matches) / 3.0

    # Winkler prefix
    prefix = 0
    for i in range(min(l1, l2, 4)):
        if s1[i] == s2[i]:
            prefix += 1
        else:
            break

    return jaro + prefix * 0.1 * (1.0 - jaro)


def normalized_levenshtein(s1: str, s2: str) -> float:
    """Normalized edit distance (1 - editdist / max_len)."""
    if s1 == s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    m, n = len(s1), len(s2)
    # Limit to reasonable length
    if m > 200 or n > 200:
        s1, s2 = s1[:200], s2[:200]
        m, n   = len(s1), len(s2)

    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            tmp     = dp[j]
            dp[j]   = prev if s1[i-1] == s2[j-1] else 1 + min(prev, dp[j], dp[j-1])
            prev    = tmp

    return 1.0 - dp[n] / max(m, n)


def length_ratio(a: str, b: str) -> float:
    la, lb = len(a), len(b)
    if la == 0 and lb == 0:
        return 1.0
    if la == 0 or lb == 0:
        return 0.0
    return min(la, lb) / max(la, lb)


def numeric_overlap(a: str, b: str) -> float:
    """Fraction of numeric tokens in a that also appear in b."""
    nums_a = set(re.findall(r"\d+", a))
    nums_b = set(re.findall(r"\d+", b))
    if not nums_a:
        return 0.0
    return len(nums_a & nums_b) / len(nums_a)


def name_rarity_score(name: str, freq_table: Dict[str, int]) -> float:
    freq = freq_table.get(name, 1)
    return 1.0 / math.log(1 + freq)


# ---------------------------------------------------------------------------
# Per-pair feature vector
# ---------------------------------------------------------------------------

def compute_pair_features(
    s1:          dict,
    candidate:   dict,
    name_freq:   Dict[str, int],
    bge_cos:     float = 0.0,
) -> dict:
    """
    Compute ~35 features for a single (S1, candidate) pair.
    """
    # ---- Name fields ----
    s1_name   = s1.get("norm_name", "")          or ""
    c_name    = candidate.get("norm_name", "")    or ""
    s1_name_s = s1.get("norm_name_stripped", "")  or ""
    c_name_s  = candidate.get("norm_name_stripped", "") or ""

    # ---- Address fields ----
    s1_addr   = s1.get("norm_addr", "")    or ""
    c_addr    = candidate.get("norm_addr", "") or ""

    # ---- Country ----
    s1_ctr    = s1.get("country_norm", "") or ""
    c_ctr     = candidate.get("country_norm", "") or ""

    # ---- Numeric ----
    s1_nums   = s1.get("addr_numbers", "")   or ""
    c_nums    = candidate.get("addr_numbers", "") or ""
    s1_post   = s1.get("postal_codes", "")   or ""
    c_post    = candidate.get("postal_codes", "") or ""

    # ---- NAME FEATURES ----
    f = {}

    f["name_exact"]            = int(s1_name == c_name and s1_name != "")
    f["name_stripped_exact"]   = int(s1_name_s == c_name_s and s1_name_s != "")
    f["name_jaro_winkler"]     = jaro_winkler(s1_name[:100], c_name[:100])
    f["name_levenshtein"]      = normalized_levenshtein(s1_name[:150], c_name[:150])
    f["name_token_jaccard"]    = token_jaccard(s1_name, c_name)
    f["name_token_overlap"]    = token_overlap_coeff(s1_name, c_name)
    f["name_3gram_cosine"]     = ngram_cosine(s1_name, c_name, 3)
    f["name_4gram_cosine"]     = ngram_cosine(s1_name, c_name, 4)
    f["name_5gram_cosine"]     = ngram_cosine(s1_name, c_name, 5)
    f["name_3gram_jaccard"]    = ngram_jaccard(s1_name, c_name, 3)
    f["name_len_ratio"]        = length_ratio(s1_name, c_name)
    f["name_token_count_diff"] = abs(
        len(s1_name.split()) - len(c_name.split())
    )
    f["name_rarity"]           = name_rarity_score(s1_name_s, name_freq)
    f["name_stripped_jaccard"] = token_jaccard(s1_name_s, c_name_s)

    # ---- ADDRESS FEATURES ----
    f["addr_exact"]            = int(s1_addr == c_addr and s1_addr != "")
    f["addr_token_jaccard"]    = token_jaccard(s1_addr, c_addr)
    f["addr_token_overlap"]    = token_overlap_coeff(s1_addr, c_addr)
    f["addr_3gram_cosine"]     = ngram_cosine(s1_addr[:100], c_addr[:100], 3)
    f["addr_levenshtein"]      = normalized_levenshtein(s1_addr[:150], c_addr[:150])
    f["addr_num_overlap"]      = numeric_overlap(s1_addr, c_addr)
    f["addr_nums_exact"]       = int(
        bool(s1_nums) and bool(c_nums) and s1_nums == c_nums
    )
    f["postal_exact"]          = int(
        bool(s1_post) and bool(c_post) and s1_post.split(",")[0] == c_post.split(",")[0]
    )
    f["addr_len_ratio"]        = length_ratio(s1_addr, c_addr)
    f["addr_token_count_diff"] = abs(
        len(s1_addr.split()) - len(c_addr.split())
    )

    # ---- SEMANTIC ----
    f["bge_cosine"]            = float(bge_cos)

    # ---- METADATA ----
    f["country_exact"]         = int(s1_ctr == c_ctr and s1_ctr != "")
    f["country_match"]         = int(s1_ctr.lower() == c_ctr.lower() and s1_ctr != "")
    f["s1_addr_missing"]       = int(s1_addr == "")
    f["cand_addr_missing"]     = int(c_addr == "")
    f["both_addr_present"]     = int(s1_addr != "" and c_addr != "")
    f["s1_name_missing"]       = int(s1_name == "")

    # ---- CONTRADICTION FEATURES ----
    high_name_sim   = f["name_token_jaccard"] > 0.5
    high_addr_sim   = f["addr_token_jaccard"] > 0.5
    postal_mismatch = (bool(s1_post) and bool(c_post) and
                       s1_post.split(",")[0] != c_post.split(",")[0])
    num_conflict    = (bool(s1_nums) and bool(c_nums) and
                       not any(n in c_nums.split(",")
                               for n in s1_nums.split(",") if n))

    f["name_high_addr_low"]    = int(high_name_sim and not high_addr_sim
                                     and s1_addr != "" and c_addr != "")
    f["addr_high_name_low"]    = int(high_addr_sim and not high_name_sim)
    f["postal_mismatch"]       = int(postal_mismatch)
    f["num_conflict"]          = int(num_conflict)
    f["name_high_postal_diff"] = int(high_name_sim and postal_mismatch)
    f["name_high_num_diff"]    = int(high_name_sim and num_conflict)
    f["address_conflict"]      = int(postal_mismatch or num_conflict)

    return f


# ---------------------------------------------------------------------------
# Batch feature extraction
# ---------------------------------------------------------------------------

def _compute_chunk(args):
    chunk, s1_meta, cand_meta, name_freq, bge_scores_chunk = args
    results = []
    for row in chunk:
        s1_id   = row["source1_entity_id"]
        c_id    = row["candidate_entity_id"]

        s1   = s1_meta.get(s1_id, {})
        cand = cand_meta.get(c_id, {})

        bge_cos = (bge_scores_chunk or {}).get(s1_id, {}).get(c_id, 0.0)

        feats = compute_pair_features(s1, cand, name_freq, bge_cos)
        feats["source1_entity_id"]  = s1_id
        feats["candidate_entity_id"] = c_id
        feats["candidate_source"]   = row.get("candidate_source", "")
        results.append(feats)

    return results


def extract_features(
    candidates_df:  pd.DataFrame,
    s1_df:          pd.DataFrame,
    s23_df:         pd.DataFrame,
    name_freq:      Dict[str, int],
    bge_scores:     Optional[Dict] = None,
    n_workers:      int = config.NUM_WORKERS,
) -> pd.DataFrame:
    """
    Extract pair-level features for all candidate pairs.
    Returns a DataFrame with one feature vector per row.
    """
    log.info(f"Extracting features for {len(candidates_df):,} pairs...")

    # Build meta dicts for fast lookup
    s1_meta   = s1_df.set_index("entity_id").to_dict("index")
    cand_meta = s23_df.set_index("entity_id").to_dict("index")

    rows = candidates_df.to_dict("records")
    chunk_size = max(1, len(rows) // (n_workers * 4))
    chunks = [rows[i:i+chunk_size] for i in range(0, len(rows), chunk_size)]

    args = [(c, s1_meta, cand_meta, name_freq, bge_scores) for c in chunks]

    if n_workers > 1:
        with Pool(n_workers) as pool:
            results = pool.map(_compute_chunk, args)
    else:
        results = [_compute_chunk(a) for a in args]

    flat = [item for sub in results for item in sub]
    feat_df = pd.DataFrame(flat)
    log.info(f"Features extracted: {len(feat_df):,} rows, {len(feat_df.columns)} cols")
    return feat_df


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
