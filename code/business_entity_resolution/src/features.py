"""Pair features for (S1, candidate) pairs. All string similarities use
rapidfuzz.process.cpdist (C++, multithreaded, element-wise over pairs).
No feature encodes country identity (blocking guarantees same country).
"""
import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler, Levenshtein
from rapidfuzz.process import cpdist


def _cp(a, b, scorer):
    return cpdist(a, b, scorer=scorer, workers=-1).astype(np.float32)


def _first(tokens_str):
    return np.array([t.split(" ", 1)[0] if t else "" for t in tokens_str], dtype=object)


def build_features(cand, s1, other):
    i1 = cand["i1"].values
    i2 = cand["i2"].values
    F = pd.DataFrame(index=cand.index)
    F["hits"] = cand["hits"].values.astype(np.float32)

    g = lambda df, col, idx: df[col].values[idx]
    raw_a = np.char.lower(g(s1, "business_name", i1).astype(str)).astype(object)
    raw_b = np.char.lower(g(other, "business_name", i2).astype(str)).astype(object)
    F["raw_name_ratio"] = _cp(raw_a, raw_b, fuzz.ratio)

    na, nb = g(s1, "name_n", i1), g(other, "name_n", i2)
    F["name_ratio"] = _cp(na, nb, fuzz.ratio)
    F["name_partial"] = _cp(na, nb, fuzz.partial_ratio)
    F["name_tsort"] = _cp(na, nb, fuzz.token_sort_ratio)
    F["name_tset"] = _cp(na, nb, fuzz.token_set_ratio)
    F["name_jw"] = _cp(na, nb, JaroWinkler.normalized_similarity)

    ca, cb = g(s1, "core", i1), g(other, "core", i2)
    F["core_ratio"] = _cp(ca, cb, fuzz.ratio)
    F["core_tsort"] = _cp(ca, cb, fuzz.token_sort_ratio)
    F["core_tset"] = _cp(ca, cb, fuzz.token_set_ratio)
    F["core_partial"] = _cp(ca, cb, fuzz.partial_ratio)
    F["core_jw"] = _cp(ca, cb, JaroWinkler.normalized_similarity)
    F["core_lev"] = _cp(ca, cb, Levenshtein.distance)
    F["core_first_eq"] = (_first(ca) == _first(cb)).astype(np.float32)

    ka, kb = g(s1, "core_skel", i1), g(other, "core_skel", i2)
    F["skel_tset"] = _cp(ka, kb, fuzz.token_set_ratio)
    F["skel_tsort"] = _cp(ka, kb, fuzz.token_sort_ratio)
    sa, sb = g(s1, "core_cat_skel", i1), g(other, "core_cat_skel", i2)
    F["catskel_ratio"] = _cp(sa, sb, fuzz.ratio)
    F["catskel_partial"] = _cp(sa, sb, fuzz.partial_ratio)

    aa, ab = g(s1, "addr_n", i1), g(other, "addr_n", i2)
    F["addr_ratio"] = _cp(aa, ab, fuzz.ratio)
    F["addr_tset"] = _cp(aa, ab, fuzz.token_set_ratio)
    F["addr_tsort"] = _cp(aa, ab, fuzz.token_sort_ratio)
    F["addr_partial"] = _cp(aa, ab, fuzz.partial_ratio)
    xa, xb = g(s1, "addr_alpha", i1), g(other, "addr_alpha", i2)
    F["addralpha_tset"] = _cp(xa, xb, fuzz.token_set_ratio)
    F["addralpha_tsort"] = _cp(xa, xb, fuzz.token_sort_ratio)

    ma, mb = g(s1, "nums", i1), g(other, "nums", i2)
    F["nums_tset"] = _cp(ma, mb, fuzz.token_set_ratio)
    F["nums_tsort"] = _cp(ma, mb, fuzz.token_sort_ratio)
    fa, fb = _first(ma), _first(mb)
    F["num1_eq"] = ((fa == fb) & (fa != "")).astype(np.float32)
    F["a_has_num"] = (fa != "").astype(np.float32)
    F["b_has_num"] = (fb != "").astype(np.float32)
    F["b_addr_missing"] = (g(other, "business_address", i2) == "").astype(np.float32)
    F["b_is_s3"] = np.char.startswith(g(other, "entity_id", i2).astype(str), "S3").astype(np.float32)
    F["b_nonlatin"] = g(other, "nonlatin", i2).astype(np.float32)
    la = np.array([len(x) for x in ca], dtype=np.float32)
    lb = np.array([len(x) for x in cb], dtype=np.float32)
    F["core_len_a"] = la
    F["core_len_diff"] = np.abs(la - lb)
    F["core_ntok_b"] = np.array([x.count(" ") + 1 if x else 0 for x in cb], dtype=np.float32)

    # context features: rank/gap among the S1's candidates and among the
    # candidate record's S1s (helps the one-owner decision)
    combo = F["core_tset"] + F["addr_tset"] + F["name_ratio"]
    F["combo"] = combo
    grp1 = pd.Series(combo.values).groupby(i1)
    F["rank_in_s1"] = grp1.rank(ascending=False, method="min").values.astype(np.float32)
    F["gap_s1_best"] = (grp1.transform("max").values - combo.values).astype(np.float32)
    F["n_cand_s1"] = grp1.transform("size").values.astype(np.float32)
    grp2 = pd.Series(combo.values).groupby(i2)
    F["rank_in_b"] = grp2.rank(ascending=False, method="min").values.astype(np.float32)
    F["gap_b_best"] = (grp2.transform("max").values - combo.values).astype(np.float32)
    F["n_cand_b"] = grp2.transform("size").values.astype(np.float32)
    return F
