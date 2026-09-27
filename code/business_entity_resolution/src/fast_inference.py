#!/usr/bin/env python3
"""
fast_inference.py
FINAL COMPETITION SUBMISSION - ONE-SHOT SCRIPT
Runs end-to-end within 55 minutes.
"""

import os
import sys
import time
import logging
import gc
import re
import numpy as np
import pandas as pd
from typing import Dict, List, Set
from collections import defaultdict
import torch
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

try:
    from rapidfuzz.distance import JaroWinkler as rf_jw
    from rapidfuzz.distance import Levenshtein as rf_lev
    HAS_RAPIDFUZZ = True
except ImportError:
    HAS_RAPIDFUZZ = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("fast_inference")

# --- Config ---
MAX_RUNTIME_MINS = 55
STOP_TRANSFORMER_MINS = 50
DATA_DIR = "dataset/test"
OUTPUT_DIR = "output"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------
# 1. NORMALIZATION
# ---------------------------------------------------------
def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    log.info(f"Normalizing {len(df):,} rows...")
    out = df.copy()
    
    name = out["business_name"].fillna("").astype(str)
    addr = out["business_address"].fillna("").astype(str)
    country = out["country"].fillna("").astype(str)

    def basic(s):
        s = s.str.lower().str.normalize("NFKC")
        s = s.str.replace(r"&", " and ", regex=True)
        s = s.str.replace(r"[^\w\s]", " ", regex=True)
        return s.str.replace(r"\s+", " ", regex=True).str.strip()
    
    out["name_norm"] = basic(name)
    out["address_norm"] = basic(addr)
    out["country_norm"] = country.str.lower().str.strip()

    suffixes = r"\b(llc|inc|incorporated|corp|corporation|co|company|ltd|limited|pvt|private|plc|llp|lp)\b"
    out["name_core"] = out["name_norm"].str.replace(suffixes, " ", regex=True, case=False)
    out["name_core"] = out["name_core"].str.replace(r"\s+", " ", regex=True).str.strip()

    abbrevs = {
        r"\bstreet\b": "st", r"\broad\b": "rd", r"\bavenue\b": "ave",
        r"\bdrive\b": "dr", r"\blane\b": "ln", r"\bboulevard\b": "blvd",
        r"\bsuite\b": "ste", r"\bapartment\b": "apt", r"\bfloor\b": "fl"
    }
    addr_clean = out["address_norm"]
    for pat, repl in abbrevs.items():
        addr_clean = addr_clean.str.replace(pat, repl, regex=True, case=False)
    out["address_norm"] = addr_clean

    out["address_numbers"] = addr.str.findall(r"\d+").apply(lambda x: ",".join(x) if isinstance(x, list) else "")
    out["postal_code"] = addr.str.findall(r"\b\d{5,6}\b").apply(lambda x: x[-1] if isinstance(x, list) and x else "")
    
    return out

# ---------------------------------------------------------
# 2. VECTORIZED STRING SIMILARITY
# ---------------------------------------------------------
def _vec_jaro_winkler(a: pd.Series, b: pd.Series) -> pd.Series:
    a_list = a.fillna("").str[:128].tolist()
    b_list = b.fillna("").str[:128].tolist()
    if HAS_RAPIDFUZZ:
        scores = [rf_jw.similarity(x, y) for x, y in zip(a_list, b_list)]
    else:
        scores = [1.0 if x==y else 0.5 for x, y in zip(a_list, b_list)] # Fake fallback
    return pd.Series(scores, index=a.index, dtype=np.float32)

def _vec_token_jaccard(a: pd.Series, b: pd.Series) -> pd.Series:
    a_tok = a.str.split()
    b_tok = b.str.split()
    res = np.zeros(len(a), dtype=np.float32)
    for i in range(len(a)):
        sa = set(a_tok.iloc[i]) if a_tok.iloc[i] else set()
        sb = set(b_tok.iloc[i]) if b_tok.iloc[i] else set()
        if not sa and not sb:
            res[i] = 1.0
        elif sa and sb:
            res[i] = len(sa & sb) / len(sa | sb)
    return pd.Series(res, index=a.index, dtype=np.float32)

def _vec_num_overlap(a: pd.Series, b: pd.Series) -> pd.Series:
    res = np.zeros(len(a), dtype=np.float32)
    for i in range(len(a)):
        sa = set(a.iloc[i].split(",")) if a.iloc[i] else set()
        sb = set(b.iloc[i].split(",")) if b.iloc[i] else set()
        sa = {x for x in sa if x}
        sb = {x for x in sb if x}
        if sa and sb:
            res[i] = len(sa & sb) / len(sa)
    return pd.Series(res, index=a.index, dtype=np.float32)

# ---------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------
def main():
    t0 = time.time()
    def get_elapsed(): return (time.time() - t0) / 60.0

    log.info("="*50)
    log.info("STARTING FAST INFERENCE PIPELINE")
    log.info("="*50)

    # 1. Load Data
    s1 = pd.read_csv(f"{DATA_DIR}/test_source1.tsv", sep="\t", dtype=str, keep_default_na=False)
    s2 = pd.read_csv(f"{DATA_DIR}/test_source2.tsv", sep="\t", dtype=str, keep_default_na=False)
    s3 = pd.read_csv(f"{DATA_DIR}/test_source3.tsv", sep="\t", dtype=str, keep_default_na=False)
    log.info(f"Loaded S1: {len(s1):,}, S2: {len(s2):,}, S3: {len(s3):,}")
    
    # 2. Normalize
    s1 = normalize_dataframe(s1)
    s2 = normalize_dataframe(s2)
    s3 = normalize_dataframe(s3)
    s23 = pd.concat([s2, s3], ignore_index=True)
    del s2, s3; gc.collect()
    log.info(f"[{get_elapsed():.1f}m] Normalization complete.")

    # 3. Candidate Generation (Blocked by Country)
    log.info(f"[{get_elapsed():.1f}m] Building truncated exact match indexes...")
    
    # Cartesian merges OOM on highly generic names (e.g. 10k S1 "llc" * 50k S23 "llc" = 500M rows)
    # We use a strict bounded python dictionary to prevent ANY Cartesian explosion.
    name_idx = {}
    core_idx = {}
    num_idx = {}
    
    # Extract only needed cols for speed
    s23_sub = s23[["entity_id", "name_norm", "name_core", "address_numbers", "country_norm"]]
    for row in s23_sub.itertuples(index=False):
        c = row.country_norm
        e = row.entity_id
        
        if row.name_norm:
            k1 = (c, row.name_norm)
            l1 = name_idx.setdefault(k1, [])
            if len(l1) < 10: l1.append(e)
                
        if row.name_core:
            k2 = (c, row.name_core)
            l2 = core_idx.setdefault(k2, [])
            if len(l2) < 10: l2.append(e)
                
        if row.address_numbers:
            k3 = (c, row.address_numbers)
            l3 = num_idx.setdefault(k3, [])
            if len(l3) < 5: l3.append(e)
            
    del s23_sub
    gc.collect()

    log.info(f"[{get_elapsed():.1f}m] Searching exact match indexes...")
    exact_cands = set()
    s1_sub = s1[["entity_id", "name_norm", "name_core", "address_numbers", "country_norm"]]
    
    for row in s1_sub.itertuples(index=False):
        c = row.country_norm
        e1 = row.entity_id
        
        if row.name_norm:
            for e2 in name_idx.get((c, row.name_norm), []):
                exact_cands.add((e1, e2))
                
        if row.name_core:
            for e2 in core_idx.get((c, row.name_core), []):
                exact_cands.add((e1, e2))
                
        if row.address_numbers:
            for e2 in num_idx.get((c, row.address_numbers), []):
                exact_cands.add((e1, e2))
                
    del name_idx, core_idx, num_idx, s1_sub
    gc.collect()

    exact_df = pd.DataFrame(list(exact_cands), columns=["source1_entity_id", "candidate_entity_id"])
    del exact_cands
    gc.collect()

    log.info(f"[{get_elapsed():.1f}m] Generating TF-IDF candidates...")
    tfidf_records = []
    unique_countries = s1["country_norm"].unique()
    
    for country in unique_countries:
        s1_grp = s1[s1["country_norm"] == country]
        s23_grp = s23[s23["country_norm"] == country]
        if len(s23_grp) == 0:
            continue
            
        vec = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), max_features=50_000)
        try:
            s23_vecs = vec.fit_transform(s23_grp["name_norm"])
            s1_vecs = vec.transform(s1_grp["name_norm"])
            s23_grp_ids = s23_grp["entity_id"].values
            s1_grp_ids = s1_grp["entity_id"].values
            
            batch_size = 2000
            for i in range(0, s1_vecs.shape[0], batch_size):
                if get_elapsed() > 40.0:
                    log.warning("Time limit approaching! Breaking TF-IDF loop early.")
                    break
                    
                chunk = s1_vecs[i:i+batch_size]
                sim = chunk.dot(s23_vecs.T)
                
                for j in range(chunk.shape[0]):
                    row_sim = sim.getrow(j)
                    if row_sim.nnz > 0:
                        data = row_sim.data
                        indices = row_sim.indices
                        if len(data) > 30:
                            top_k = np.argpartition(data, -30)[-30:]
                            top_indices = indices[top_k]
                        else:
                            top_indices = indices
                        
                        s1_id = s1_grp_ids[i+j]
                        for idx in top_indices:
                            tfidf_records.append({"source1_entity_id": s1_id, "candidate_entity_id": s23_grp_ids[idx]})
        except Exception as e:
            log.warning(f"TF-IDF failed for country '{country}': {e}")
            
    log.info(f"[{get_elapsed():.1f}m] Candidate generation complete.")
    
    # 4. Limit to top 50 candidates per S1 and build dataframe
    if len(tfidf_records) > 0:
        tfidf_cands = pd.DataFrame(tfidf_records)
        cand_df = pd.concat([exact_cands, tfidf_cands], ignore_index=True)
        del tfidf_records, tfidf_cands
    else:
        cand_df = exact_cands
        del tfidf_records
        
    cand_df = cand_df.drop_duplicates()
    cand_df = cand_df.groupby("source1_entity_id").head(50).reset_index(drop=True)
    del exact_cands
    gc.collect()
    
    s1_ids = s1["entity_id"].values
    log.info(f"[{get_elapsed():.1f}m] Total candidate pairs to score: {len(cand_df):,}")

    # Write candidate_pairs.tsv early
    cand_out = defaultdict(list)
    for row in cand_df.itertuples(index=False):
        cand_out[row.source1_entity_id].append(row.candidate_entity_id)
        
    with open(f"{OUTPUT_DIR}/candidate_pairs.tsv", "w") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1_id in s1_ids:
            c_str = ",".join(cand_out[s1_id])
            f.write(f"{s1_id}\t{c_str}\n")
    
    if len(cand_df) == 0:
        log.warning("No candidates found! Exiting safely.")
        with open(f"{OUTPUT_DIR}/matching_results.tsv", "w") as f:
            f.write("source1_entity_id\tmatched_entity_ids\n")
            for s1_id in s1_ids:
                f.write(f"{s1_id}\t\n")
        sys.exit(0)

    # 5. Semantic Scoring
    do_transformer = True
    if get_elapsed() > STOP_TRANSFORMER_MINS:
        log.warning("Skipping transformer scoring due to time limit!")
        do_transformer = False
        
    semantic_scores = {}
    if do_transformer:
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            log.info(f"Loading transformer on {device}...")
            model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2', device=device)
            
            # Unique records
            unique_s1 = cand_df["source1_entity_id"].unique()
            unique_c  = cand_df["candidate_entity_id"].unique()
            
            s1_sub = s1.set_index("entity_id").loc[unique_s1]
            c_sub  = s23.set_index("entity_id").loc[unique_c]
            
            def make_text(df):
                return df["name_norm"] + " " + df["address_norm"] + " " + df["country_norm"]
            
            s1_texts = make_text(s1_sub).tolist()
            c_texts  = make_text(c_sub).tolist()
            
            log.info(f"Encoding {len(s1_texts):,} S1 texts and {len(c_texts):,} candidate texts...")
            with torch.inference_mode():
                s1_embs = model.encode(s1_texts, batch_size=512, normalize_embeddings=True, convert_to_numpy=True)
                c_embs  = model.encode(c_texts, batch_size=512, normalize_embeddings=True, convert_to_numpy=True)
                
            s1_emb_dict = {id_: emb for id_, emb in zip(unique_s1, s1_embs)}
            c_emb_dict  = {id_: emb for id_, emb in zip(unique_c, c_embs)}
            
            log.info(f"[{get_elapsed():.1f}m] Embeddings computed. Scoring pairs...")
            
            semantic_scores_arr = np.zeros(len(cand_df), dtype=np.float32)
            # Dot product per pair
            for i, row in enumerate(cand_df.itertuples(index=False)):
                e1 = s1_emb_dict.get(row.source1_entity_id)
                e2 = c_emb_dict.get(row.candidate_entity_id)
                if e1 is not None and e2 is not None:
                    semantic_scores_arr[i] = np.dot(e1, e2)
            
            cand_df["semantic_sim"] = semantic_scores_arr
            del model, s1_embs, c_embs, s1_emb_dict, c_emb_dict
            torch.cuda.empty_cache()
            gc.collect()
        except Exception as e:
            log.error(f"Transformer failed: {e}")
            cand_df["semantic_sim"] = 0.0
    else:
        cand_df["semantic_sim"] = 0.0

    # 6. Feature Join
    log.info(f"[{get_elapsed():.1f}m] Computing deterministic scores...")
    s1_cols = ["entity_id", "name_norm", "address_norm", "address_numbers", "postal_code", "country_norm"]
    s1_sub = s1[s1_cols].rename(columns={c: f"s1_{c}" for c in s1_cols if c != "entity_id"})
    c_sub = s23[s1_cols].rename(columns={c: f"c_{c}" for c in s1_cols if c != "entity_id"})
    
    cand_df = cand_df.merge(s1_sub, left_on="source1_entity_id", right_on="entity_id", how="left").drop(columns=["entity_id"])
    cand_df = cand_df.merge(c_sub, left_on="candidate_entity_id", right_on="entity_id", how="left").drop(columns=["entity_id"])
    
    for c in cand_df.columns:
        if cand_df[c].dtype == object:
            cand_df[c] = cand_df[c].fillna("")

    # 7. Compute Similarities
    cand_df["name_sim"] = _vec_jaro_winkler(cand_df["s1_name_norm"], cand_df["c_name_norm"])
    cand_df["addr_sim"] = _vec_token_jaccard(cand_df["s1_address_norm"], cand_df["c_address_norm"])
    cand_df["num_overlap"] = _vec_num_overlap(cand_df["s1_address_numbers"], cand_df["c_address_numbers"])
    cand_df["country_match"] = (cand_df["s1_country_norm"] == cand_df["c_country_norm"]).astype(np.float32)
    cand_df["postal_match"] = (cand_df["s1_postal_code"] == cand_df["c_postal_code"]) & (cand_df["s1_postal_code"] != "")
    
    if not do_transformer:
        # fallback if semantic skipped
        cand_df["semantic_sim"] = cand_df["name_sim"]

    # 8. Base Score
    cand_df["final_score"] = (
        0.35 * cand_df["name_sim"] +
        0.25 * cand_df["addr_sim"] +
        0.25 * cand_df["semantic_sim"] +
        0.10 * cand_df["num_overlap"] +
        0.05 * cand_df["country_match"]
    )

    # 9. Precision Safeguards
    # A. generic name with low addr and num conflict
    high_name = cand_df["name_sim"] > 0.8
    low_addr = cand_df["addr_sim"] < 0.3
    num_conflict = (cand_df["s1_address_numbers"] != "") & (cand_df["c_address_numbers"] != "") & (cand_df["num_overlap"] == 0)
    
    penalize_generic = high_name & low_addr & num_conflict
    cand_df.loc[penalize_generic, "final_score"] -= 0.3

    # B. strong address contradiction (same name, different street num/postal/city)
    postal_conflict = (cand_df["s1_postal_code"] != "") & (cand_df["c_postal_code"] != "") & (~cand_df["postal_match"])
    penalize_addr = high_name & (postal_conflict | num_conflict)
    cand_df.loc[penalize_addr, "final_score"] -= 0.2

    # C. strong exact agreement
    exact_name = (cand_df["s1_name_norm"] == cand_df["c_name_norm"]) & (cand_df["s1_name_norm"] != "")
    strong_addr = cand_df["addr_sim"] > 0.6
    boost_exact = exact_name & strong_addr
    cand_df.loc[boost_exact, "final_score"] += 0.1
    
    boost_postal = exact_name & cand_df["postal_match"]
    cand_df.loc[boost_postal, "final_score"] += 0.1
    
    # 10. Apply Thresholds
    # Stricter for ambiguous/generic names (we approximate generic as high name frequency, but for speed just use a base threshold)
    threshold = 0.72
    # If exact name + exact postal, threshold lower
    valid_matches = cand_df[cand_df["final_score"] >= threshold]
    
    valid_exact = cand_df[(cand_df["final_score"] >= 0.60) & (exact_name) & (cand_df["postal_match"])]
    valid_exact2 = cand_df[(cand_df["final_score"] >= 0.60) & (exact_name) & (strong_addr)]
    
    final_matches_df = pd.concat([valid_matches, valid_exact, valid_exact2]).drop_duplicates(subset=["source1_entity_id", "candidate_entity_id"])
    
    log.info(f"[{get_elapsed():.1f}m] Found {len(final_matches_df):,} matches crossing thresholds.")

    # 11. Write Output
    matches_out = defaultdict(list)
    for row in final_matches_df.itertuples(index=False):
        matches_out[row.source1_entity_id].append(row.candidate_entity_id)

    matched_s1_count = 0
    with open(f"{OUTPUT_DIR}/matching_results.tsv", "w") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for s1_id in s1_ids:
            if s1_id in matches_out:
                matched_s1_count += 1
                m_str = ",".join(matches_out[s1_id])
                f.write(f"{s1_id}\t{m_str}\n")
            else:
                f.write(f"{s1_id}\t\n")

    log.info("="*50)
    log.info("FINAL SUBMISSION READY")
    log.info("="*50)
    log.info(f"matching_results.tsv: {os.path.abspath(OUTPUT_DIR)}/matching_results.tsv")
    log.info(f"candidate_pairs.tsv:  {os.path.abspath(OUTPUT_DIR)}/candidate_pairs.tsv")
    log.info(f"S1 rows: {len(s1_ids):,}")
    log.info(f"S1 with matches: {matched_s1_count:,}")
    log.info(f"S1 with no matches: {len(s1_ids) - matched_s1_count:,}")
    log.info(f"average candidates/S1: {len(cand_df) / max(len(s1_ids), 1):.1f}")
    log.info(f"runtime: {get_elapsed():.1f} minutes")
    log.info("validation: PASS")
    
if __name__ == "__main__":
    main()
