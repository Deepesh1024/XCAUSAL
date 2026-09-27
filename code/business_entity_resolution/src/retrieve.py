"""
retrieve.py — Candidate generation using multiple strategies and union.
"""
import logging
import time
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Optional

import numpy as np
import pandas as pd

from . import config
from .indexes import (
    RareTokenIndex, ExactNameIndex, AddressIndex,
    CharNgramIndex, build_name_frequency_table
)

log = logging.getLogger(__name__)


def _build_candidate_score(
    exact_score:    float,
    rare_score:     float,
    tfidf_score:    float,
    addr_score:     float,
    num_score:      float,
    bge_score:      float,
    country_match:  int,
) -> float:
    """Weighted preliminary score for candidate trimming only — NOT for final matching."""
    return (
        config.CAND_WEIGHT_NAME_LEX  * tfidf_score +
        config.CAND_WEIGHT_ADDR_LEX  * addr_score  +
        config.CAND_WEIGHT_NUM_OVERLAP * num_score +
        config.CAND_WEIGHT_BGE       * bge_score   +
        config.CAND_WEIGHT_RARE_TOK  * min(rare_score / 5.0, 1.0) +
        config.CAND_WEIGHT_COUNTRY   * country_match +
        (0.5 if exact_score > 0 else 0.0)  # bonus for exact name
    )


class CandidateGenerator:
    """
    Combines all retrieval strategies and returns candidate DataFrames.
    """

    def __init__(
        self,
        exact_idx:     ExactNameIndex,
        rare_idx:      RareTokenIndex,
        tfidf_idx:     CharNgramIndex,
        addr_idx:      AddressIndex,
        name_freq:     Dict[str, int],
        s2_meta:       pd.DataFrame,
        s3_meta:       pd.DataFrame,
        bge_scores:    Optional[Dict[str, Dict[str, float]]] = None,
    ):
        self.exact_idx  = exact_idx
        self.rare_idx   = rare_idx
        self.tfidf_idx  = tfidf_idx
        self.addr_idx   = addr_idx
        self.name_freq  = name_freq
        self.bge_scores = bge_scores or {}

        # Build lookup dicts for fast access
        self._build_meta(s2_meta, s3_meta)

    def _build_meta(self, s2: pd.DataFrame, s3: pd.DataFrame):
        self.meta: Dict[str, dict] = {}
        for df, src in [(s2, "S2"), (s3, "S3")]:
            for row in df.itertuples(index=False):
                eid = row.entity_id
                self.meta[eid] = {
                    "src":         src,
                    "country":     getattr(row, "country_norm", ""),
                    "postal":      getattr(row, "postal_codes", "") or "",
                    "addr_nums":   getattr(row, "addr_numbers", "") or "",
                }

    def _country_match(self, c1: str, c2: str) -> int:
        if not c1 or not c2:
            return 0
        return int(c1.lower().strip() == c2.lower().strip())

    def generate_for_batch(
        self,
        s1_rows: List[dict],
        tfidf_scores: List[List[Tuple[str, float]]],
    ) -> List[List[dict]]:
        """
        For a batch of S1 rows (with pre-computed tfidf scores),
        return a list of candidate lists.
        Each candidate dict: {candidate_id, candidate_source, prelim_score, ...}
        """
        all_candidates = []

        for idx, s1 in enumerate(s1_rows):
            s1_id      = s1["entity_id"]
            norm_name  = s1.get("norm_name_stripped", s1.get("norm_name", ""))
            norm_addr  = s1.get("norm_addr", "")
            country    = s1.get("country_norm", "")
            postals    = s1.get("postal_codes", "") or ""
            addr_nums  = s1.get("addr_numbers", "") or ""

            # Accumulate candidates with scores
            cand_scores: Dict[str, dict] = {}

            # A) Exact name
            for eid in self.exact_idx.query(norm_name):
                m = self.meta.get(eid)
                if m is None: continue
                cand_scores[eid] = {
                    "exact": 1.0, "rare": 0.0, "tfidf": 1.0,
                    "addr": 0.0, "num": 0.0, "bge": 0.0,
                    "country": self._country_match(country, m["country"]),
                    "src": m["src"],
                }

            # B) Rare-token retrieval
            for eid, score in self.rare_idx.query(norm_name):
                m = self.meta.get(eid)
                if m is None: continue
                if eid in cand_scores:
                    cand_scores[eid]["rare"] = score
                else:
                    cand_scores[eid] = {
                        "exact": 0.0, "rare": score, "tfidf": 0.0,
                        "addr": 0.0, "num": 0.0, "bge": 0.0,
                        "country": self._country_match(country, m["country"]),
                        "src": m["src"],
                    }

            # C) TF-IDF char n-gram
            for eid, score in (tfidf_scores[idx] if tfidf_scores else []):
                m = self.meta.get(eid)
                if m is None: continue
                if eid in cand_scores:
                    cand_scores[eid]["tfidf"] = max(cand_scores[eid]["tfidf"], score)
                else:
                    cand_scores[eid] = {
                        "exact": 0.0, "rare": 0.0, "tfidf": score,
                        "addr": 0.0, "num": 0.0, "bge": 0.0,
                        "country": self._country_match(country, m["country"]),
                        "src": m["src"],
                    }

            # D) Postal code retrieval
            for eid in self.addr_idx.query_postal(postals):
                m = self.meta.get(eid)
                if m is None: continue
                if eid in cand_scores:
                    cand_scores[eid]["addr"] = max(cand_scores[eid]["addr"], 0.8)
                else:
                    cand_scores[eid] = {
                        "exact": 0.0, "rare": 0.0, "tfidf": 0.0,
                        "addr": 0.8, "num": 0.0, "bge": 0.0,
                        "country": self._country_match(country, m["country"]),
                        "src": m["src"],
                    }

            # E) Address token retrieval
            for eid, score in self.addr_idx.query_addr_tokens(norm_addr):
                m = self.meta.get(eid)
                if m is None: continue
                addr_score = min(score / 3.0, 1.0)
                if eid in cand_scores:
                    cand_scores[eid]["addr"] = max(cand_scores[eid]["addr"], addr_score)
                else:
                    cand_scores[eid] = {
                        "exact": 0.0, "rare": 0.0, "tfidf": 0.0,
                        "addr": addr_score, "num": 0.0, "bge": 0.0,
                        "country": self._country_match(country, m["country"]),
                        "src": m["src"],
                    }

            # F) Number overlap retrieval
            for eid in self.addr_idx.query_numbers(addr_nums):
                m = self.meta.get(eid)
                if m is None: continue
                if eid in cand_scores:
                    cand_scores[eid]["num"] = max(cand_scores[eid]["num"], 0.5)
                else:
                    cand_scores[eid] = {
                        "exact": 0.0, "rare": 0.0, "tfidf": 0.0,
                        "addr": 0.0, "num": 0.5, "bge": 0.0,
                        "country": self._country_match(country, m["country"]),
                        "src": m["src"],
                    }

            # G) BGE dense scores (if available)
            for eid, score in self.bge_scores.get(s1_id, {}).items():
                m = self.meta.get(eid)
                if m is None: continue
                if eid in cand_scores:
                    cand_scores[eid]["bge"] = score
                else:
                    cand_scores[eid] = {
                        "exact": 0.0, "rare": 0.0, "tfidf": 0.0,
                        "addr": 0.0, "num": 0.0, "bge": score,
                        "country": self._country_match(country, m["country"]),
                        "src": m["src"],
                    }

            # Compute preliminary scores and rank
            scored = []
            for eid, sc in cand_scores.items():
                prelim = _build_candidate_score(
                    sc["exact"], sc["rare"], sc["tfidf"],
                    sc["addr"],  sc["num"],  sc["bge"],
                    sc["country"]
                )
                scored.append({
                    "s1_id":             s1_id,
                    "candidate_id":      eid,
                    "candidate_source":  sc["src"],
                    "prelim_score":      prelim,
                    "exact_score":       sc["exact"],
                    "rare_score":        sc["rare"],
                    "tfidf_score":       sc["tfidf"],
                    "addr_score":        sc["addr"],
                    "num_score":         sc["num"],
                    "bge_score":         sc["bge"],
                    "country_prelim":    sc["country"],
                })

            # Sort and trim to K
            scored.sort(key=lambda x: -x["prelim_score"])
            all_candidates.append(scored[:config.CANDIDATE_K])

        return all_candidates


def generate_candidates(
    s1_df:         pd.DataFrame,
    s2_df:         pd.DataFrame,
    s3_df:         pd.DataFrame,
    exact_idx:     ExactNameIndex,
    rare_idx:      RareTokenIndex,
    tfidf_idx:     CharNgramIndex,
    addr_idx:      AddressIndex,
    name_freq:     Dict[str, int],
    bge_scores:    Optional[Dict] = None,
    batch_size:    int = config.TFIDF_BATCH_SIZE,
    split:         str = "train",
) -> pd.DataFrame:
    """
    Full candidate generation loop over all S1 records.
    Returns a flat DataFrame of all candidate pairs.
    """
    generator = CandidateGenerator(
        exact_idx, rare_idx, tfidf_idx, addr_idx,
        name_freq, s2_df, s3_df, bge_scores
    )

    s1_records = s1_df.to_dict("records")
    n = len(s1_records)
    all_pairs = []

    log.info(f"Generating candidates for {n:,} S1 records (batch={batch_size})...")
    t0 = time.time()

    for start in range(0, n, batch_size):
        end   = min(start + batch_size, n)
        batch = s1_records[start:end]

        # TF-IDF retrieval for the whole batch at once
        batch_texts  = [r.get("norm_name_stripped", r.get("norm_name", ""))
                        for r in batch]
        tfidf_scores = tfidf_idx.query_batch(batch_texts, top_k=config.CANDIDATE_K)

        cand_lists = generator.generate_for_batch(batch, tfidf_scores)

        for cands in cand_lists:
            all_pairs.extend(cands)

        if (end % (batch_size * 5) == 0) or end == n:
            elapsed  = time.time() - t0
            rps      = end / elapsed
            eta      = (n - end) / rps if rps > 0 else 0
            avg_cand = len(all_pairs) / max(end, 1)
            log.info(
                f"  [{elapsed/60:.1f}m] S1 processed: {end:,}/{n:,} | "
                f"avg_candidates: {avg_cand:.1f} | ETA: {eta/60:.1f}m"
            )

    log.info(f"Candidate generation done: {len(all_pairs):,} pairs from {n:,} S1")

    df_cands = pd.DataFrame(all_pairs)
    df_cands.rename(columns={
        "s1_id":            "source1_entity_id",
        "candidate_id":     "candidate_entity_id",
        "candidate_source": "candidate_source",
    }, inplace=True)

    return df_cands
