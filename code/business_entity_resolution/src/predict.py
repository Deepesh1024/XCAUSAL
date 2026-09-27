"""
predict.py — Inference: apply LightGBM + threshold to produce final matches.
"""
import logging
from typing import Dict, List, Optional, Set

import numpy as np
import pandas as pd
import lightgbm as lgb

from . import config
from .features import FEATURE_COLS

log = logging.getLogger(__name__)


def predict_proba(
    feat_df:  pd.DataFrame,
    booster:  lgb.Booster,
) -> np.ndarray:
    """Run LightGBM inference."""
    for col in FEATURE_COLS:
        if col not in feat_df.columns:
            feat_df[col] = 0.0

    X     = feat_df[FEATURE_COLS].values.astype(np.float32)
    proba = booster.predict(X)
    return proba


def apply_threshold(
    feat_df:   pd.DataFrame,
    probas:    np.ndarray,
    threshold: float,
    s1_ids:    List[str],
) -> Dict[str, List[str]]:
    """
    Apply the decision threshold to produce final matches.
    Returns {s1_id: [matched_ids...]}.
    Every S1 appears in the output (empty list = no match).
    """
    feat_df = feat_df.copy()
    feat_df["proba"] = probas

    results: Dict[str, List[str]] = {s1: [] for s1 in s1_ids}

    # For each s1, collect candidates above threshold
    for row in feat_df.itertuples(index=False):
        s1_id  = row.source1_entity_id
        c_id   = row.candidate_entity_id
        prob   = row.proba

        if prob >= threshold:
            if s1_id not in results:
                results[s1_id] = []
            if c_id not in results[s1_id]:
                results[s1_id].append(c_id)

    # Conservative singleton guard:
    # For s1 with exactly 1 candidate just above threshold with weak evidence,
    # require proba >= threshold + 0.05
    HIGH_CONF = threshold + 0.05

    for s1_id in list(results.keys()):
        matches = results[s1_id]
        if len(matches) == 1:
            # Check the proba
            mask = (
                (feat_df["source1_entity_id"] == s1_id) &
                (feat_df["candidate_entity_id"] == matches[0])
            )
            row_proba = feat_df.loc[mask, "proba"].values
            if len(row_proba) > 0 and row_proba[0] < HIGH_CONF:
                # Check name evidence
                name_sim = feat_df.loc[mask, "name_token_jaccard"].values
                if len(name_sim) > 0 and name_sim[0] < 0.4:
                    results[s1_id] = []  # Reject weak singleton match

    matched   = sum(1 for v in results.values() if v)
    no_match  = sum(1 for v in results.values() if not v)
    multi     = sum(1 for v in results.values() if len(v) > 1)

    log.info(
        f"Threshold={threshold:.3f} | "
        f"Matched={matched:,} | No-match={no_match:,} | Multi={multi:,}"
    )
    return results


def validate_outputs(
    results:    Dict[str, List[str]],
    s1_ids:     List[str],
    s2_ids:     Set[str],
    s3_ids:     Set[str],
    candidates: pd.DataFrame,
) -> bool:
    """Run sanity checks. Returns True if all pass."""
    valid = s2_ids | s3_ids
    errors = []

    # 1. Every S1 appears
    result_s1 = set(results.keys())
    missing   = set(s1_ids) - result_s1
    if missing:
        errors.append(f"Missing {len(missing)} S1 IDs from results")

    # 2. No duplicates in matched IDs
    for s1, matches in results.items():
        if len(matches) != len(set(matches)):
            errors.append(f"Duplicate matches for {s1}")
            break

    # 3. Every matched ID belongs to S2 or S3
    for s1, matches in results.items():
        for m in matches:
            if m not in valid:
                errors.append(f"Invalid matched ID: {m}")
                break

    # 4. No NaN
    for s1, matches in results.items():
        for m in matches:
            if not m or str(m).lower() == "nan":
                errors.append(f"NaN in results for {s1}")
                break

    # 5. Every final match exists in candidates
    cand_pairs = set(zip(candidates["source1_entity_id"],
                         candidates["candidate_entity_id"]))
    for s1, matches in results.items():
        for m in matches:
            if (s1, m) not in cand_pairs:
                errors.append(f"Match ({s1}, {m}) not in candidate_pairs.tsv")
                break

    if errors:
        log.error("SANITY CHECK FAILURES:")
        for e in errors[:10]:
            log.error(f"  {e}")
        return False

    log.info("All sanity checks PASSED")
    return True
