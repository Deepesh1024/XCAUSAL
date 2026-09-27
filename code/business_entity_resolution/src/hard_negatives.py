"""
hard_negatives.py — Construct hard negative pairs for LightGBM training.
"""
import logging
import random
from typing import Dict, List, Set, Tuple

import pandas as pd
import numpy as np

from . import config

log = logging.getLogger(__name__)

random.seed(config.RANDOM_SEED)
np.random.seed(config.RANDOM_SEED)


def build_hard_negatives(
    candidates_df:   pd.DataFrame,
    gt_dict:         Dict[str, Set[str]],
    s1_df:           pd.DataFrame,
    s23_df:          pd.DataFrame,
    max_neg_per_pos: int = 5,
) -> pd.DataFrame:
    """
    Build hard negative pairs from the candidate pool.
    These are candidates that were retrieved but are NOT in GT.

    Strategy:
    1. Same-country similar-name negatives (highest priority)
    2. High-tfidf-score but wrong match negatives
    3. Address-conflict negatives
    4. Random negatives from candidates
    """
    log.info("Building hard negatives from candidate pool...")

    # Label every candidate as pos/neg
    pos_pairs = set()
    for s1_id, matches in gt_dict.items():
        for m in matches:
            pos_pairs.add((s1_id, m))

    candidates_df["label"] = candidates_df.apply(
        lambda r: int((r["source1_entity_id"], r["candidate_entity_id"]) in pos_pairs),
        axis=1
    )

    positives = candidates_df[candidates_df["label"] == 1]
    negatives = candidates_df[candidates_df["label"] == 0]

    log.info(f"  Positives in candidate pool: {len(positives):,}")
    log.info(f"  Negatives in candidate pool: {len(negatives):,}")

    # Strategy 1: High tfidf score negatives (lexically similar but wrong)
    hard_negs_tfidf = negatives.nlargest(
        min(len(positives) * max_neg_per_pos, len(negatives)),
        "tfidf_score"
    )

    # Strategy 2: High address score negatives (addr similar but wrong)
    hard_negs_addr  = negatives.nlargest(
        min(len(positives) * 2, len(negatives)),
        "addr_score"
    )

    # Strategy 3: Random negatives
    n_random = min(len(positives) * 3, len(negatives))
    rand_negs = negatives.sample(n=n_random, random_state=config.RANDOM_SEED)

    # Union
    all_negs = pd.concat([hard_negs_tfidf, hard_negs_addr, rand_negs], ignore_index=True)
    all_negs = all_negs.drop_duplicates(
        subset=["source1_entity_id", "candidate_entity_id"]
    )

    all_training = pd.concat([positives, all_negs], ignore_index=True)

    log.info(f"  Training pairs: {len(all_training):,} "
             f"(pos={len(positives):,}, neg={len(all_negs):,})")

    return all_training
