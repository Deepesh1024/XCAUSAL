"""
validate.py — Final output validation and statistics reporting.
"""
import logging
from typing import Dict, List, Set

import pandas as pd
import numpy as np

log = logging.getLogger(__name__)


def report_statistics(
    results:      Dict[str, List[str]],
    s1_ids:       List[str],
    candidates:   pd.DataFrame,
    gt_dict:      Dict[str, Set[str]] = None,
    split:        str = "test",
):
    """Print comprehensive statistics about the final outputs."""
    total_s1         = len(s1_ids)
    total_candidates = len(candidates)
    avg_cands        = total_candidates / max(total_s1, 1)

    max_cands_per_s1 = candidates.groupby("source1_entity_id").size().max() \
                       if len(candidates) > 0 else 0

    matched_s1   = sum(1 for v in results.values() if v)
    no_match_s1  = total_s1 - matched_s1
    multi_s1     = sum(1 for v in results.values() if len(v) > 1)

    s2_matches   = sum(1 for matches in results.values()
                       for m in matches if m.startswith("S2"))
    s3_matches   = sum(1 for matches in results.values()
                       for m in matches if m.startswith("S3"))
    total_matches= sum(len(v) for v in results.values())

    log.info("=" * 60)
    log.info(f"FINAL STATISTICS [{split}]")
    log.info("=" * 60)
    log.info(f"  Total S1 records:       {total_s1:>12,}")
    log.info(f"  Total candidate pairs:  {total_candidates:>12,}")
    log.info(f"  Avg candidates/S1:      {avg_cands:>12.1f}")
    log.info(f"  Max candidates/S1:      {max_cands_per_s1:>12}")
    log.info(f"  Matched S1:             {matched_s1:>12,}  "
             f"({matched_s1/total_s1*100:.1f}%)")
    log.info(f"  No-match S1:            {no_match_s1:>12,}  "
             f"({no_match_s1/total_s1*100:.1f}%)")
    log.info(f"  Multi-match S1:         {multi_s1:>12,}  "
             f"({multi_s1/total_s1*100:.1f}%)")
    log.info(f"  Total S2 matches:       {s2_matches:>12,}")
    log.info(f"  Total S3 matches:       {s3_matches:>12,}")
    log.info(f"  Total matches:          {total_matches:>12,}")

    if gt_dict is not None:
        # Compute actual F0.5 against ground truth
        tp = fp = fn = 0
        for s1 in s1_ids:
            pred_set = set(results.get(s1, []))
            true_set = set(gt_dict.get(s1, []))
            tp += len(pred_set & true_set)
            fp += len(pred_set - true_set)
            fn += len(true_set - pred_set)

        precision = tp / (tp + fp + 1e-9)
        recall    = tp / (tp + fn + 1e-9)
        beta      = 0.5
        f05       = (1 + beta**2) * precision * recall / \
                    (beta**2 * precision + recall + 1e-9)

        log.info(f"  GT Precision:           {precision:>12.4f}")
        log.info(f"  GT Recall:              {recall:>12.4f}")
        log.info(f"  GT F0.5:                {f05:>12.4f}")
    log.info("=" * 60)
