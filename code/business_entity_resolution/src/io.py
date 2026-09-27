"""
io.py — Data loading, checkpoint helpers, and output writers.
"""
import os
import json
import logging
import tempfile
import pickle
import pandas as pd
import numpy as np

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Safe atomic writes
# ---------------------------------------------------------------------------

def atomic_save_pickle(obj, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(obj, f, protocol=4)
    os.replace(tmp, path)
    log.debug(f"Saved checkpoint: {path}")


def atomic_load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def atomic_save_parquet(df: pd.DataFrame, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)
    log.debug(f"Saved parquet: {path}")


def atomic_load_parquet(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)


def atomic_save_numpy(arr: np.ndarray, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    np.save(tmp, arr)
    os.replace(tmp + ".npy" if not tmp.endswith(".npy") else tmp, path)
    log.debug(f"Saved numpy: {path}")


def atomic_save_json(obj, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def atomic_load_json(path: str):
    with open(path, "r") as f:
        return json.load(f)


def checkpoint_exists(*paths) -> bool:
    return all(os.path.exists(p) for p in paths)


# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------

def load_source(path: str, source_tag: str) -> pd.DataFrame:
    """Load a source TSV. source_tag is 'S1', 'S2', or 'S3'."""
    log.info(f"Loading {source_tag} from {path}")
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    df["_src"] = source_tag

    # Ensure columns always exist
    for col in ["business_name", "business_address", "country"]:
        if col not in df.columns:
            df[col] = ""
        else:
            df[col] = df[col].fillna("")

    log.info(f"  {source_tag}: {len(df):,} records")
    return df


def load_ground_truth(path: str) -> pd.DataFrame:
    log.info(f"Loading ground truth from {path}")
    gt = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    gt["matched_entity_ids"] = gt["matched_entity_ids"].fillna("")
    log.info(f"  GT rows: {len(gt):,}")
    return gt


def parse_gt_to_pairs(gt: pd.DataFrame):
    """Return list of (s1_id, match_id) for all positive pairs."""
    pairs = []
    for row in gt.itertuples(index=False):
        s1 = row.source1_entity_id
        for m in row.matched_entity_ids.split(","):
            m = m.strip()
            if m:
                pairs.append((s1, m))
    return pairs


def parse_gt_to_dict(gt: pd.DataFrame) -> dict:
    """Return {s1_id: set_of_match_ids}."""
    d = {}
    for row in gt.itertuples(index=False):
        s1 = row.source1_entity_id
        matches = {m.strip() for m in row.matched_entity_ids.split(",") if m.strip()}
        d[s1] = matches
    return d


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def write_matching_results(results: dict, s1_ids: list, path: str):
    """
    results: {s1_id: [matched_id, ...]}
    Write one row per S1.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rows = []
    s1_set = set(s1_ids)
    seen = set()
    for s1 in s1_ids:
        if s1 in seen:
            continue
        seen.add(s1)
        matches = results.get(s1, [])
        # Deduplicate
        matches = list(dict.fromkeys(matches))
        rows.append({
            "source1_entity_id": s1,
            "matched_entity_ids": ",".join(matches)
        })

    df_out = pd.DataFrame(rows, columns=["source1_entity_id", "matched_entity_ids"])
    tmp = path + ".tmp"
    df_out.to_csv(tmp, sep="\t", index=False)
    os.replace(tmp, path)
    log.info(f"Wrote matching results: {path} ({len(df_out):,} rows)")


def write_candidate_pairs(candidates: pd.DataFrame, path: str):
    """
    candidates: DataFrame with columns [source1_entity_id, candidate_entity_id, candidate_source]
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    out = candidates[["source1_entity_id", "candidate_entity_id", "candidate_source"]].copy()
    out = out.drop_duplicates(subset=["source1_entity_id", "candidate_entity_id"])
    out.to_csv(tmp, sep="\t", index=False)
    os.replace(tmp, path)
    log.info(f"Wrote candidate pairs: {path} ({len(out):,} rows)")
