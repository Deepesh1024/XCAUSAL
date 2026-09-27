"""File I/O: read source/ground-truth TSVs, write submission TSVs.

Every read uses sep="\t" and keeps all fields as strings (empty -> "").
"""
import csv
import os
import time

import pandas as pd

COLS = ["entity_id", "business_name", "business_address", "country"]


class Timer:
    """Context manager that prints how long a stage took."""

    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.t0 = time.time()
        print(f"[{self.name}] start", flush=True)
        return self

    def __exit__(self, *exc):
        print(f"[{self.name}] done in {time.time() - self.t0:.1f}s", flush=True)


def read_source(path):
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False,
                     quoting=csv.QUOTE_NONE, on_bad_lines="warn")
    for c in COLS:
        if c not in df.columns:
            df[c] = ""
    df = df[COLS].fillna("")
    return df.reset_index(drop=True)


def read_split(data_dir, split):
    """Return (s1, s2, s3) DataFrames for split in {'train', 'test'}."""
    d = os.path.join(data_dir, split)
    return tuple(read_source(os.path.join(d, f"{split}_source{i}.tsv")) for i in (1, 2, 3))


def read_ground_truth(path):
    """Return dict: s1_id -> set of matched S2/S3 ids (empty set for singletons)."""
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, quoting=csv.QUOTE_NONE)
    gt = {}
    for s1, m in zip(df["source1_entity_id"], df["matched_entity_ids"]):
        gt[s1] = {x.strip() for x in m.split(",") if x.strip()}
    return gt


def write_id_lists(path, s1_ids, mapping, col_name):
    """Write one row per S1 id; empty list when no ids. Dedupes, keeps order."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(f"source1_entity_id\t{col_name}\n")
        for s1 in s1_ids:
            ids = list(dict.fromkeys(mapping.get(s1, [])))
            f.write(f"{s1}\t{','.join(ids)}\n")
