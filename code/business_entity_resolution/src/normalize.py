"""
normalize.py — All text normalization for names and addresses.
Runs on CPU with multiprocessing.
"""
import re
import unicodedata
import logging
from typing import Optional
import pandas as pd
import numpy as np
from multiprocessing import Pool
from functools import partial

from . import config

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Legal suffix regex
# ---------------------------------------------------------------------------
_LEGAL_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(s) for s in config.LEGAL_SUFFIXES) + r")\b",
    flags=re.IGNORECASE,
)

# Address abbreviation regex
_ADDR_ABBREV_PATTERNS = {
    re.compile(r"\b" + re.escape(k) + r"\b", re.IGNORECASE): v
    for k, v in config.ADDR_ABBREVS.items()
}

# Numeric sequence extractor
_NUM_PATTERN = re.compile(r"\d+")

# Postal/PIN code patterns (5-digit US zip, 6-digit India PIN)
_POSTAL_PATTERN = re.compile(r"\b\d{5,6}\b")


# ---------------------------------------------------------------------------
# Core text helpers
# ---------------------------------------------------------------------------

def unicode_normalize(text: str) -> str:
    """NFKC normalize, keeps Unicode letters."""
    return unicodedata.normalize("NFKC", text)


def normalize_basic(text: str) -> str:
    """Lowercase, NFKC, collapse whitespace, minimal punctuation strip."""
    if not text:
        return ""
    text = unicode_normalize(text.strip())
    text = text.lower()
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text)
    return text


def normalize_alnum(text: str) -> str:
    """Keep only alphanumeric and spaces."""
    if not text:
        return ""
    text = normalize_basic(text)
    # Keep unicode letters/numbers, replace else with space
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def normalize_name(text: str) -> str:
    """Business-name normalization: remove punctuation, collapse suffixes."""
    if not text:
        return ""
    text = normalize_basic(text)
    # Remove punctuation except hyphens within words
    text = re.sub(r"[^\w\s\-]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def strip_legal_suffixes(text: str) -> str:
    """Remove legal suffixes from normalized name."""
    if not text:
        return ""
    result = _LEGAL_PATTERN.sub(" ", text)
    return re.sub(r"\s+", " ", result).strip()


def extract_legal_suffix(text: str) -> str:
    """Return the first matched legal suffix (for feature use)."""
    if not text:
        return ""
    m = _LEGAL_PATTERN.search(text.lower())
    return m.group(1) if m else ""


def normalize_address(text: str) -> str:
    """Address normalization: lowercase, NFKC, abbreviate."""
    if not text:
        return ""
    text = normalize_basic(text)
    # Apply abbreviations
    for pat, repl in _ADDR_ABBREV_PATTERNS.items():
        text = pat.sub(repl, text)
    text = re.sub(r"[^\w\s,\-\.]", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def address_alnum(text: str) -> str:
    """Alphanumeric-only address."""
    if not text:
        return ""
    text = normalize_address(text)
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def extract_numbers(text: str) -> list:
    """Extract all numeric sequences from text."""
    if not text:
        return []
    return _NUM_PATTERN.findall(text)


def extract_postal_codes(text: str) -> list:
    """Extract postal/PIN codes (5-6 digit sequences)."""
    if not text:
        return []
    return _POSTAL_PATTERN.findall(text)


def tokenize(text: str) -> list:
    """Simple whitespace tokenizer."""
    if not text:
        return []
    return text.split()


# ---------------------------------------------------------------------------
# Per-row normalization (applied to each DataFrame row)
# ---------------------------------------------------------------------------

def _normalize_row(row) -> dict:
    """Normalize a single row dict. Returns new fields to add."""
    name = row.get("business_name", "") or ""
    addr = row.get("business_address", "") or ""
    country = row.get("country", "") or ""

    norm_name     = normalize_name(name)
    norm_name_suf = strip_legal_suffixes(norm_name)
    norm_addr     = normalize_address(addr)
    addr_alnum    = address_alnum(addr)
    name_basic    = normalize_basic(name)
    addr_basic    = normalize_basic(addr)

    nums_name     = extract_numbers(norm_name)
    nums_addr     = extract_numbers(norm_addr)
    postals       = extract_postal_codes(addr)

    legal_suf     = extract_legal_suffix(name_basic)
    country_norm  = normalize_basic(country)

    return {
        "norm_name":        norm_name,
        "norm_name_stripped": norm_name_suf,
        "norm_addr":        norm_addr,
        "addr_alnum":       addr_alnum,
        "name_basic":       name_basic,
        "addr_basic":       addr_basic,
        "name_numbers":     ",".join(nums_name),
        "addr_numbers":     ",".join(nums_addr),
        "postal_codes":     ",".join(postals),
        "legal_suffix":     legal_suf,
        "country_norm":     country_norm,
        "name_missing":     int(name == ""),
        "addr_missing":     int(addr == ""),
    }


def _worker_normalize(rows_chunk):
    """Normalize a chunk of rows (each row is a dict)."""
    return [_normalize_row(r) for r in rows_chunk]


def normalize_dataframe(df: pd.DataFrame, n_workers: int = None) -> pd.DataFrame:
    """
    Vectorized normalization of a full DataFrame.
    Returns df with extra normalized columns.
    """
    if n_workers is None:
        n_workers = config.NUM_WORKERS

    log.info(f"  Normalizing {len(df):,} records with {n_workers} workers...")

    rows = df[["business_name", "business_address", "country"]].to_dict("records")

    chunk_size = max(1, len(rows) // (n_workers * 4))
    chunks = [rows[i:i+chunk_size] for i in range(0, len(rows), chunk_size)]

    if n_workers > 1:
        with Pool(n_workers) as pool:
            results = pool.map(_worker_normalize, chunks)
    else:
        results = [_worker_normalize(c) for c in chunks]

    flat = [item for sub in results for item in sub]
    norm_df = pd.DataFrame(flat, index=df.index)

    out = pd.concat([df, norm_df], axis=1)
    log.info(f"  Normalization done: {len(out):,} rows")
    return out
