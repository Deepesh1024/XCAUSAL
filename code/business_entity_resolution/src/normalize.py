"""
normalize.py — All text normalization for names and addresses.
Runs on CPU with multiprocessing.
"""
import re
import unicodedata
import logging
import time
from typing import Optional
import pandas as pd
import numpy as np

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
# Vectorized normalization (NO apply/iterrows — uses pandas .str ops)
# ---------------------------------------------------------------------------

def _vec_normalize_basic(series: pd.Series) -> pd.Series:
    """Vectorized lowercase + NFKC + whitespace collapse."""
    s = series.fillna("").astype(str)
    # Python's str.lower + unicode normalize via str accessor
    s = s.str.lower().str.strip()
    s = s.str.normalize("NFKC")
    s = s.str.replace(r"\s+", " ", regex=True)
    return s


def _vec_strip_legal(series: pd.Series) -> pd.Series:
    """Vectorized removal of legal suffixes."""
    pat = r"\b(" + "|".join(re.escape(s) for s in config.LEGAL_SUFFIXES) + r")\b"
    result = series.str.replace(pat, " ", regex=True, case=False)
    return result.str.replace(r"\s+", " ", regex=True).str.strip()


def _vec_normalize_name(series: pd.Series) -> pd.Series:
    """Vectorized name normalization."""
    s = _vec_normalize_basic(series)
    s = s.str.replace(r"[^\w\s\-]", " ", regex=True)
    return s.str.replace(r"\s+", " ", regex=True).str.strip()


def _vec_normalize_addr(series: pd.Series) -> pd.Series:
    """Vectorized address normalization with abbreviation expansion."""
    s = _vec_normalize_basic(series)
    for word, abbrev in config.ADDR_ABBREVS.items():
        s = s.str.replace(r"\b" + re.escape(word) + r"\b", abbrev, regex=True)
    s = s.str.replace(r"[^\w\s,\.\-]", " ", regex=True)
    return s.str.replace(r"\s+", " ", regex=True).str.strip()


def _vec_extract_numbers(series: pd.Series) -> pd.Series:
    """Extract comma-joined numeric sequences from each cell."""
    return series.str.findall(r"\d+").apply(
        lambda x: ",".join(x) if isinstance(x, list) else ""
    )


def _vec_extract_postal(series: pd.Series) -> pd.Series:
    """Extract comma-joined 5-6 digit postal codes."""
    return series.str.findall(r"\b\d{5,6}\b").apply(
        lambda x: ",".join(x) if isinstance(x, list) else ""
    )


def _vec_extract_legal_suffix(series: pd.Series) -> pd.Series:
    """Extract first legal suffix per name."""
    pat = r"\b(" + "|".join(re.escape(s) for s in config.LEGAL_SUFFIXES) + r")\b"
    return series.str.extract(pat, flags=re.IGNORECASE, expand=False).fillna("")


def normalize_dataframe(df: pd.DataFrame, n_workers: int = None) -> pd.DataFrame:
    """
    Fully vectorized normalization using pandas .str operations.
    No apply(), no iterrows(), no Python row loops.
    """
    log.info(f"  Normalizing {len(df):,} records (vectorized)...")
    t0 = time.time()

    out = df.copy()
    name_raw = out["business_name"].fillna("").astype(str)
    addr_raw = out["business_address"].fillna("").astype(str)
    ctr_raw  = out["country"].fillna("").astype(str)

    out["norm_name"]          = _vec_normalize_name(name_raw)
    out["norm_name_stripped"] = _vec_strip_legal(out["norm_name"])
    out["norm_addr"]          = _vec_normalize_addr(addr_raw)
    out["addr_alnum"]         = out["norm_addr"].str.replace(r"[^\w\s]", " ", regex=True).str.strip()
    out["name_basic"]         = _vec_normalize_basic(name_raw)
    out["addr_basic"]         = _vec_normalize_basic(addr_raw)
    out["name_numbers"]       = _vec_extract_numbers(out["norm_name"])
    out["addr_numbers"]       = _vec_extract_numbers(out["norm_addr"])
    out["postal_codes"]       = _vec_extract_postal(addr_raw)
    out["legal_suffix"]       = _vec_extract_legal_suffix(name_raw)
    out["country_norm"]       = _vec_normalize_basic(ctr_raw)
    out["name_missing"]       = (name_raw == "").astype(np.int8)
    out["addr_missing"]       = (addr_raw == "").astype(np.int8)

    elapsed = time.time() - t0
    rps = len(df) / elapsed if elapsed > 0 else 0
    log.info(f"  Normalization done: {len(out):,} rows in {elapsed:.2f}s ({rps:,.0f} rows/sec)")
    return out
