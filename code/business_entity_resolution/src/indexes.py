"""
indexes.py — Build inverted indexes and TF-IDF matrices for sparse retrieval.
"""
import os
import logging
import pickle
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Set

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy.sparse as sp

from . import config
from .io import atomic_save_pickle, atomic_load_pickle, checkpoint_exists

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Rare-token inverted index
# ---------------------------------------------------------------------------

class RareTokenIndex:
    """
    Builds an inverted index of rare tokens for fast candidate retrieval.
    Tokens that appear in > RARE_TOKEN_MAX_DF fraction of documents
    are considered 'common' and excluded.
    """
    def __init__(self, max_df: float = config.RARE_TOKEN_MAX_DF,
                 top_k: int = config.RARE_TOKEN_TOP_K):
        self.max_df   = max_df
        self.top_k    = top_k
        self.index: Dict[str, Set[str]] = defaultdict(set)
        self.idf: Dict[str, float] = {}
        self._n_docs  = 0

    def build(self, df: pd.DataFrame, id_col: str = "entity_id",
              text_col: str = "norm_name_stripped"):
        log.info(f"  Building rare-token index on {len(df):,} rows...")
        self._n_docs = len(df)
        df_freq: Counter = Counter()

        # Count document frequencies
        for text in df[text_col]:
            tokens = set(str(text).split()) if text else set()
            df_freq.update(tokens)

        max_count = int(self.max_df * self._n_docs)

        # Compute IDF (log(N / df))
        self.idf = {
            tok: np.log(self._n_docs / cnt)
            for tok, cnt in df_freq.items()
            if cnt <= max_count and len(tok) >= 3
        }

        # Build inverted index
        for row in df.itertuples(index=False):
            eid  = getattr(row, id_col)
            text = getattr(row, text_col) or ""
            tokens = set(text.split())
            for tok in tokens:
                if tok in self.idf:
                    self.index[tok].add(eid)

        log.info(f"  Rare-token index: {len(self.idf):,} rare tokens, "
                 f"{sum(len(v) for v in self.index.values()):,} entries")

    def query(self, text: str, top_k: int = None) -> List[Tuple[str, float]]:
        """Return (entity_id, score) list for the given text."""
        top_k = top_k or self.top_k
        tokens = str(text).split() if text else []

        # Score by IDF-weighted token overlap
        scores: Dict[str, float] = defaultdict(float)
        for tok in tokens:
            if tok in self.idf:
                for eid in self.index.get(tok, []):
                    scores[eid] += self.idf[tok]

        if not scores:
            return []

        ranked = sorted(scores.items(), key=lambda x: -x[1])
        return ranked[:top_k]


# ---------------------------------------------------------------------------
# Exact-name index
# ---------------------------------------------------------------------------

class ExactNameIndex:
    """Maps normalized name -> set of entity IDs."""

    def __init__(self):
        self.index: Dict[str, Set[str]] = defaultdict(set)

    def build(self, df: pd.DataFrame, id_col="entity_id", text_col="norm_name"):
        log.info(f"  Building exact-name index on {len(df):,} rows...")
        for row in df.itertuples(index=False):
            name = getattr(row, text_col) or ""
            if name:
                self.index[name].add(getattr(row, id_col))
        log.info(f"  Exact-name index: {len(self.index):,} unique names")

    def query(self, name: str) -> Set[str]:
        return self.index.get(name, set())


# ---------------------------------------------------------------------------
# Address / postal index
# ---------------------------------------------------------------------------

class AddressIndex:
    """
    Separate indexes for postal codes and rare address tokens.
    """

    def __init__(self, max_df: float = 0.005):
        self.max_df = max_df
        self.postal_index: Dict[str, Set[str]]       = defaultdict(set)
        self.addr_token_index: Dict[str, Set[str]]   = defaultdict(set)
        self.number_index: Dict[str, Set[str]]       = defaultdict(set)
        self._common_addr_tokens: Set[str]           = set()

    def build(self, df: pd.DataFrame, id_col="entity_id"):
        log.info(f"  Building address index on {len(df):,} rows...")
        n_docs = len(df)
        max_count = int(self.max_df * n_docs)

        # Compute address token frequencies
        tok_freq: Counter = Counter()
        for text in df["norm_addr"]:
            tok_freq.update(str(text).split() if text else [])

        self._common_addr_tokens = {t for t, c in tok_freq.items() if c > max_count}

        for row in df.itertuples(index=False):
            eid     = getattr(row, id_col)
            postals = getattr(row, "postal_codes") or ""
            addr    = getattr(row, "norm_addr") or ""
            nums    = getattr(row, "addr_numbers") or ""

            for p in postals.split(","):
                p = p.strip()
                if p:
                    self.postal_index[p].add(eid)

            for tok in str(addr).split():
                if len(tok) >= 4 and tok not in self._common_addr_tokens:
                    self.addr_token_index[tok].add(eid)

            for n in nums.split(","):
                n = n.strip()
                if n and len(n) >= 2:
                    self.number_index[n].add(eid)

        log.info(f"  Address index: "
                 f"{len(self.postal_index):,} postal codes, "
                 f"{len(self.addr_token_index):,} addr tokens, "
                 f"{len(self.number_index):,} numbers")

    def query_postal(self, postal_codes_str: str) -> Set[str]:
        result = set()
        for p in postal_codes_str.split(","):
            p = p.strip()
            if p:
                result.update(self.postal_index.get(p, set()))
        return result

    def query_addr_tokens(self, norm_addr: str, top_k: int = 20) -> List[Tuple[str, float]]:
        scores: Dict[str, int] = defaultdict(int)
        for tok in str(norm_addr).split():
            if len(tok) >= 4 and tok not in self._common_addr_tokens:
                for eid in self.addr_token_index.get(tok, []):
                    scores[eid] += 1
        ranked = sorted(scores.items(), key=lambda x: -x[1])
        return ranked[:top_k]

    def query_numbers(self, addr_numbers_str: str) -> Set[str]:
        result = set()
        for n in addr_numbers_str.split(","):
            n = n.strip()
            if n and len(n) >= 2:
                result.update(self.number_index.get(n, set()))
        return result


# ---------------------------------------------------------------------------
# TF-IDF char n-gram index
# ---------------------------------------------------------------------------

class CharNgramIndex:
    """
    Sparse TF-IDF char n-gram matrix for approximate retrieval.
    Uses batched sparse matrix multiplication.
    """

    def __init__(self,
                 max_features: int = config.TFIDF_MAX_FEATURES,
                 ngram_range: tuple = config.TFIDF_NGRAM_RANGE,
                 min_df: int = config.TFIDF_MIN_DF):
        self.vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=ngram_range,
            min_df=min_df,
            max_features=max_features,
            dtype=np.float32,
            sublinear_tf=True,
        )
        self.matrix: sp.csr_matrix = None
        self.ids: np.ndarray       = None

    def build(self, df: pd.DataFrame, text_col: str = "norm_name_stripped",
              id_col: str = "entity_id"):
        log.info(f"  Fitting TF-IDF char n-gram on {len(df):,} rows...")
        texts = df[text_col].fillna("").tolist()
        self.matrix = self.vectorizer.fit_transform(texts)
        self.ids    = df[id_col].values
        log.info(f"  TF-IDF matrix shape: {self.matrix.shape}")

    def query_batch(self, query_texts: List[str],
                    top_k: int = config.CANDIDATE_K) -> List[List[Tuple[str, float]]]:
        """
        For each query text, return top_k (entity_id, score) pairs.
        Uses sparse dot-product in chunks.
        """
        q_mat = self.vectorizer.transform(query_texts)  # (n_queries, n_feats)
        # Sparse dot product: (n_queries, n_corpus)
        sims = (q_mat @ self.matrix.T).toarray()        # float32 numpy

        results = []
        for row in sims:
            top_idx = np.argpartition(row, -top_k)[-top_k:]
            top_idx = top_idx[np.argsort(-row[top_idx])]
            results.append([(self.ids[i], float(row[i])) for i in top_idx if row[i] > 0])
        return results


# ---------------------------------------------------------------------------
# Name frequency table (for generic-name feature)
# ---------------------------------------------------------------------------

def build_name_frequency_table(dfs: List[pd.DataFrame]) -> Dict[str, int]:
    """Build {normalized_name: count} across all sources."""
    freq: Counter = Counter()
    for df in dfs:
        col = "norm_name_stripped" if "norm_name_stripped" in df.columns else "norm_name"
        for name in df[col]:
            if name:
                freq[name] += 1
    return dict(freq)


# ---------------------------------------------------------------------------
# Serialize / deserialize all indexes
# ---------------------------------------------------------------------------

def save_indexes(indexes: dict, path_prefix: str):
    os.makedirs(path_prefix, exist_ok=True)
    for name, obj in indexes.items():
        atomic_save_pickle(obj, os.path.join(path_prefix, f"{name}.pkl"))
    log.info(f"Saved indexes to {path_prefix}")


def load_indexes(names: List[str], path_prefix: str) -> dict:
    return {n: atomic_load_pickle(os.path.join(path_prefix, f"{n}.pkl"))
            for n in names}
