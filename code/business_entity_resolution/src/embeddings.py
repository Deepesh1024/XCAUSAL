"""
embeddings.py — BGE-M3 dense embeddings with GPU batching, caching, and FAISS retrieval.
"""
import os
import logging
import time
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from . import config
from .io import checkpoint_exists

log = logging.getLogger(__name__)


def _get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def format_record(name: str, address: str, country: str) -> str:
    """Serialize a record for BGE input."""
    name    = name    or ""
    address = address or ""
    country = country or ""
    return f"[NAME] {name} [ADDRESS] {address} [COUNTRY] {country}"


def embed_texts(
    texts: List[str],
    model,
    tokenizer,
    batch_size: int  = config.BGE_BATCH_SIZE,
    max_length: int  = config.BGE_MAX_SEQ_LEN,
    device             = None,
) -> np.ndarray:
    """
    Embed a list of texts using a HuggingFace model.
    Returns L2-normalized float16 embeddings as numpy array.
    """
    if device is None:
        device = _get_device()

    all_embeddings = []
    n = len(texts)

    for start in range(0, n, batch_size):
        batch = texts[start:start + batch_size]
        enc   = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)

        with torch.inference_mode():
            out = model(**enc)
            # Use CLS token or mean pool
            if hasattr(out, "last_hidden_state"):
                emb = out.last_hidden_state[:, 0, :]   # CLS
            else:
                emb = out.pooler_output

            emb = F.normalize(emb, p=2, dim=-1)

            if config.BGE_FP16:
                emb = emb.half()

            all_embeddings.append(emb.cpu().numpy())

        if (start // batch_size) % 20 == 0 and start > 0:
            log.info(f"  Embedded {start + len(batch):,}/{n:,}")

    return np.concatenate(all_embeddings, axis=0)


def load_bge_model():
    """Load BGE-M3 model and tokenizer."""
    from transformers import AutoTokenizer, AutoModel
    log.info(f"Loading BGE model: {config.BGE_MODEL_NAME}")
    device    = _get_device()
    tokenizer = AutoTokenizer.from_pretrained(config.BGE_MODEL_NAME)
    model     = AutoModel.from_pretrained(config.BGE_MODEL_NAME)
    if config.BGE_FP16:
        model = model.half()
    model = model.to(device).eval()
    log.info(f"BGE model loaded on {device}")
    return model, tokenizer, device


class BGEEmbedder:
    """
    Manages embedding computation, caching, and cosine retrieval.
    """

    def __init__(self, cache_dir: str = config.CKPT_BGE):
        self.cache_dir = cache_dir
        self.model     = None
        self.tokenizer = None
        self.device    = None
        os.makedirs(cache_dir, exist_ok=True)

    def _load_model(self):
        if self.model is None:
            self.model, self.tokenizer, self.device = load_bge_model()

    def _cache_path(self, name: str) -> str:
        return os.path.join(self.cache_dir, f"{name}.npy")

    def compute_or_load(self, df: pd.DataFrame, name: str) -> np.ndarray:
        """Compute embeddings for df or load from cache."""
        path = self._cache_path(name)
        if os.path.exists(path):
            log.info(f"  Loading cached embeddings: {path}")
            return np.load(path, mmap_mode="r")

        self._load_model()

        texts = [
            format_record(
                row.get("business_name", "") or "",
                row.get("business_address", "") or "",
                row.get("country", "") or "",
            )
            for row in df.to_dict("records")
        ]

        log.info(f"  Computing {len(texts):,} BGE embeddings for '{name}'...")
        embs = embed_texts(texts, self.model, self.tokenizer, device=self.device)

        tmp  = path + ".tmp.npy"
        np.save(tmp, embs)
        os.replace(tmp, path)
        log.info(f"  Saved embeddings: {path} shape={embs.shape}")
        return embs

    def cosine_retrieval(
        self,
        query_embs:  np.ndarray,
        corpus_embs: np.ndarray,
        corpus_ids:  np.ndarray,
        top_k:       int,
        batch_size:  int = 512,
    ) -> List[List[Tuple[str, float]]]:
        """
        For each query embedding, find top_k most similar corpus records.
        Batched to avoid OOM.
        """
        n_q = query_embs.shape[0]
        results = []

        # Ensure float32 for matmul precision
        q_f32 = query_embs.astype(np.float32)
        c_f32 = corpus_embs.astype(np.float32)

        for start in range(0, n_q, batch_size):
            batch = q_f32[start:start + batch_size]
            sims  = batch @ c_f32.T          # (batch, n_corpus)
            top_k_actual = min(top_k, sims.shape[1])
            idx   = np.argpartition(sims, -top_k_actual, axis=1)[:, -top_k_actual:]
            for i, row_idx in enumerate(idx):
                row_scores = sims[i, row_idx]
                order      = np.argsort(-row_scores)
                results.append([
                    (corpus_ids[row_idx[j]], float(row_scores[j]))
                    for j in order
                ])

            if start % (batch_size * 20) == 0 and start > 0:
                log.info(f"  Dense retrieval: {start:,}/{n_q:,}")

        return results

    def free_model(self):
        """Release GPU memory."""
        if self.model is not None:
            del self.model
            self.model = None
            torch.cuda.empty_cache()
            log.info("BGE model freed from GPU")
