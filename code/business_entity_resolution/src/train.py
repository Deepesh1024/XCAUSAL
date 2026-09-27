"""
train.py — LightGBM training with precision-heavy threshold optimization for F0.5.
"""
import os
import logging
import pickle
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split

from . import config
from .features import FEATURE_COLS
from .io import atomic_save_pickle, atomic_save_json

log = logging.getLogger(__name__)


def compute_f05(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute F0.5 score (precision-weighted)."""
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))

    precision = tp / (tp + fp + 1e-9)
    recall    = tp / (tp + fn + 1e-9)

    beta = 0.5
    f05  = (1 + beta**2) * precision * recall / (beta**2 * precision + recall + 1e-9)
    return float(f05)


def find_best_threshold(
    y_true:  np.ndarray,
    y_proba: np.ndarray,
) -> Tuple[float, float]:
    """
    Two-stage threshold search optimizing F0.5.
    Returns (best_threshold, best_f05).
    """
    # Coarse search
    best_thr, best_f05 = 0.5, 0.0
    thresholds = np.arange(
        config.THRESHOLD_COARSE_START,
        config.THRESHOLD_COARSE_END + 0.01,
        config.THRESHOLD_COARSE_STEP,
    )
    for thr in thresholds:
        y_pred = (y_proba >= thr).astype(int)
        f05    = compute_f05(y_true, y_pred)
        if f05 > best_f05:
            best_f05, best_thr = f05, thr

    log.info(f"  Coarse search: best_thr={best_thr:.3f}, F0.5={best_f05:.4f}")

    # Fine search
    fine_thresholds = np.arange(
        max(0.01, best_thr - config.THRESHOLD_FINE_HALF_WINDOW),
        min(0.99, best_thr + config.THRESHOLD_FINE_HALF_WINDOW + 0.001),
        config.THRESHOLD_FINE_STEP,
    )
    for thr in fine_thresholds:
        y_pred = (y_proba >= thr).astype(int)
        f05    = compute_f05(y_true, y_pred)
        if f05 > best_f05:
            best_f05, best_thr = f05, thr

    log.info(f"  Fine search:   best_thr={best_thr:.3f}, F0.5={best_f05:.4f}")
    return best_thr, best_f05


def train_lightgbm(
    feat_df:        pd.DataFrame,
    model_dir:      str = config.CKPT_MODEL,
    n_estimators:   int = None,
) -> Tuple[lgb.Booster, float, float]:
    """
    Train LightGBM binary classifier.
    Returns (booster, best_threshold, val_f05).
    """
    os.makedirs(model_dir, exist_ok=True)

    # Ensure all feature columns exist
    for col in FEATURE_COLS:
        if col not in feat_df.columns:
            feat_df[col] = 0.0

    X = feat_df[FEATURE_COLS].values.astype(np.float32)
    y = feat_df["label"].values.astype(np.int32)

    log.info(f"Training LightGBM: {len(X):,} pairs | "
             f"pos={y.sum():,} ({y.mean()*100:.1f}%)")

    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y,
        test_size=config.LGB_VAL_FRACTION,
        random_state=config.RANDOM_SEED,
        stratify=y,
    )

    params = dict(config.LGB_PARAMS)
    if n_estimators:
        params["n_estimators"] = n_estimators

    # Class weight
    pos_weight = max(1, (y_tr == 0).sum() / max((y_tr == 1).sum(), 1))
    params["scale_pos_weight"] = pos_weight
    log.info(f"  scale_pos_weight={pos_weight:.2f}")

    dtrain = lgb.Dataset(X_tr, label=y_tr)
    dval   = lgb.Dataset(X_val, label=y_val, reference=dtrain)

    callbacks = [
        lgb.early_stopping(config.LGB_EARLY_STOPPING, verbose=False),
        lgb.log_evaluation(period=200),
    ]

    booster = lgb.train(
        params,
        dtrain,
        valid_sets=[dtrain, dval],
        valid_names=["train", "val"],
        callbacks=callbacks,
    )

    val_proba = booster.predict(X_val)
    best_thr, best_f05 = find_best_threshold(y_val, val_proba)

    log.info(f"LightGBM training done | val F0.5={best_f05:.4f} | thr={best_thr:.4f}")

    # Save model and threshold
    model_path = os.path.join(model_dir, "lgb_model.pkl")
    meta_path  = os.path.join(model_dir, "meta.json")
    atomic_save_pickle(booster, model_path)
    atomic_save_json({"threshold": best_thr, "val_f05": best_f05}, meta_path)

    # Feature importance
    imp = pd.DataFrame({
        "feature":    FEATURE_COLS,
        "importance": booster.feature_importance(importance_type="gain"),
    }).sort_values("importance", ascending=False)
    log.info(f"Top-10 features:\n{imp.head(10).to_string(index=False)}")

    return booster, best_thr, best_f05
