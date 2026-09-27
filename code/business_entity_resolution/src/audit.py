#!/usr/bin/env python3
"""
audit.py — Static + Runtime audit and preflight for the production pipeline.

Runs BEFORE any expensive stage. Checks:
  1. System report (GPU, RAM, CPU, disk)
  2. Static code audit (forbidden patterns)
  3. Complexity / memory bounds check
  4. GPU smoke test
  5. Normalization vectorization check
  6. Micro-test (1K S1 / 5K S2 / 5K S3) — end-to-end
  7. Throughput benchmarks
  8. Memory extrapolation

Usage:
    python -m code.business_entity_resolution.src.audit [--skip-micro]
"""
import argparse
import ast
import gc
import hashlib
import logging
import os
import re
import sys
import time
import traceback
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import psutil

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("audit")

REPO_ROOT = Path(__file__).parents[4]
SRC_DIR   = Path(__file__).parent

PASS = "✅ PASS"
FAIL = "❌ FAIL"
WARN = "⚠️  WARN"

audit_results: List[Tuple[str, str, str]] = []


def record(check: str, status: str, detail: str = ""):
    audit_results.append((check, status, detail))
    icon = status
    log.info(f"  {icon}  {check}: {detail}")


# ============================================================
# 1. SYSTEM REPORT
# ============================================================

def system_report():
    log.info("\n" + "="*60)
    log.info("SYSTEM REPORT")
    log.info("="*60)

    # Python
    log.info(f"  Python:      {sys.version}")

    # CPU
    n_cpu = psutil.cpu_count(logical=True)
    n_phy = psutil.cpu_count(logical=False)
    log.info(f"  CPU cores:   {n_phy} physical / {n_cpu} logical")

    # RAM
    ram = psutil.virtual_memory()
    log.info(f"  RAM total:   {ram.total/1e9:.1f} GB")
    log.info(f"  RAM avail:   {ram.available/1e9:.1f} GB")

    # Disk
    disk = psutil.disk_usage(str(REPO_ROOT))
    log.info(f"  Disk total:  {disk.total/1e9:.1f} GB")
    log.info(f"  Disk free:   {disk.free/1e9:.1f} GB")

    # GPU
    try:
        import torch
        if torch.cuda.is_available():
            name  = torch.cuda.get_device_name(0)
            vram  = torch.cuda.get_device_properties(0).total_memory / 1e9
            log.info(f"  GPU:         {name}")
            log.info(f"  GPU VRAM:    {vram:.1f} GB")
            log.info(f"  CUDA ver:    {torch.version.cuda}")
            log.info(f"  PyTorch:     {torch.__version__}")
            record("GPU available", PASS, f"{name} {vram:.1f}GB")
        else:
            log.warning("  GPU:         NOT AVAILABLE — will use CPU fallback")
            record("GPU available", WARN, "CUDA not available. CPU fallback will be used.")
    except ImportError:
        record("PyTorch", FAIL, "torch not installed")

    # cuDF
    try:
        import cudf
        log.info(f"  cuDF:        {cudf.__version__}")
        record("cuDF installed", PASS, cudf.__version__)
    except ImportError:
        log.info(f"  cuDF:        NOT INSTALLED (ok — CPU fallback)")
        record("cuDF installed", WARN, "Not installed. CPU vectorized ops will be used.")

    # LightGBM
    try:
        import lightgbm as lgb
        log.info(f"  LightGBM:    {lgb.__version__}")
        record("LightGBM installed", PASS, lgb.__version__)
    except ImportError:
        record("LightGBM installed", FAIL, "lightgbm not installed")

    # Transformers
    try:
        import transformers
        log.info(f"  Transformers: {transformers.__version__}")
        record("Transformers installed", PASS, transformers.__version__)
    except ImportError:
        record("Transformers installed", FAIL, "transformers not installed")

    # Dataset paths
    log.info("\n  Dataset files:")
    from . import config
    for label, path in [
        ("TRAIN_S1", config.TRAIN_S1), ("TRAIN_S2", config.TRAIN_S2),
        ("TRAIN_S3", config.TRAIN_S3), ("TRAIN_GT", config.TRAIN_GT),
        ("TEST_S1",  config.TEST_S1),  ("TEST_S2",  config.TEST_S2),
        ("TEST_S3",  config.TEST_S3),
    ]:
        exists = os.path.exists(path)
        size   = os.path.getsize(path) / 1e9 if exists else 0
        log.info(f"    {label:10s}: {'EXISTS' if exists else 'MISSING':8s} {size:.2f}GB  {path}")
        if "TEST" in label:
            record(f"File {label}", PASS if exists else WARN,
                   "exists" if exists else "Not found (test data may not be available yet)")
        else:
            record(f"File {label}", PASS if exists else FAIL,
                   "exists" if exists else f"MISSING: {path}")


# ============================================================
# 2. STATIC CODE AUDIT
# ============================================================

FORBIDDEN_PATTERNS = [
    # Pattern, severity, note
    (r"\.iterrows\(\)", "FAIL", "iterrows() on millions of rows"),
    (r"\.to_dict\(['\"]index['\"]",   "FAIL", "to_dict('index') on large DF"),
    (r"\.to_dict\(['\"]records['\"]", "FAIL", "to_dict('records') on large DF"),
    (r"\.apply\(lambda",              "WARN", "apply(lambda) may be slow on large DF"),
    (r"for\s+_?,?\s*\w+\s+in\s+\w+\.iterrows\(\)", "FAIL", "iterrows loop"),
    (r"for\s+\w+\s+in\s+\w+\.itertuples\(\)", "WARN", "itertuples loop — check scale"),
    (r"groupby.*\.apply\(",            "WARN", "groupby.apply with custom fn"),
]

# Patterns that are ONLY forbidden if NOT bounded to small subset
SCALE_FORBIDDEN = [
    r"\.apply\(\w+,\s*axis=1\)",
    r"for\s+row\s+in\s+df",
]


def static_audit():
    log.info("\n" + "="*60)
    log.info("STATIC CODE AUDIT")
    log.info("="*60)

    py_files = list(SRC_DIR.rglob("*.py"))
    log.info(f"  Scanning {len(py_files)} Python files in {SRC_DIR}")

    failures = []
    warnings = []

    for filepath in py_files:
        # Skip audit.py itself and eda/
        if "audit" in filepath.name or "eda" in str(filepath):
            continue

        with open(filepath) as f:
            source = f.read()
            lines  = source.splitlines()

        for pat, severity, note in FORBIDDEN_PATTERNS:
            for i, line in enumerate(lines, 1):
                # Skip comment lines
                stripped = line.strip()
                if stripped.startswith("#"):
                    continue
                if re.search(pat, line):
                    msg = f"{filepath.relative_to(REPO_ROOT)}:{i}: {note}"
                    if severity == "FAIL":
                        failures.append(msg)
                        record(f"Forbidden pattern: {pat[:30]}", FAIL, msg)
                    else:
                        warnings.append(msg)
                        record(f"Warning pattern: {pat[:30]}", WARN, msg)

    # Also check for N×M patterns
    for filepath in py_files:
        if "audit" in filepath.name or "eda" in str(filepath):
            continue
        with open(filepath) as f:
            source = f.read()
        # Nested for loops over dataframes (rough heuristic)
        if re.search(r"for.*s1.*:\s*\n.*for.*s2", source, re.IGNORECASE | re.MULTILINE):
            failures.append(f"{filepath.name}: Possible nested S1×S2 loop")
            record("N×M Cartesian loop", FAIL, str(filepath.name))

    if not failures and not warnings:
        record("Static audit", PASS, f"No forbidden patterns in {len(py_files)} files")
    elif not failures:
        record("Static audit", WARN, f"{len(warnings)} warnings (no hard failures)")
    else:
        record("Static audit", FAIL, f"{len(failures)} forbidden patterns found")

    return len(failures) == 0


# ============================================================
# 3. COMPLEXITY & MEMORY BOUNDS
# ============================================================

def complexity_audit():
    log.info("\n" + "="*60)
    log.info("COMPLEXITY & MEMORY BOUNDS")
    log.info("="*60)

    S1, S2, S3 = 2_206_821, 5_034_616, 5_285_603
    K = 50

    log.info(f"  S1={S1:,}  S2={S2:,}  S3={S3:,}")
    log.info(f"  S1×S2 = {S1*S2/1e9:.1f}B pairs  ← FORBIDDEN")
    log.info(f"  S1×S3 = {S1*S3/1e9:.1f}B pairs  ← FORBIDDEN")
    log.info(f"  S1×K  = {S1*K/1e6:.1f}M candidate pairs  ← OK")

    # Memory estimates
    # TF-IDF sparse matrix: S2+S3 × 2M features, ~5% non-zero
    corpus = S2 + S3
    vocab  = 2_000_000
    nnz_est = corpus * 50  # estimated ~50 non-zero per doc
    sparse_mem_gb = nnz_est * (4 + 4) / 1e9  # float32 data + int32 indices
    log.info(f"\n  TF-IDF matrix ({corpus:,} docs × {vocab:,} vocab):")
    log.info(f"    Est. nnz: {nnz_est/1e6:.0f}M")
    log.info(f"    Est. RAM: {sparse_mem_gb:.1f} GB")

    ram_avail = psutil.virtual_memory().available / 1e9
    if sparse_mem_gb > ram_avail * 0.5:
        record("TF-IDF memory", WARN,
               f"Est {sparse_mem_gb:.1f}GB > 50% of available {ram_avail:.1f}GB. "
               f"Reduce max_features if needed.")
    else:
        record("TF-IDF memory", PASS,
               f"Est {sparse_mem_gb:.1f}GB < 50% of {ram_avail:.1f}GB available")

    # Feature DataFrame memory
    n_cands  = S1 * K
    n_feats  = 38
    feat_mem = n_cands * n_feats * 4 / 1e9  # float32
    log.info(f"\n  Feature DataFrame ({n_cands/1e6:.0f}M rows × {n_feats} cols):")
    log.info(f"    Est. RAM: {feat_mem:.1f} GB")
    if feat_mem > ram_avail * 0.4:
        record("Feature DataFrame memory", WARN,
               f"Est {feat_mem:.1f}GB. Consider chunked processing.")
    else:
        record("Feature DataFrame memory", PASS, f"Est {feat_mem:.1f}GB")

    # BGE embedding memory
    # Embedding only S1 + candidate pool records (not full corpus)
    s1_emb_gb   = S1 * 1024 * 2 / 1e9      # 1024-dim FP16
    cand_pool_gb = corpus * 0.1 * 1024 * 2 / 1e9  # 10% in candidate pool
    log.info(f"\n  BGE embeddings:")
    log.info(f"    S1 embeddings: {s1_emb_gb:.1f} GB (FP16)")
    log.info(f"    Candidate pool (~10% corpus): {cand_pool_gb:.1f} GB")
    record("BGE memory (S1 only)", PASS, f"{s1_emb_gb:.1f}GB FP16")

    # Inverted index memory (rough)
    inv_idx_gb = corpus * 20 / 1e9  # ~20 bytes overhead per record in index
    log.info(f"\n  Inverted index est: {inv_idx_gb:.1f} GB")
    record("Complexity bounds", PASS, f"K={K} candidates, ~{S1*K/1e6:.0f}M pairs max")


# ============================================================
# 4. GPU SMOKE TEST
# ============================================================

def gpu_smoke_test():
    log.info("\n" + "="*60)
    log.info("GPU SMOKE TEST")
    log.info("="*60)

    try:
        import torch
        if not torch.cuda.is_available():
            record("GPU smoke test", WARN, "CUDA not available — CPU fallback")
            return True

        device = torch.device("cuda:0")

        # Small matmul
        a = torch.randn(1024, 1024, device=device, dtype=torch.float16)
        b = torch.randn(1024, 1024, device=device, dtype=torch.float16)
        with torch.inference_mode():
            c = a @ b
        assert c.shape == (1024, 1024), "Matmul shape wrong"
        del a, b, c
        torch.cuda.empty_cache()

        vram_used = torch.cuda.memory_allocated() / 1e9
        record("GPU smoke test", PASS, f"FP16 matmul OK, VRAM used={vram_used:.2f}GB")
        return True

    except Exception as e:
        record("GPU smoke test", FAIL, str(e))
        return False


# ============================================================
# 5. NORMALIZATION VECTORIZATION CHECK
# ============================================================

def normalization_check():
    log.info("\n" + "="*60)
    log.info("NORMALIZATION VECTORIZATION CHECK")
    log.info("="*60)

    # Create a 100K synthetic DataFrame and time normalization
    n = 100_000
    df = pd.DataFrame({
        "entity_id":       [f"S1-{i}" for i in range(n)],
        "business_name":   ["Prabhav Business Center LLC"] * n,
        "business_address":["1234 Main St, New York, NY 10001"] * n,
        "country":         ["US"] * n,
    })

    try:
        from .normalize import normalize_dataframe
        t0 = time.time()
        result = normalize_dataframe(df, n_workers=1)  # single worker for benchmark
        elapsed = time.time() - t0

        rps = n / elapsed
        # Extrapolate to 13M
        eta_13m = 13_000_000 / rps / 60
        log.info(f"  Normalization: {n:,} rows in {elapsed:.2f}s = {rps:,.0f} rows/sec")
        log.info(f"  Extrapolated 13M rows ETA: {eta_13m:.1f} min")

        # Verify no forbidden ops — check result
        assert "norm_name" in result.columns, "norm_name missing"
        assert "norm_addr" in result.columns, "norm_addr missing"

        if rps < 10_000:
            record("Normalization speed", WARN,
                   f"{rps:,.0f} rows/sec — too slow. Consider pure vectorized.")
        else:
            record("Normalization speed", PASS,
                   f"{rps:,.0f} rows/sec, 13M ETA: {eta_13m:.1f}min")
        return rps

    except Exception as e:
        record("Normalization check", FAIL, str(e))
        traceback.print_exc()
        return 0


# ============================================================
# 6. MICRO TEST (1K S1 / 5K S2 / 5K S3)
# ============================================================

def micro_test(skip: bool = False):
    log.info("\n" + "="*60)
    log.info("MICRO TEST (1K×5K×5K)")
    log.info("="*60)

    if skip:
        record("Micro test", WARN, "Skipped by --skip-micro flag")
        return True

    from . import config
    if not os.path.exists(config.TRAIN_S1):
        record("Micro test", WARN, "Train data not found — skipping micro test")
        return True

    try:
        # Load tiny slices
        s1 = pd.read_csv(config.TRAIN_S1, sep="\t", dtype=str, nrows=1000,
                          keep_default_na=False)
        s2 = pd.read_csv(config.TRAIN_S2, sep="\t", dtype=str, nrows=5000,
                          keep_default_na=False)
        s3 = pd.read_csv(config.TRAIN_S3, sep="\t", dtype=str, nrows=5000,
                          keep_default_na=False)
        gt = pd.read_csv(config.TRAIN_GT, sep="\t", dtype=str, nrows=1000,
                          keep_default_na=False)

        for df, name in [(s1,"S1"),(s2,"S2"),(s3,"S3"),(gt,"GT")]:
            df.fillna("", inplace=True)
            log.info(f"  {name}: {len(df):,} rows, cols={list(df.columns)}")

        # Normalize
        from .normalize import normalize_dataframe
        t0 = time.time()
        s1 = normalize_dataframe(s1, n_workers=2)
        s2 = normalize_dataframe(s2, n_workers=2)
        s3 = normalize_dataframe(s3, n_workers=2)
        norm_t = time.time() - t0
        log.info(f"  Normalization: {norm_t:.2f}s")

        # Build indexes
        from .indexes import (ExactNameIndex, RareTokenIndex, CharNgramIndex,
                              AddressIndex, build_name_frequency_table)
        s23 = pd.concat([s2, s3], ignore_index=True)

        t0 = time.time()
        exact_idx = ExactNameIndex(); exact_idx.build(s23)
        rare_idx  = RareTokenIndex(); rare_idx.build(s23)
        tfidf_idx = CharNgramIndex(); tfidf_idx.build(s23)
        addr_idx  = AddressIndex();   addr_idx.build(s23)
        name_freq = build_name_frequency_table([s1, s2, s3])
        idx_t = time.time() - t0
        log.info(f"  Index build: {idx_t:.2f}s")

        # Candidate generation
        from .retrieve import generate_candidates
        t0 = time.time()
        cands = generate_candidates(
            s1_df=s1, s2_df=s2, s3_df=s3,
            exact_idx=exact_idx, rare_idx=rare_idx,
            tfidf_idx=tfidf_idx, addr_idx=addr_idx,
            name_freq=name_freq,
            batch_size=100,
        )
        cand_t = time.time() - t0
        avg_c = len(cands) / max(len(s1), 1)
        log.info(f"  Candidate gen: {cand_t:.2f}s, {len(cands):,} pairs, "
                 f"avg={avg_c:.1f} cands/S1")

        # Feature extraction
        from .features import extract_features, FEATURE_COLS
        s23_all = pd.concat([s2, s3], ignore_index=True)
        t0 = time.time()
        feats = extract_features(cands, s1, s23_all, name_freq, n_workers=2)
        feat_t = time.time() - t0
        log.info(f"  Feature extraction: {feat_t:.2f}s, {len(feats):,} rows, "
                 f"{len(feats.columns)} cols")

        # Check no NaN in feature cols
        missing_cols = [c for c in FEATURE_COLS if c not in feats.columns]
        if missing_cols:
            record("Micro test features", WARN, f"Missing cols: {missing_cols}")
        else:
            record("Micro test features", PASS, f"{len(FEATURE_COLS)} feature cols present")

        nan_count = feats[FEATURE_COLS].isna().sum().sum()
        if nan_count > 0:
            record("Micro test NaN check", WARN, f"{nan_count} NaN values in features")
        else:
            record("Micro test NaN check", PASS, "No NaN in feature cols")

        total_t = norm_t + idx_t + cand_t + feat_t
        record("Micro test", PASS,
               f"Complete in {total_t:.1f}s | avg {avg_c:.1f} cands/S1")

        # Throughput extrapolation
        s1_full = 2_206_821
        cand_rps = len(s1) / cand_t if cand_t > 0 else 1
        feat_rps = len(feats) / feat_t if feat_t > 0 else 1
        cand_eta  = s1_full / cand_rps / 60
        feat_eta  = s1_full * 50 / feat_rps / 60
        log.info(f"\n  THROUGHPUT EXTRAPOLATION:")
        log.info(f"    Candidate gen: {cand_rps:,.0f} S1/sec → full {cand_eta:.1f}min")
        log.info(f"    Feature extr:  {feat_rps:,.0f} pairs/sec → full {feat_eta:.1f}min")

        return True

    except Exception as e:
        record("Micro test", FAIL, str(e))
        traceback.print_exc()
        return False


# ============================================================
# 7. MEMORY MICRO TEST (100K rows)
# ============================================================

def memory_micro_test():
    log.info("\n" + "="*60)
    log.info("MEMORY MICRO TEST (100K rows)")
    log.info("="*60)

    from . import config
    if not os.path.exists(config.TRAIN_S2):
        record("Memory micro test", WARN, "Train data not found — skipping")
        return

    try:
        n = 100_000
        s23 = pd.read_csv(config.TRAIN_S2, sep="\t", dtype=str, nrows=n,
                           keep_default_na=False)
        s23.fillna("", inplace=True)
        from .normalize import normalize_dataframe
        s23 = normalize_dataframe(s23, n_workers=4)

        ram_before = psutil.virtual_memory().used / 1e9

        from .indexes import CharNgramIndex
        idx = CharNgramIndex()
        idx.build(s23)

        ram_after = psutil.virtual_memory().used / 1e9
        ram_used  = ram_after - ram_before

        # Extrapolate
        scale = (5_034_616 + 5_285_603) / n
        est_full_gb = ram_used * scale

        log.info(f"  100K rows TF-IDF RAM: {ram_used:.2f}GB")
        log.info(f"  Extrapolated full corpus: {est_full_gb:.1f}GB")
        avail = psutil.virtual_memory().available / 1e9
        if est_full_gb > avail * 0.7:
            record("Memory micro test", WARN,
                   f"Est {est_full_gb:.1f}GB > 70% available RAM. "
                   f"Reduce max_features in config.py")
        else:
            record("Memory micro test", PASS,
                   f"Est {est_full_gb:.1f}GB for full corpus TF-IDF")

        del idx
        gc.collect()

    except Exception as e:
        record("Memory micro test", FAIL, str(e))
        traceback.print_exc()


# ============================================================
# 8. COUNTRY OPENSET CHECK
# ============================================================

def country_openset_check():
    log.info("\n" + "="*60)
    log.info("COUNTRY OPEN-SET CHECK")
    log.info("="*60)

    # Scan for hard-coded country strings
    forbidden = ['"US"', "'US'", '"India"', "'India'", "US, India", "India, US"]
    py_files  = list(SRC_DIR.rglob("*.py"))
    found     = []

    for filepath in py_files:
        if "audit" in filepath.name or "eda" in str(filepath):
            continue
        with open(filepath) as f:
            source = f.read()
        for pat in forbidden:
            if pat in source:
                # Allow in config comments, not in logic
                for i, line in enumerate(source.splitlines(), 1):
                    if pat in line and not line.strip().startswith("#"):
                        found.append(f"{filepath.name}:{i}: {line.strip()[:80]}")

    if found:
        record("Country open-set", WARN,
               f"Hard-coded country strings found:\n" + "\n".join(found[:5]))
    else:
        record("Country open-set", PASS, "No hard-coded US/India in logic paths")


# ============================================================
# FINAL REPORT
# ============================================================

def final_report():
    log.info("\n" + "="*60)
    log.info("AUDIT SUMMARY")
    log.info("="*60)

    n_pass = sum(1 for _, s, _ in audit_results if "PASS" in s)
    n_warn = sum(1 for _, s, _ in audit_results if "WARN" in s)
    n_fail = sum(1 for _, s, _ in audit_results if "FAIL" in s)

    log.info(f"  {PASS}  {n_pass} checks passed")
    log.info(f"  {WARN}  {n_warn} warnings")
    log.info(f"  {FAIL}  {n_fail} failures")

    if n_fail > 0:
        log.error("\n🚨 AUDIT FAILED — DO NOT START FULL PIPELINE. Fix above failures first.\n")
        return False
    elif n_warn > 0:
        log.warning("\n⚠️  Audit passed with warnings. Review above before full run.\n")
        return True
    else:
        log.info("\n✅ All audit checks passed. Safe to start full pipeline.\n")
        return True


# ============================================================
# ENTRY POINT
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Pipeline Audit + Preflight")
    parser.add_argument("--skip-micro",   action="store_true",
                        help="Skip the micro end-to-end test")
    parser.add_argument("--skip-memory",  action="store_true",
                        help="Skip the 100K memory extrapolation test")
    args = parser.parse_args()

    t0 = time.time()

    system_report()
    ok_static = static_audit()

    if not ok_static:
        log.error("Static audit FAILED. Fix forbidden patterns before continuing.")
        # Don't abort — show full picture
    
    complexity_audit()
    gpu_smoke_test()
    norm_rps = normalization_check()
    country_openset_check()

    if not args.skip_memory:
        memory_micro_test()

    ok_micro = micro_test(skip=args.skip_micro)

    ok = final_report()

    elapsed = time.time() - t0
    log.info(f"Audit completed in {elapsed:.1f}s")

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
