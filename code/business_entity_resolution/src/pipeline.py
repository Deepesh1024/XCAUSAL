"""
pipeline.py — Main competition pipeline. Fully checkpointed and restartable.

Usage:
    python -m code.business_entity_resolution.src.pipeline [--stage all] [--resume]
"""
import argparse
import logging
import os
import sys
import time
import torch

# --- bootstrap logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

from . import config
from .io import (
    load_source, load_ground_truth, parse_gt_to_dict,
    atomic_save_pickle, atomic_load_pickle,
    atomic_save_parquet, atomic_load_parquet,
    atomic_save_json, atomic_load_json,
    checkpoint_exists,
    write_matching_results, write_candidate_pairs,
)
from .normalize import normalize_dataframe
from .indexes import (
    ExactNameIndex, RareTokenIndex, CharNgramIndex, AddressIndex,
    build_name_frequency_table, save_indexes, load_indexes,
)
from .retrieve import generate_candidates
from .embeddings import BGEEmbedder
from .features import extract_features, FEATURE_COLS
from .hard_negatives import build_hard_negatives
from .train import train_lightgbm
from .predict import predict_proba, apply_threshold, validate_outputs
from .validate import report_statistics

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PIPELINE_START = time.time()


def elapsed() -> float:
    return time.time() - PIPELINE_START


def elapsed_str() -> str:
    e = elapsed()
    return f"{e/60:.1f}m"


def gpu_mem_str() -> str:
    if torch.cuda.is_available():
        used = torch.cuda.memory_allocated() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        return f"GPU {used:.1f}/{total:.1f}GB"
    return "CPU-only"


def time_guard(limit_sec: float, name: str) -> bool:
    """Return True if we are past the time limit."""
    if elapsed() > limit_sec:
        log.warning(f"TIME GUARD: {name} skipped (elapsed={elapsed_str()})")
        return True
    return False


def make_dirs():
    for d in [
        config.CKPT_NORM_TRAIN, config.CKPT_NORM_TEST,
        config.CKPT_INDEXES,
        config.CKPT_CAND_TRAIN, config.CKPT_CAND_TEST,
        config.CKPT_BGE,
        config.CKPT_FEAT_TRAIN, config.CKPT_FEAT_TEST,
        config.CKPT_MODEL,
        config.CKPT_PREDS, config.CKPT_FINAL,
        config.OUTPUT_DIR,
    ]:
        os.makedirs(d, exist_ok=True)


# ---------------------------------------------------------------------------
# Stage functions
# ---------------------------------------------------------------------------

def stage_preprocess(resume: bool = True) -> dict:
    """Load and normalize all datasets."""
    log.info(f"\n{'='*60}")
    log.info(f"STAGE: preprocess  [{elapsed_str()}]  {gpu_mem_str()}")
    log.info(f"{'='*60}")

    data = {}
    splits = {
        "train": (config.TRAIN_S1, config.TRAIN_S2, config.TRAIN_S3),
        "test":  (config.TEST_S1,  config.TEST_S2,  config.TEST_S3),
    }

    for split, (s1_path, s2_path, s3_path) in splits.items():
        ckpt_dir = config.CKPT_NORM_TRAIN if split == "train" else config.CKPT_NORM_TEST
        paths    = {k: os.path.join(ckpt_dir, f"{k}.parquet")
                    for k in ["s1", "s2", "s3"]}

        if resume and checkpoint_exists(*paths.values()):
            log.info(f"  [{split}] Loading cached normalized data...")
            data[split] = {k: atomic_load_parquet(v) for k, v in paths.items()}
            for k, df in data[split].items():
                log.info(f"    {k.upper()}: {len(df):,}")
            continue

        # Check if test files exist
        if split == "test":
            for p in [s1_path, s2_path, s3_path]:
                if not os.path.exists(p):
                    log.warning(f"  Test file not found: {p} — skipping test split")
                    data[split] = None
                    break
            else:
                pass

        if data.get(split) is None:
            continue

        s1 = load_source(s1_path, "S1")
        s2 = load_source(s2_path, "S2")
        s3 = load_source(s3_path, "S3")

        log.info(f"  Normalizing {split} data...")
        s1 = normalize_dataframe(s1)
        s2 = normalize_dataframe(s2)
        s3 = normalize_dataframe(s3)

        data[split] = {"s1": s1, "s2": s2, "s3": s3}

        for k, df in data[split].items():
            atomic_save_parquet(df, paths[k])

        log.info(f"  [{split}] Normalization complete.")

    # Ground truth (train only)
    gt_path = os.path.join(config.CKPT_NORM_TRAIN, "gt.parquet")
    if resume and os.path.exists(gt_path):
        data["gt"] = atomic_load_parquet(gt_path)
    else:
        if os.path.exists(config.TRAIN_GT):
            data["gt"] = load_ground_truth(config.TRAIN_GT)
            atomic_save_parquet(data["gt"], gt_path)

    return data


def stage_indexes(data: dict, resume: bool = True) -> dict:
    """Build all retrieval indexes on S2+S3."""
    log.info(f"\n{'='*60}")
    log.info(f"STAGE: indexes  [{elapsed_str()}]  {gpu_mem_str()}")
    log.info(f"{'='*60}")

    index_names = ["exact_idx", "rare_idx", "tfidf_idx", "addr_idx", "name_freq"]
    paths = {n: os.path.join(config.CKPT_INDEXES, f"{n}.pkl") for n in index_names}

    if resume and checkpoint_exists(*paths.values()):
        log.info("  Loading cached indexes...")
        return {n: atomic_load_pickle(paths[n]) for n in index_names}

    # Combine S2 and S3 for indexing
    train_data = data["train"]
    s2 = train_data["s2"]
    s3 = train_data["s3"]
    s23 = pd.concat([s2, s3], ignore_index=True)

    log.info(f"  Building indexes on {len(s23):,} S2+S3 records...")

    exact_idx = ExactNameIndex()
    exact_idx.build(s23)

    rare_idx = RareTokenIndex()
    rare_idx.build(s23)

    tfidf_idx = CharNgramIndex()
    tfidf_idx.build(s23)

    addr_idx = AddressIndex()
    addr_idx.build(s23)

    name_freq = build_name_frequency_table([train_data["s1"], s2, s3])

    indexes = {
        "exact_idx": exact_idx,
        "rare_idx":  rare_idx,
        "tfidf_idx": tfidf_idx,
        "addr_idx":  addr_idx,
        "name_freq": name_freq,
    }

    for name, obj in indexes.items():
        atomic_save_pickle(obj, paths[name])

    log.info(f"  Indexes built and saved. [{elapsed_str()}]")
    return indexes


def stage_candidates(data: dict, indexes: dict, split: str,
                     bge_scores: dict = None, resume: bool = True) -> pd.DataFrame:
    """Generate candidates for a given split."""
    ckpt_dir = config.CKPT_CAND_TRAIN if split == "train" else config.CKPT_CAND_TEST
    ckpt     = os.path.join(ckpt_dir, "candidates.parquet")

    log.info(f"\n{'='*60}")
    log.info(f"STAGE: candidates [{split}]  [{elapsed_str()}]  {gpu_mem_str()}")
    log.info(f"{'='*60}")

    if resume and checkpoint_exists(ckpt):
        log.info(f"  Loading cached candidates from {ckpt}")
        return atomic_load_parquet(ckpt)

    split_data = data.get(split)
    if split_data is None:
        log.warning(f"  No data for split {split}, skipping.")
        return pd.DataFrame()

    # For test: build new indexes on test s2+s3 (or reuse train indexes)
    if split == "test":
        test_s23 = pd.concat([split_data["s2"], split_data["s3"]], ignore_index=True)
        exact_idx = ExactNameIndex(); exact_idx.build(test_s23)
        rare_idx  = RareTokenIndex(); rare_idx.build(test_s23)
        tfidf_idx = CharNgramIndex(); tfidf_idx.build(test_s23)
        addr_idx  = AddressIndex();   addr_idx.build(test_s23)
        _indexes  = {**indexes, "exact_idx": exact_idx, "rare_idx": rare_idx,
                     "tfidf_idx": tfidf_idx, "addr_idx": addr_idx}
    else:
        _indexes  = indexes
        test_s23  = None

    s2 = split_data["s2"]
    s3 = split_data["s3"]

    cands = generate_candidates(
        s1_df      = split_data["s1"],
        s2_df      = s2,
        s3_df      = s3,
        exact_idx  = _indexes["exact_idx"],
        rare_idx   = _indexes["rare_idx"],
        tfidf_idx  = _indexes["tfidf_idx"],
        addr_idx   = _indexes["addr_idx"],
        name_freq  = indexes["name_freq"],
        bge_scores = bge_scores,
        split      = split,
    )

    atomic_save_parquet(cands, ckpt)
    log.info(f"  Candidates saved: {len(cands):,} pairs. [{elapsed_str()}]")
    return cands


def stage_bge(data: dict, candidates: pd.DataFrame, split: str,
              resume: bool = True) -> dict:
    """Compute BGE embeddings and dense cosine scores."""
    log.info(f"\n{'='*60}")
    log.info(f"STAGE: bge [{split}]  [{elapsed_str()}]  {gpu_mem_str()}")
    log.info(f"{'='*60}")

    if time_guard(config.TIME_SKIP_DENSE, "BGE dense retrieval"):
        return {}

    scores_path = os.path.join(config.CKPT_BGE, f"bge_scores_{split}.pkl")
    if resume and checkpoint_exists(scores_path):
        log.info("  Loading cached BGE scores...")
        return atomic_load_pickle(scores_path)

    split_data = data.get(split)
    if split_data is None:
        return {}

    embedder = BGEEmbedder()

    # Only embed S1 and the candidate-pool records (not the full corpus)
    cand_ids = set(candidates["candidate_entity_id"].tolist())
    s23_df   = pd.concat([split_data["s2"], split_data["s3"]], ignore_index=True)
    s23_cand = s23_df[s23_df["entity_id"].isin(cand_ids)].copy()
    s23_cand = s23_cand.drop_duplicates("entity_id").reset_index(drop=True)

    s1_df = split_data["s1"]

    try:
        s1_embs   = embedder.compute_or_load(s1_df,   f"s1_{split}")
        cand_embs = embedder.compute_or_load(s23_cand, f"s23_cand_{split}")
    except Exception as e:
        log.error(f"BGE embedding failed: {e}. Falling back to no BGE.")
        embedder.free_model()
        return {}

    log.info(f"  Computing cosine scores... [{elapsed_str()}]")
    # Build s1 -> scores dict
    s1_ids   = s1_df["entity_id"].values
    cand_ids_arr = s23_cand["entity_id"].values

    results_list = embedder.cosine_retrieval(
        s1_embs, cand_embs, cand_ids_arr, top_k=config.CANDIDATE_K
    )

    bge_scores = {}
    for s1_id, top_pairs in zip(s1_ids, results_list):
        bge_scores[s1_id] = {eid: score for eid, score in top_pairs}

    embedder.free_model()
    atomic_save_pickle(bge_scores, scores_path)
    log.info(f"  BGE scores computed and saved. [{elapsed_str()}]")
    return bge_scores


def stage_features(data: dict, candidates: pd.DataFrame,
                   indexes: dict, bge_scores: dict,
                   split: str, resume: bool = True) -> pd.DataFrame:
    """Extract LightGBM features for all candidate pairs."""
    log.info(f"\n{'='*60}")
    log.info(f"STAGE: features [{split}]  [{elapsed_str()}]  {gpu_mem_str()}")
    log.info(f"{'='*60}")

    ckpt_dir = config.CKPT_FEAT_TRAIN if split == "train" else config.CKPT_FEAT_TEST
    ckpt     = os.path.join(ckpt_dir, "features.parquet")

    if resume and checkpoint_exists(ckpt):
        log.info(f"  Loading cached features from {ckpt}")
        return atomic_load_parquet(ckpt)

    split_data = data.get(split)
    if split_data is None or len(candidates) == 0:
        return pd.DataFrame()

    s23 = pd.concat([split_data["s2"], split_data["s3"]], ignore_index=True)

    feat_df = extract_features(
        candidates_df = candidates,
        s1_df         = split_data["s1"],
        s23_df        = s23,
        name_freq     = indexes["name_freq"],
        bge_scores    = bge_scores,
    )

    atomic_save_parquet(feat_df, ckpt)
    log.info(f"  Features saved. [{elapsed_str()}]")
    return feat_df


def stage_train(feat_df: pd.DataFrame, candidates: pd.DataFrame,
                gt_df: pd.DataFrame, resume: bool = True):
    """Train LightGBM."""
    log.info(f"\n{'='*60}")
    log.info(f"STAGE: train  [{elapsed_str()}]  {gpu_mem_str()}")
    log.info(f"{'='*60}")

    model_path = os.path.join(config.CKPT_MODEL, "lgb_model.pkl")
    meta_path  = os.path.join(config.CKPT_MODEL, "meta.json")

    if resume and checkpoint_exists(model_path, meta_path):
        log.info("  Loading cached model...")
        booster   = atomic_load_pickle(model_path)
        meta      = atomic_load_json(meta_path)
        return booster, meta["threshold"], meta["val_f05"]

    gt_dict = parse_gt_to_dict(gt_df)

    # Label candidates
    pos_pairs = {(s1, m) for s1, matches in gt_dict.items() for m in matches}
    feat_df["label"] = feat_df.apply(
        lambda r: int((r["source1_entity_id"], r["candidate_entity_id"]) in pos_pairs),
        axis=1,
    )

    # Build hard negatives (adds label column based on GT)
    labeled_df = build_hard_negatives(candidates, gt_dict,
                                      data["train"]["s1"],
                                      pd.concat([data["train"]["s2"],
                                                 data["train"]["s3"]], ignore_index=True))

    # Merge labels into feature df
    feat_df = feat_df.copy()
    label_map = dict(zip(
        zip(labeled_df["source1_entity_id"], labeled_df["candidate_entity_id"]),
        labeled_df["label"]
    ))
    feat_df["label"] = feat_df.apply(
        lambda r: label_map.get((r["source1_entity_id"], r["candidate_entity_id"]),
                                int((r["source1_entity_id"], r["candidate_entity_id"])
                                    in pos_pairs)),
        axis=1,
    )

    booster, best_thr, val_f05 = train_lightgbm(feat_df)
    return booster, best_thr, val_f05


def stage_predict(feat_df: pd.DataFrame, booster, threshold: float,
                  s1_ids: list, candidates: pd.DataFrame,
                  split: str, resume: bool = True) -> dict:
    """Run inference."""
    log.info(f"\n{'='*60}")
    log.info(f"STAGE: predict [{split}]  [{elapsed_str()}]  {gpu_mem_str()}")
    log.info(f"{'='*60}")

    ckpt = os.path.join(config.CKPT_PREDS, f"results_{split}.pkl")
    if resume and checkpoint_exists(ckpt):
        log.info("  Loading cached predictions...")
        return atomic_load_pickle(ckpt)

    if len(feat_df) == 0:
        results = {s1: [] for s1 in s1_ids}
        atomic_save_pickle(results, ckpt)
        return results

    probas  = predict_proba(feat_df, booster)
    results = apply_threshold(feat_df, probas, threshold, s1_ids)
    atomic_save_pickle(results, ckpt)
    log.info(f"  Predictions saved. [{elapsed_str()}]")
    return results


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

data = {}  # module-level, used by stage_train closure


def run_pipeline(stages: list, resume: bool = True):
    global data

    make_dirs()

    run_all = ("all" in stages or not stages)

    # ---------- preprocess ----------
    if run_all or "preprocess" in stages:
        data = stage_preprocess(resume)
    else:
        # Load cached
        data = stage_preprocess(resume=True)

    # ---------- indexes ----------
    indexes = {}
    if run_all or "indexes" in stages or "retrieve" in stages:
        indexes = stage_indexes(data, resume)
    else:
        index_names = ["exact_idx", "rare_idx", "tfidf_idx", "addr_idx", "name_freq"]
        idx_paths = {n: os.path.join(config.CKPT_INDEXES, f"{n}.pkl") for n in index_names}
        if checkpoint_exists(*idx_paths.values()):
            indexes = {n: atomic_load_pickle(idx_paths[n]) for n in index_names}

    # ---------- BGE pass 1 (for enriching candidates) ----------
    bge_train = {}
    if not time_guard(config.TIME_SKIP_DENSE, "BGE train pre-candidate"):
        if run_all or "embed" in stages:
            # We'll do BGE after candidates using candidates as corpus
            pass

    # ---------- candidates (train) ----------
    cands_train = pd.DataFrame()
    if run_all or "retrieve" in stages:
        cands_train = stage_candidates(data, indexes, "train", None, resume)

    # ---------- BGE for train ----------
    bge_train = {}
    if run_all or "embed" in stages:
        bge_train = stage_bge(data, cands_train, "train", resume)

    # If we have BGE scores and hadn't used them yet in candidates, regenerate
    if bge_train and resume:
        ckpt_cand = os.path.join(config.CKPT_CAND_TRAIN, "candidates_with_bge.parquet")
        if not checkpoint_exists(ckpt_cand):
            log.info("Re-generating candidates with BGE scores...")
            cands_train_bge = stage_candidates(
                data, indexes, "train", bge_train, resume=False
            )
            atomic_save_parquet(cands_train_bge, ckpt_cand)
            cands_train = cands_train_bge

    # ---------- features (train) ----------
    feat_train = pd.DataFrame()
    if run_all or "features" in stages:
        feat_train = stage_features(data, cands_train, indexes, bge_train,
                                    "train", resume)

    # ---------- train ----------
    booster = threshold = None
    if run_all or "train" in stages:
        if len(feat_train) > 0 and "gt" in data:
            booster, threshold, val_f05 = stage_train(feat_train, cands_train,
                                                      data["gt"], resume)
            log.info(f"  Model ready: threshold={threshold:.4f}, val F0.5={val_f05:.4f}")
        else:
            log.error("Cannot train: missing features or GT.")
            return

    else:
        # Load model from checkpoint
        model_path = os.path.join(config.CKPT_MODEL, "lgb_model.pkl")
        meta_path  = os.path.join(config.CKPT_MODEL, "meta.json")
        if checkpoint_exists(model_path, meta_path):
            booster   = atomic_load_pickle(model_path)
            meta      = atomic_load_json(meta_path)
            threshold = meta["threshold"]

    if booster is None:
        log.error("No model available. Aborting.")
        return

    # ---------- test candidates ----------
    cands_test = pd.DataFrame()
    if run_all or "predict" in stages:
        test_data = data.get("test")
        if test_data is not None:
            cands_test = stage_candidates(data, indexes, "test", None, resume)
            bge_test   = stage_bge(data, cands_test, "test", resume)
            if bge_test:
                ckpt_ct = os.path.join(config.CKPT_CAND_TEST, "candidates_with_bge.parquet")
                if not checkpoint_exists(ckpt_ct):
                    cands_test = stage_candidates(data, indexes, "test", bge_test,
                                                  resume=False)
                    atomic_save_parquet(cands_test, ckpt_ct)
            feat_test = stage_features(data, cands_test, indexes, bge_test,
                                       "test", resume)
        else:
            log.warning("No test data. Running inference on train split as demo.")
            feat_test  = feat_train
            cands_test = cands_train
            test_data  = data["train"]

    # ---------- predict ----------
    if run_all or "predict" in stages:
        split_data = data.get("test") or data["train"]
        s1_ids = split_data["s1"]["entity_id"].tolist()

        results = stage_predict(
            feat_test, booster, threshold,
            s1_ids, cands_test, "test", resume
        )

    # ---------- validate ----------
    if run_all or "validate" in stages:
        split_data = data.get("test") or data["train"]
        s1_ids = split_data["s1"]["entity_id"].tolist()
        s2_ids = set(split_data["s2"]["entity_id"].tolist())
        s3_ids = set(split_data["s3"]["entity_id"].tolist())
        gt_dict = parse_gt_to_dict(data["gt"]) if "gt" in data else None

        ok = validate_outputs(results, s1_ids, s2_ids, s3_ids, cands_test)
        report_statistics(results, s1_ids, cands_test, gt_dict,
                          split="test" if data.get("test") else "train")

    # ---------- write outputs ----------
    if run_all or "predict" in stages:
        split_data = data.get("test") or data["train"]
        s1_ids = split_data["s1"]["entity_id"].tolist()

        write_matching_results(results, s1_ids, config.OUTPUT_MATCHING)
        write_candidate_pairs(cands_test, config.OUTPUT_CANDIDATES)

        log.info(f"\nOutputs written:")
        log.info(f"  {config.OUTPUT_MATCHING}")
        log.info(f"  {config.OUTPUT_CANDIDATES}")
        log.info(f"\nTotal elapsed: {elapsed_str()}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Business Entity Resolution Competition Pipeline"
    )
    parser.add_argument(
        "--stage",
        default="all",
        choices=["all", "preprocess", "indexes", "retrieve",
                 "embed", "features", "train", "predict", "validate"],
        help="Pipeline stage to run"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Resume from checkpoints if available"
    )
    parser.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Force rerun all stages"
    )
    args = parser.parse_args()

    log.info("=" * 60)
    log.info("Business Entity Resolution — Competition Pipeline")
    log.info(f"Stage: {args.stage} | Resume: {args.resume}")
    log.info(f"Device: {'CUDA:' + str(torch.cuda.get_device_name(0)) if torch.cuda.is_available() else 'CPU'}")
    log.info(f"BGE model: {config.BGE_MODEL_NAME}")
    log.info("=" * 60)

    run_pipeline([args.stage], args.resume)


if __name__ == "__main__":
    main()
