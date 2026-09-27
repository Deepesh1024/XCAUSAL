"""
config.py — Central configuration for the Business Entity Resolution pipeline.
All paths, hyper-parameters, and toggles live here.
"""
import os
import multiprocessing

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
)

DATA_ROOT_TRAIN = os.path.join(REPO_ROOT, "dataset", "train")
DATA_ROOT_TEST  = os.path.join(REPO_ROOT, "dataset", "test")

TRAIN_S1  = os.path.join(DATA_ROOT_TRAIN, "train_source1.tsv")
TRAIN_S2  = os.path.join(DATA_ROOT_TRAIN, "train_source2.tsv")
TRAIN_S3  = os.path.join(DATA_ROOT_TRAIN, "train_source3.tsv")
TRAIN_GT  = os.path.join(DATA_ROOT_TRAIN, "train_ground_truth.tsv")

TEST_S1   = os.path.join(DATA_ROOT_TEST,  "test_source1.tsv")
TEST_S2   = os.path.join(DATA_ROOT_TEST,  "test_source2.tsv")
TEST_S3   = os.path.join(DATA_ROOT_TEST,  "test_source3.tsv")

CHECKPOINT_DIR     = os.path.join(REPO_ROOT, "checkpoints")
OUTPUT_DIR         = os.path.join(REPO_ROOT, "outputs")

# Checkpoint sub-dirs
CKPT_NORM_TRAIN    = os.path.join(CHECKPOINT_DIR, "normalized", "train")
CKPT_NORM_TEST     = os.path.join(CHECKPOINT_DIR, "normalized", "test")
CKPT_INDEXES       = os.path.join(CHECKPOINT_DIR, "indexes")
CKPT_CAND_TRAIN    = os.path.join(CHECKPOINT_DIR, "candidates_train")
CKPT_CAND_TEST     = os.path.join(CHECKPOINT_DIR, "candidates_test")
CKPT_BGE           = os.path.join(CHECKPOINT_DIR, "bge")
CKPT_FEAT_TRAIN    = os.path.join(CHECKPOINT_DIR, "features_train")
CKPT_FEAT_TEST     = os.path.join(CHECKPOINT_DIR, "features_test")
CKPT_MODEL         = os.path.join(CHECKPOINT_DIR, "model")
CKPT_PREDS         = os.path.join(CHECKPOINT_DIR, "predictions")
CKPT_FINAL         = os.path.join(CHECKPOINT_DIR, "final")

OUTPUT_MATCHING    = os.path.join(OUTPUT_DIR, "matching_results.tsv")
OUTPUT_CANDIDATES  = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")

# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------
LEGAL_SUFFIXES = [
    "llc", "inc", "incorporated", "corp", "corporation",
    "co", "company", "ltd", "limited", "pvt", "private",
    "plc", "llp", "lp", "ltda", "gmbh", "ag", "sa", "srl",
    "services", "solutions", "group", "enterprises", "associates",
    "partners", "ventures", "holdings", "international", "global",
]

ADDR_ABBREVS = {
    "street": "st", "road": "rd", "avenue": "ave", "drive": "dr",
    "lane": "ln", "boulevard": "blvd", "highway": "hwy",
    "suite": "ste", "apartment": "apt", "floor": "fl",
    "place": "pl", "court": "ct", "circle": "cir",
}

# ---------------------------------------------------------------------------
# Candidate generation
# ---------------------------------------------------------------------------
CANDIDATE_K              = 50     # max candidates per S1
RARE_TOKEN_MAX_DF        = 0.01   # tokens appearing in > 1% docs are "common"
RARE_TOKEN_TOP_K         = 30     # top K rare tokens per name used for lookup

TFIDF_MAX_FEATURES       = 2_000_000
TFIDF_NGRAM_RANGE        = (3, 5)
TFIDF_MIN_DF             = 2
TFIDF_BATCH_SIZE         = 10_000  # rows per sparse retrieval batch

# ---------------------------------------------------------------------------
# BGE-M3
# ---------------------------------------------------------------------------
BGE_MODEL_NAME           = "BAAI/bge-m3"
BGE_MAX_SEQ_LEN          = 128
BGE_BATCH_SIZE           = 512
BGE_FP16                 = True
BGE_ENABLED              = True    # runtime-toggled by time guard

# ---------------------------------------------------------------------------
# LoRA fine-tuning (optional)
# ---------------------------------------------------------------------------
LORA_ENABLED             = True
LORA_RANK                = 16
LORA_ALPHA               = 32
LORA_DROPOUT             = 0.05
LORA_MAX_MINUTES         = 15

# ---------------------------------------------------------------------------
# Cross-encoder reranker (optional)
# ---------------------------------------------------------------------------
RERANKER_MODEL           = "BAAI/bge-reranker-v2-m3"
RERANKER_ENABLED         = True
RERANKER_TOP_K           = 15     # candidates to rerank per S1
RERANKER_MAX_MINUTES     = 20

# ---------------------------------------------------------------------------
# LightGBM
# ---------------------------------------------------------------------------
LGB_PARAMS = {
    "objective":        "binary",
    "metric":           "binary_logloss",
    "num_leaves":       255,
    "learning_rate":    0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq":     5,
    "min_child_samples": 20,
    "n_estimators":     1500,
    "verbose":          -1,
    "n_jobs":           -1,
    "device_type":      "gpu",
}
LGB_EARLY_STOPPING       = 100
LGB_VAL_FRACTION         = 0.1

# Threshold search
THRESHOLD_COARSE_START   = 0.50
THRESHOLD_COARSE_END     = 0.95
THRESHOLD_COARSE_STEP    = 0.05
THRESHOLD_FINE_HALF_WINDOW = 0.06
THRESHOLD_FINE_STEP      = 0.002

# ---------------------------------------------------------------------------
# Candidate ranking weights (used ONLY to rank/trim candidates, not for matching)
# ---------------------------------------------------------------------------
CAND_WEIGHT_NAME_LEX     = 0.35
CAND_WEIGHT_ADDR_LEX     = 0.20
CAND_WEIGHT_NUM_OVERLAP  = 0.15
CAND_WEIGHT_BGE          = 0.15
CAND_WEIGHT_RARE_TOK     = 0.10
CAND_WEIGHT_COUNTRY      = 0.05

# ---------------------------------------------------------------------------
# Time guards (seconds)
# ---------------------------------------------------------------------------
TIME_SKIP_DENSE          = 30 * 60   # 30 min → skip dense BGE retrieval
TIME_SKIP_LORA           = 45 * 60   # 45 min → skip LoRA
TIME_SKIP_RERANKER       = 75 * 60   # 75 min → skip cross-encoder
TIME_HARD_LIMIT          = 120 * 60  # 120 min → force final output

# ---------------------------------------------------------------------------
# Parallelism
# ---------------------------------------------------------------------------
NUM_WORKERS = min(multiprocessing.cpu_count(), 16)

# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------
RANDOM_SEED              = 42
LOG_INTERVAL             = 10_000   # records between progress logs
