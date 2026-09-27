# Business Entity Resolution — Competition Pipeline

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run full pipeline
python -m code.business_entity_resolution.src.pipeline --stage all --resume

# Resume from last checkpoint
python -m code.business_entity_resolution.src.pipeline --stage all --resume

# Run individual stages
python -m code.business_entity_resolution.src.pipeline --stage preprocess
python -m code.business_entity_resolution.src.pipeline --stage indexes
python -m code.business_entity_resolution.src.pipeline --stage retrieve
python -m code.business_entity_resolution.src.pipeline --stage embed
python -m code.business_entity_resolution.src.pipeline --stage features
python -m code.business_entity_resolution.src.pipeline --stage train
python -m code.business_entity_resolution.src.pipeline --stage predict
python -m code.business_entity_resolution.src.pipeline --stage validate
```

## Architecture

```
S1 (2.2M records)
        |
  normalization (CPU multiprocessing)
        |
  ┌─────+──────────────────────────────────────┐
  │                                             │
  Sparse Retrieval:                    BGE-M3 (GPU, FP16):
  - Exact name index                   - Embeds only candidate-pool records
  - Rare-token inverted index          - Cosine retrieval via batched matmul
  - TF-IDF char 3-5 gram index
  - Postal/PIN code index
  - Address token index
  - Number overlap index
  │                                             │
  └─────────────┬───────────────────────────────┘
                │
         Candidate union
         K = 50 per S1
                │
         Pair feature extraction (~38 features):
         - Name: Jaccard, Jaro-Winkler, Levenshtein, n-gram cosine
         - Address: token Jaccard, numeric overlap, postal exact
         - Contradiction: name_high_addr_low, postal_mismatch, num_conflict
         - Metadata: country_exact, addr_missing, name_rarity
         - Semantic: BGE cosine
                │
         LightGBM binary classifier
         (trained on hard negatives from candidate pool)
                │
         Precision-heavy threshold (F0.5 optimized)
                │
         Final matches → matching_results.tsv + candidate_pairs.tsv
```

## Time Budget (RTX 4090, 120 min target)

| Stage            | Target Time |
|------------------|-------------|
| preprocess       | 0–10 min    |
| indexes          | 10–20 min   |
| retrieve         | 20–30 min   |
| embed (BGE)      | 30–50 min   |
| features         | 50–75 min   |
| train (LightGBM) | 75–90 min   |
| predict          | 90–105 min  |
| validate/output  | 105–120 min |

## Outputs

- `outputs/matching_results.tsv` — one row per S1 with comma-separated matched IDs
- `outputs/candidate_pairs.tsv` — all evaluated candidate pairs

## Checkpoints

Every stage writes atomic checkpoints to `checkpoints/`. The pipeline is safely
restartable at any stage. If a stage is killed, the next run will resume from the
last completed checkpoint.

## Time Guards

- 30 min: skip BGE dense retrieval if not completed
- 45 min: skip LoRA fine-tuning
- 75 min: skip cross-encoder reranker
- 120 min: force final output regardless of state
