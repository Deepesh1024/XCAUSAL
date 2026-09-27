# Business Entity Resolution — pipeline

Data → normalize → key blocking → pair features → LightGBM → one-owner assignment +
F0.5-tuned threshold → `output/matching_results.tsv` + `output/candidate_pairs.tsv`.

Everything runs on CPU. No pretrained models, no external data or APIs.

## Setup

```bash
# from the repo root (the folder that contains dataset/ and utils/)
pip install -r code/business_entity_resolution/requirements.txt
```

Expected data layout (default `--data-dir` is `<repo>/dataset`):

```
dataset/train/train_source{1,2,3}.tsv  dataset/train/train_ground_truth.tsv
dataset/test/test_source{1,2,3}.tsv
```

## One command: full pipeline

```bash
python code/business_entity_resolution/src/run.py --sample 100000
```

This (1) builds a training sample of 100k Source 1 entities plus all their true matches
plus distractors, (2) trains and validates on a grouped 80/20 split by S1 entity, tunes the
threshold for macro F0.5, (3) retrains on train+val, (4) predicts the **full** test set one
country at a time, (5) writes both output files to `output/` and runs
`utils/validate_submission.py` on them.

Training on a sample is deliberate: 100k entities give about 1.5M labelled pairs, which is plenty
for LightGBM. Test prediction always covers every test S1 entity.

### Flags

| flag | meaning |
|---|---|
| `--sample N` | train/validate on N sampled S1 entities (0 = all train, slow) |
| `--mode full\|fast` | blocking config. `fast` = fewer key families, tighter caps, top-12 instead of top-25 |
| `--skip-test` | only train + validate (dev iterations) |
| `--threshold T` | override the tuned threshold |
| `--no-one-owner` | disable "each S2/S3 record goes to at most one S1" |
| `--workers N` | processes for normalization (default: cores−1) |
| `--test-dir`, `--out-dir`, `--model-dir` | paths |
| `--device cuda` | only tries GPU LightGBM (needs a GPU build; falls back to CPU). Not needed |
| `--use-embeddings` | reserved, not implemented in this version |

## For the team leader: full run on the complete dataset

Run everything from the repo root, on branch `anushka-pipeline`.

**1. Fast mode first (secures a valid leaderboard file):**

```bash
python code/business_entity_resolution/src/run.py --sample 100000 --mode fast 2>&1 | tee run_fast.log
```

**2. Then, if time allows, full mode (higher recall). Write it to a separate folder so the fast
output isn't overwritten:**

```bash
python code/business_entity_resolution/src/run.py --sample 150000 --mode full --out-dir output_full 2>&1 | tee run_full.log
```

**Measured on a 12-core laptop (full mode)**, on a partial test set (about half the real S1 and
about 18% of the real S2/S3):

| country (partial) | S1 / S2+S3 | normalize | blocking | features | candidates/S1 |
|---|---|---|---|---|---|
| France | 135k / 256k | 13 s | 35 s | 70 s | 22.6 |
| India | 423k / 844k | 30 s | 164 s | ~3–4 min | 22.8 |

**Expected on the full test set** (about 1.7M S1 and 10M S2+S3). These are extrapolated estimates;
blocking grows faster than linearly with S2/S3 size:

| | fast | full |
|---|---|---|
| train/validate (100k–150k sample) | ~5 min | ~8 min |
| test total | ~25–40 min | ~45–75 min |
| peak RAM (blocking is chunked over S1) | ~16 GB | ~24–32 GB |

**If it's after ~23:00 IST, run only `--mode fast`.**

Test data is processed one country at a time, so peak RAM is driven by the largest country
(India). If RAM runs out, lower `--workers` (normalization copies chunks to worker
processes) and use `--mode fast`.

**3. Outputs:**

- `output/matching_results.tsv` is the file to upload to the portal.
- `output/candidate_pairs.tsv` is the exact candidate set the model scored, for the zip.

The run validates them automatically at the end and must print `PASS`. To re-validate manually,
with the ID-existence check on:

```bash
python utils/validate_submission.py --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv --test-dir dataset/test --check-ids
```

**4. What to send back to Anushka:** the `.log` file(s), or at least these lines:

- `VALIDATION (t=...)` block: blocking recall, pair precision/recall, **MACRO F0.5**,
  singletons vs non-singletons, per country
- `blocking recall (all sampled S1)`
- per test country: `union` / `final` pair counts and `accepted=... S1 with >=1 match=...`
- stage timings (`[... ] done in ...s`) and the validator result

**Troubleshooting**

- LightGBM "access violation" on Windows: `lightgbm` must be imported before `pandas`. `run.py`
  already does this; keep that import order if you edit it.
- Too slow: use `--mode fast`, or a smaller `--sample` (training is the smaller cost).

## Code layout (`src/`)

| file | role |
|---|---|
| `run.py` | CLI entry point, orchestrates everything |
| `io_utils.py` | TSV reading (`sep="\t"`, all strings, NaN→"") and writing |
| `normalize.py` | transliteration (anyascii), cleaning, abbreviation/legal-form maps (US/IN/FR), state codes, number words, phonetic skeletons |
| `sample.py` | dev sample: S1 entities whose matches are all present + distractors, stratified by match count |
| `blocking.py` | within-country key blocking (families A–D) + top-k pruning |
| `features.py` | rapidfuzz similarity features + rank/gap context features |
| `evaluate.py` | exact macro F0.5 (singleton-aware) + diagnostics; also a CLI scorer |
| `inspect_data.py` | dataset profile |

## Key design decisions

- **Country is an opaque label.** Blocking keys are prefixed by the country string. The ground truth
  has 0 cross-country matches, and an unseen country like France just forms its own blocks. No
  feature encodes country identity.
- **Key blocking instead of all-pairs or kNN.** The most selective keys (name skeleton + house
  number, house number + street word) stay selective at 10M records. Rarer name tokens and a
  concatenated-name skeleton add recall, and bucket-size caps stop explosions.
- **Phonetic skeleton** (drop vowels, ph→f, m→n, c/q→k, …) makes transliterated Indic names
  ("phainems") match English ones ("finance") and absorbs vowel typos.
- **One-owner assignment.** In the training ground truth no S2/S3 record belongs to more than one
  S1 entity (0 of 5.46M checked), so each record is kept only for its highest-probability S1.
- **Threshold tuned for the real metric** (macro F0.5 with singleton credit), not accuracy or pair F1.

## Licenses

pandas (BSD-3), numpy (BSD-3), rapidfuzz (MIT), lightgbm (MIT), anyascii (ISC). No pretrained
models are used.
