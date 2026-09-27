"""End-to-end pipeline: data -> normalize -> blocking -> features -> LightGBM
-> one-owner assignment + F0.5-tuned threshold -> output TSVs.

Examples
  # dev on a sample (no test prediction)
  python src/run.py --data-dir ../../dataset --sample 20000 --skip-test
  # full run: train on a 150k-entity sample of train, predict the full test set
  python src/run.py --data-dir ../../dataset --sample 150000 --out-dir ../../output
"""
# NOTE: lightgbm must be imported before pandas. On some Windows setups importing
# pandas first makes LightGBM crash ("access violation" at Dataset construction).
import lightgbm as lgb  # noqa: I001

import argparse
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from blocking import CONFIGS, generate_candidates  # noqa: E402
from evaluate import report  # noqa: E402
from features import build_features  # noqa: E402
from io_utils import Timer, read_ground_truth, read_source, write_id_lists  # noqa: E402
from normalize import normalize_df  # noqa: E402
from sample import build_sample  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))


# ---------------------------------------------------------------- decisions
def decide(cand, p, t, one_owner=True):
    """Boolean mask of accepted pairs. one_owner: each candidate record is kept
    only for the S1 with its highest probability."""
    keep = p >= t
    if one_owner:
        s = pd.Series(p)
        best = s.groupby(cand["i2"].values).transform("max").values
        keep &= p >= best
    return keep


def macro_f05_fast(i1, y, keep, n_true, n_s1):
    """Vectorized macro F0.5 over n_s1 entities (entity index = i1).
    n_true[i] = number of true matches of entity i (incl. ones outside candidates)."""
    tp = np.bincount(i1, weights=(keep & y).astype(float), minlength=n_s1)
    npred = np.bincount(i1, weights=keep.astype(float), minlength=n_s1)
    p = np.divide(tp, npred, out=np.zeros(n_s1), where=npred > 0)
    r = np.divide(tp, n_true, out=np.zeros(n_s1), where=n_true > 0)
    f = np.divide(1.25 * p * r, 0.25 * p + r, out=np.zeros(n_s1), where=(p + r) > 0)
    f[(n_true == 0) & (npred == 0)] = 1.0
    return f.mean()


def tune_threshold(i1, y, p, cand, n_true, n_s1, one_owner):
    best = (0.5, -1)
    for t in np.round(np.arange(0.05, 0.99, 0.01), 2):
        s = macro_f05_fast(i1, y, decide(cand, p, t, one_owner), n_true, n_s1)
        if s > best[1]:
            best = (float(t), s)
    return best


def to_lists(cand, s1, other, mask=None):
    sub = cand if mask is None else cand[mask]
    ids1 = s1["entity_id"].values[sub["i1"].values]
    ids2 = other["entity_id"].values[sub["i2"].values]
    out = {}
    for a, b in zip(ids1, ids2):
        out.setdefault(a, []).append(b)
    return out


# ---------------------------------------------------------------- model
FEATS = None
LGB_PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=40,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                  verbose=-1, num_threads=0, seed=0)


def train_lgb(X, y, Xv=None, yv=None, rounds=2000, device="cpu"):
    params = dict(LGB_PARAMS)
    if device == "cuda":
        params["device_type"] = "gpu"  # needs a GPU-enabled LightGBM build
    dtr = lgb.Dataset(X, y)
    try:
        if Xv is not None:
            dv = lgb.Dataset(Xv, yv, reference=dtr)
            m = lgb.train(params, dtr, rounds, valid_sets=[dv],
                          callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(200)])
        else:
            m = lgb.train(params, dtr, rounds)
    except lgb.basic.LightGBMError as e:
        if device == "cuda":
            print(f"  [model] GPU LightGBM unavailable ({e}); falling back to CPU")
            return train_lgb(X, y, Xv, yv, rounds, "cpu")
        raise
    return m


def print_examples(cand, s1, other, y, keep, p, n=20):
    cols = ["business_name", "business_address"]
    def show(title, mask):
        idx = np.flatnonzero(mask)[:n]
        print(f"\n--- {title} ({mask.sum():,} total, showing {len(idx)})")
        for k in idx:
            a, b = s1.iloc[cand["i1"].values[k]], other.iloc[cand["i2"].values[k]]
            print(f"  p={p[k]:.3f} | {a[cols[0]]} | {a[cols[1]]}\n"
                  f"          -> {b['entity_id'][:2]} | {b[cols[0]]} | {b[cols[1]]}")
    show("FALSE MERGES", keep & ~y)
    show("MISSED MATCHES (in candidates)", ~keep & y)


# ---------------------------------------------------------------- stages
def prepare(s1, other, mode, workers):
    with Timer("normalize"):
        normalize_df(s1, workers)
        normalize_df(other, workers)
    with Timer("blocking"):
        cand = generate_candidates(s1, other, mode)
    with Timer("features"):
        X = build_features(cand, s1, other)
    return cand, X


def train_stage(a):
    tr = os.path.join(a.data_dir, "train")
    with Timer("load train"):
        s1 = read_source(os.path.join(tr, "train_source1.tsv"))
        other = pd.concat([read_source(os.path.join(tr, "train_source2.tsv")),
                           read_source(os.path.join(tr, "train_source3.tsv"))], ignore_index=True)
        gt = read_ground_truth(os.path.join(tr, "train_ground_truth.tsv"))
        print(f"  S1={len(s1):,} S2+S3={len(other):,} GT={len(gt):,}")
    if a.sample:
        with Timer("sample"):
            s1, other, gt = build_sample(s1, other, gt, a.sample, seed=a.seed)
    else:
        s1 = s1[s1["entity_id"].isin(gt)].reset_index(drop=True)

    cand, X = prepare(s1, other, a.mode, a.workers)
    ids1 = s1["entity_id"].values
    ids2 = other["entity_id"].values
    y = np.array([b in gt.get(x, ()) for x, b in
                  zip(ids1[cand["i1"].values], ids2[cand["i2"].values])])
    n_true = np.array([len(gt.get(x, ())) for x in ids1], dtype=float)

    # group split by S1 entity
    rng = np.random.default_rng(a.seed)
    is_val_s1 = rng.random(len(s1)) < a.val_frac
    vmask = is_val_s1[cand["i1"].values]
    print(f"  pairs: train={(~vmask).sum():,} val={vmask.sum():,}  pos rate={y.mean():.3f}")

    with Timer("train (holdout)"):
        m = train_lgb(X[~vmask], y[~vmask], X[vmask], y[vmask], device=a.device)
        best_iter = m.best_iteration or m.current_iteration()
        print(f"  best_iteration={best_iter}")

    # ---- validation diagnostics
    cv = cand[vmask].reset_index(drop=True)
    pv = m.predict(X[vmask], num_iteration=best_iter)
    yv = y[vmask]
    val_s1_idx = np.flatnonzero(is_val_s1)
    remap = -np.ones(len(s1), dtype=np.int64)
    remap[val_s1_idx] = np.arange(len(val_s1_idx))
    i1v = remap[cv["i1"].values]
    for oo in ([True, False] if a.one_owner else [False]):
        t, s = tune_threshold(i1v, yv, pv, cv, n_true[val_s1_idx], len(val_s1_idx), oo)
        print(f"  tuned threshold (one_owner={oo}): t={t:.2f}  macro F0.5={s:.5f}")
        if oo == a.one_owner:
            thr = t
    if a.threshold is not None:
        thr = a.threshold
    keep = decide(cv, pv, thr, a.one_owner)
    val_ids = ids1[val_s1_idx]
    preds = to_lists(cv, s1, other, keep)
    cands = to_lists(cv, s1, other)
    country_of = dict(zip(s1["entity_id"].values, s1["country"].values))
    report(preds, gt, val_ids, country_of, cands, title=f"VALIDATION (t={thr:.2f})")
    # blocking recall on all sampled entities
    all_c = to_lists(cand, s1, other)
    tot = sum(len(v) for v in gt.values() if v)
    hit = sum(len(set(all_c.get(k, ())) & v) for k, v in gt.items())
    print(f"blocking recall (all sampled S1): {hit / max(tot, 1):.5f}")
    print_examples(cv, s1, other, yv, keep, pv)
    imp = sorted(zip(m.feature_importance("gain"), X.columns), reverse=True)[:15]
    print("\ntop features (gain):", ", ".join(f"{c}={g:.0f}" for g, c in imp))

    # ---- final model on train + val
    with Timer("train (final, train+val)"):
        final = train_lgb(X, y, rounds=max(50, int(best_iter * 1.1)), device=a.device)
    os.makedirs(a.model_dir, exist_ok=True)
    final.save_model(os.path.join(a.model_dir, "lgb_model.txt"))
    with open(os.path.join(a.model_dir, "config.json"), "w") as f:
        json.dump({"threshold": thr, "one_owner": a.one_owner, "mode": a.mode,
                   "features": list(X.columns)}, f, indent=1)
    return final, thr


def test_stage(a, model, thr):
    te = a.test_dir or os.path.join(a.data_dir, "test")
    with Timer("load test"):
        s1 = read_source(os.path.join(te, "test_source1.tsv"))
        other = pd.concat([read_source(os.path.join(te, "test_source2.tsv")),
                           read_source(os.path.join(te, "test_source3.tsv"))], ignore_index=True)
        print(f"  S1={len(s1):,} S2+S3={len(other):,}")
    all_ids = s1["entity_id"].tolist()
    matches, cands = {}, {}
    key1 = s1["country"].str.strip().str.lower()
    key2 = other["country"].str.strip().str.lower()
    # process one country at a time (blocking is within-country anyway) to bound memory
    for c in sorted(set(key1)):
        a1 = s1[key1 == c].reset_index(drop=True)
        o1 = other[key2 == c].reset_index(drop=True)
        print(f"\n==== test country={c!r}: S1={len(a1):,} S2+S3={len(o1):,}", flush=True)
        if len(o1) == 0:
            continue
        cand, X = prepare(a1, o1, a.mode, a.workers)
        with Timer("predict"):
            p = model.predict(X)
        keep = decide(cand, p, thr, a.one_owner)
        cands.update(to_lists(cand, a1, o1))
        matches.update(to_lists(cand, a1, o1, keep))
        print(f"  candidates={len(cand):,} accepted={keep.sum():,} "
              f"S1 with >=1 match={len(to_lists(cand, a1, o1, keep)):,}/{len(a1):,}")
        del cand, X, a1, o1
    mp = os.path.join(a.out_dir, "matching_results.tsv")
    cp = os.path.join(a.out_dir, "candidate_pairs.tsv")
    write_id_lists(mp, all_ids, matches, "matched_entity_ids")
    write_id_lists(cp, all_ids, cands, "candidate_entity_ids")
    print(f"\nwrote {mp}\nwrote {cp}")
    val = a.validator or os.path.join(REPO, "utils", "validate_submission.py")
    if os.path.exists(val):
        subprocess.run([sys.executable, val, "--matching", mp, "--candidate", cp,
                        "--test-dir", te])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=os.path.join(REPO, "dataset"))
    ap.add_argument("--test-dir", default=None, help="default: <data-dir>/test")
    ap.add_argument("--out-dir", default=os.path.join(REPO, "output"))
    ap.add_argument("--model-dir", default=os.path.join(REPO, "models"))
    ap.add_argument("--sample", type=int, default=0,
                    help="train on N sampled S1 entities (+ all their matches + distractors); 0 = all")
    ap.add_argument("--mode", choices=list(CONFIGS), default="full", help="blocking config; 'fast' = fallback")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--threshold", type=float, default=None, help="override tuned threshold")
    ap.add_argument("--no-one-owner", dest="one_owner", action="store_false")
    ap.add_argument("--skip-test", action="store_true", help="only train/validate")
    ap.add_argument("--use-embeddings", action="store_true",
                    help="reserved; embeddings are not implemented in this version")
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--validator", default=None)
    a = ap.parse_args()
    if a.use_embeddings:
        print("WARNING: --use-embeddings is not implemented in this version; ignoring.")
    print("config:", vars(a), flush=True)
    with Timer("TOTAL"):
        model, thr = train_stage(a)
        if not a.skip_test:
            test_stage(a, model, thr)


if __name__ == "__main__":
    main()
