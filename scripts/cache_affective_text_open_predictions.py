"""Build a fully open AffectiveText prediction cache with out-of-fold regressors."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import Normalizer

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import EMOTION_NAMES, load_affective_text

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def macro_pearson(a: np.ndarray, b: np.ndarray) -> float:
    vals = []
    for j in range(a.shape[1]):
        aj = a[:, j]
        bj = b[:, j]
        if np.std(aj) <= 1e-12 or np.std(bj) <= 1e-12:
            continue
        vals.append(float(np.corrcoef(aj, bj)[0, 1]))
    return float(np.mean(vals)) if vals else float("nan")


def fit_predict_fold(
    train_texts: list[str],
    test_texts: list[str],
    train_targets: np.ndarray,
    n_components: int,
    n_neighbors: int,
) -> np.ndarray:
    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        sublinear_tf=True,
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.95,
        stop_words="english",
    )
    x_train = vectorizer.fit_transform(train_texts)
    x_test = vectorizer.transform(test_texts)

    max_rank = min(x_train.shape[0] - 1, x_train.shape[1] - 1)
    if max_rank >= 2:
        rank = min(n_components, max_rank)
        svd = TruncatedSVD(n_components=rank, random_state=0)
        normalizer = Normalizer(copy=False)
        x_train = normalizer.fit_transform(svd.fit_transform(x_train))
        x_test = normalizer.transform(svd.transform(x_test))
    else:
        x_train = x_train.toarray()
        x_test = x_test.toarray()

    knn = KNeighborsRegressor(
        n_neighbors=min(n_neighbors, len(train_texts)),
        weights="distance",
        metric="minkowski",
        p=2,
    )
    knn.fit(x_train, train_targets)
    return np.asarray(knn.predict(x_test), dtype=float)


def build_open_predictions(
    headlines: list[str],
    raw_scores: np.ndarray,
    n_splits: int,
    n_components: int,
    n_neighbors: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    n = len(headlines)
    preds = np.zeros_like(raw_scores, dtype=float)
    folds = np.full(n, -1, dtype=int)
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    global_mean = raw_scores.mean(axis=0)

    for fold_id, (train_idx, test_idx) in enumerate(splitter.split(headlines)):
        train_texts = [headlines[i] for i in train_idx]
        test_texts = [headlines[i] for i in test_idx]
        train_targets = raw_scores[train_idx]
        fold_preds = fit_predict_fold(
            train_texts=train_texts,
            test_texts=test_texts,
            train_targets=train_targets,
            n_components=n_components,
            n_neighbors=n_neighbors,
        )
        fold_preds = np.clip(fold_preds, 0.0, None)
        zero_rows = fold_preds.sum(axis=1) <= 1e-12
        if np.any(zero_rows):
            fold_preds[zero_rows] = global_mean
        preds[test_idx] = fold_preds
        folds[test_idx] = fold_id
        log.info("Finished fold %d/%d", fold_id + 1, n_splits)

    return preds, folds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/raw/AffectiveText.Semeval.2007")
    parser.add_argument("--output", default="data/processed/affective_text_open_oof_predictions.jsonl")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--n-components", type=int, default=128)
    parser.add_argument("--n-neighbors", type=int, default=25)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output_path = Path(args.output)
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists: {output_path}")

    data = load_affective_text(args.data_dir)
    ids = data["ids"]
    headlines = data["headlines"]
    raw_scores = np.asarray(data["raw_scores"], dtype=float)
    if args.limit is not None:
        ids = ids[:args.limit]
        headlines = headlines[:args.limit]
        raw_scores = raw_scores[:args.limit]

    pred_scores, folds = build_open_predictions(
        headlines=headlines,
        raw_scores=raw_scores,
        n_splits=args.n_splits,
        n_components=args.n_components,
        n_neighbors=args.n_neighbors,
        seed=args.seed,
    )

    macro_r = macro_pearson(raw_scores, pred_scores)
    flat_r = float(np.corrcoef(raw_scores.reshape(-1), pred_scores.reshape(-1))[0, 1])
    log.info(
        "Open fallback predictor quality: macro Pearson=%.3f, flattened Pearson=%.3f",
        macro_r,
        flat_r,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for idx, headline, scores, fold_id in zip(ids, headlines, pred_scores, folds):
            row = {
                "id": idx,
                "headline": headline,
                "emotions": EMOTION_NAMES,
                "scores": [float(x) for x in scores],
                "provider": "open_fallback",
                "model": "tfidf_svd_knn_oof",
                "fold": int(fold_id),
                "builder": {
                    "n_splits": int(args.n_splits),
                    "n_components": int(args.n_components),
                    "n_neighbors": int(args.n_neighbors),
                    "seed": int(args.seed),
                },
                "notes": "Deterministic out-of-fold TF-IDF+SVD+kNN regression fallback.",
            }
            f.write(json.dumps(row, ensure_ascii=True) + "\n")

    log.info("Finished. Predictions cached at %s", output_path)


if __name__ == "__main__":
    main()
