from __future__ import annotations

from pathlib import Path

import numpy as np


def heldout_stage_dir(root: Path, seed: int, stage: str) -> Path:
    return root / f"seed{seed}" / stage


def validate_split(
    n_samples: int,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> None:
    if len(np.intersect1d(train_idx, test_idx)) != 0:
        raise ValueError("train/test split overlaps")
    expected = np.arange(n_samples)
    if not np.array_equal(np.union1d(train_idx, test_idx), expected):
        raise ValueError("train/test split does not cover the dataset")


def _pad_affine(values: np.ndarray) -> np.ndarray:
    return np.hstack(
        [values, np.ones((values.shape[0], 1), dtype=values.dtype)]
    )


def fit_affine_transform(reference: np.ndarray, embedding: np.ndarray) -> np.ndarray:
    transform, *_ = np.linalg.lstsq(
        _pad_affine(embedding),
        _pad_affine(reference),
        rcond=None,
    )
    return transform


def apply_affine_transform(
    embedding: np.ndarray,
    transform: np.ndarray,
) -> np.ndarray:
    return (_pad_affine(embedding) @ transform)[:, :-1]


def localization_error_summary_aligned(
    reference: np.ndarray,
    aligned_embedding: np.ndarray,
) -> dict[str, float]:
    errors = np.linalg.norm(aligned_embedding - reference, axis=1)
    return {
        "mean_m": float(np.mean(errors)),
        "median_m": float(np.median(errors)),
        "p90_m": float(np.percentile(errors, 90)),
        "p95_m": float(np.percentile(errors, 95)),
        "max_m": float(np.max(errors)),
    }


def deterministic_subset_indices(
    n_samples: int,
    fraction: float,
    seed: int,
) -> np.ndarray:
    subset_size = max(1, int(n_samples * fraction))
    rng = np.random.default_rng(seed)
    return np.sort(
        rng.choice(n_samples, size=subset_size, replace=False).astype(np.int64)
    )
