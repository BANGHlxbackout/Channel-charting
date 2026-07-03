from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HELDOUT_DIR = ROOT / "held-out"
for path in (ROOT, HELDOUT_DIR):
    path_text = str(path)
    if path_text not in sys.path:
        sys.path.insert(0, path_text)

from common_heldout import heldout_stage_dir
from common_metrics import write_json


def load_npz(path: Path) -> np.lib.npyio.NpzFile:
    if not path.exists():
        raise FileNotFoundError(path)
    return np.load(path)


def assert_same_indices(name: str, left: np.ndarray, right: np.ndarray) -> None:
    if not np.array_equal(left, right):
        raise ValueError(f"{name} indices do not match")


def fit_standardizer(x_train: np.ndarray, eps: float = 1e-8) -> tuple[np.ndarray, np.ndarray]:
    mean = np.mean(x_train, axis=0).astype(np.float32)
    std = np.std(x_train, axis=0).astype(np.float32)
    std = np.where(std > eps, std, 1.0).astype(np.float32)
    return mean, std


def apply_standardizer(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((x - mean[None, :]) / std[None, :]).astype(np.float32)


def fit_ridge(
    x_train: np.ndarray,
    y_train: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    if alpha < 0:
        raise ValueError("--ridge-alpha must be non-negative")
    ones = np.ones((x_train.shape[0], 1), dtype=np.float32)
    design = np.hstack([ones, x_train]).astype(np.float64)
    target = y_train.astype(np.float64)
    penalty = np.eye(design.shape[1], dtype=np.float64) * float(alpha)
    penalty[0, 0] = 0.0
    coef = np.linalg.solve(design.T @ design + penalty, design.T @ target)
    intercept = coef[0].astype(np.float32)
    weights = coef[1:].astype(np.float32)
    return intercept, weights


def predict_ridge(x: np.ndarray, intercept: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return (x.astype(np.float32) @ weights + intercept[None, :]).astype(np.float32)


def residual_fit_summary(predicted: np.ndarray, target: np.ndarray) -> dict[str, float]:
    error = predicted - target
    norm_error = np.linalg.norm(error, axis=1)
    target_norm = np.linalg.norm(target, axis=1)
    return {
        "mse": float(np.mean(np.square(error))),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "vector_norm_mean": float(np.mean(norm_error)),
        "vector_norm_median": float(np.median(norm_error)),
        "vector_norm_p95": float(np.percentile(norm_error, 95)),
        "target_norm_mean": float(np.mean(target_norm)),
        "target_norm_p95": float(np.percentile(target_norm, 95)),
        "relative_norm_mean": float(np.mean(norm_error) / max(float(np.mean(target_norm)), 1e-12)),
    }


def vector_norm_summary(values: np.ndarray) -> dict[str, float]:
    norms = np.linalg.norm(values, axis=1)
    return {
        "mean": float(np.mean(norms)),
        "median": float(np.median(norms)),
        "p90": float(np.percentile(norms, 90)),
        "p95": float(np.percentile(norms, 95)),
        "max": float(np.max(norms)),
    }


def validate_predictions(
    variant: str,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    y0_train: np.ndarray,
    y0_test: np.ndarray,
    delta_train: np.ndarray,
    delta_test: np.ndarray,
) -> None:
    if delta_train.shape != y0_train.shape:
        raise ValueError(f"{variant}: delta_train shape does not match Y0_train")
    if delta_test.shape != y0_test.shape:
        raise ValueError(f"{variant}: delta_test shape does not match Y0_test")
    for name, values in {
        "train_indices": train_indices,
        "test_indices": test_indices,
        "delta_train": delta_train,
        "delta_test": delta_test,
        "embedding_train": y0_train + delta_train,
        "embedding_test": y0_test + delta_test,
    }.items():
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{variant}: {name} contains non-finite values")


def save_variant_embeddings(
    out_dir: Path,
    variant: str,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    y0_train: np.ndarray,
    y0_test: np.ndarray,
    delta_train: np.ndarray,
    delta_test: np.ndarray,
) -> tuple[Path, Path]:
    train_path = out_dir / f"{variant}_embedding_train.npz"
    test_path = out_dir / f"{variant}_embedding_test.npz"
    np.savez(
        train_path,
        indices=train_indices,
        init_embedding=y0_train,
        predicted_residual=delta_train,
        embedding=y0_train + delta_train,
    )
    np.savez(
        test_path,
        indices=test_indices,
        init_embedding=y0_test,
        predicted_residual=delta_test,
        embedding=y0_test + delta_test,
    )
    return train_path, test_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train CF-HRD Stage 4 residual students and export train/test predictions."
    )
    parser.add_argument(
        "--teacher-train",
        type=Path,
        default=Path("results/CF_HRD/seed10/cross_fitted_teacher/teacher_train.npz"),
    )
    parser.add_argument(
        "--context",
        type=Path,
        default=Path("results/CF_HRD/seed10/landmark_context/context.npz"),
    )
    parser.add_argument(
        "--embedding-train",
        type=Path,
        default=Path("results/heldout/seed10/adp_siamese/embedding_train.npz"),
    )
    parser.add_argument(
        "--embedding-test",
        type=Path,
        default=Path("results/heldout/seed10/adp_siamese/embedding_test.npz"),
    )
    parser.add_argument("--out-dir", type=Path, default=Path("results/CF_HRD"))
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    args = parser.parse_args()

    started = time.time()
    teacher = load_npz(args.teacher_train)
    context = load_npz(args.context)
    emb_train = load_npz(args.embedding_train)
    emb_test = load_npz(args.embedding_test)

    train_indices = teacher["indices"].astype(np.int64)
    test_indices = context["test_indices"].astype(np.int64)
    assert_same_indices("teacher/context train", train_indices, context["train_indices"].astype(np.int64))
    assert_same_indices("teacher/embedding train", train_indices, emb_train["indices"].astype(np.int64))
    assert_same_indices("context/embedding test", test_indices, emb_test["indices"].astype(np.int64))

    y0_train = teacher["init_embedding"].astype(np.float32)
    y0_test = emb_test["embedding"].astype(np.float32)
    if not np.array_equal(y0_train, emb_train["embedding"].astype(np.float32)):
        raise ValueError("teacher init_embedding does not match held-out train embedding")
    teacher_residual = teacher["teacher_residual"].astype(np.float32)
    q_train = context["q_train"].astype(np.float32)
    q_test = context["q_test"].astype(np.float32)

    if y0_train.shape != teacher_residual.shape:
        raise ValueError("teacher residual shape does not match Y0_train")
    if q_train.shape[0] != y0_train.shape[0]:
        raise ValueError("q_train row count does not match Y0_train")
    if q_test.shape[0] != y0_test.shape[0]:
        raise ValueError("q_test row count does not match Y0_test")

    variants = {
        "y0_only": (y0_train, y0_test),
        "q_only": (q_train, q_test),
        "y0_q": (
            np.hstack([y0_train, q_train]).astype(np.float32),
            np.hstack([y0_test, q_test]).astype(np.float32),
        ),
    }

    out_dir = heldout_stage_dir(args.out_dir, args.seed, "residual_student")
    out_dir.mkdir(parents=True, exist_ok=True)

    result: dict[str, object] = {
        "protocol": "cf_hrd_stage4_residual_student",
        "description": (
            "Train fixed ridge residual students on train-only teacher residuals "
            "and export train/test chart predictions. No test labels, test positions, "
            "or test statistics are used for training, scaling, or model selection."
        ),
        "teacher_train": str(args.teacher_train),
        "context": str(args.context),
        "embedding_train": str(args.embedding_train),
        "embedding_test": str(args.embedding_test),
        "seed": int(args.seed),
        "num_train": int(len(train_indices)),
        "num_test": int(len(test_indices)),
        "ridge_alpha": float(args.ridge_alpha),
        "uses_test_labels_for_training": False,
        "uses_test_positions": False,
        "uses_test_statistics_for_scaling": False,
        "variants": {},
    }

    for variant, (x_train_raw, x_test_raw) in variants.items():
        feature_mean, feature_std = fit_standardizer(x_train_raw)
        x_train = apply_standardizer(x_train_raw, feature_mean, feature_std)
        x_test = apply_standardizer(x_test_raw, feature_mean, feature_std)
        intercept, weights = fit_ridge(x_train, teacher_residual, alpha=args.ridge_alpha)
        delta_train = predict_ridge(x_train, intercept, weights)
        delta_test = predict_ridge(x_test, intercept, weights)
        validate_predictions(
            variant,
            train_indices,
            test_indices,
            y0_train,
            y0_test,
            delta_train,
            delta_test,
        )
        train_path, test_path = save_variant_embeddings(
            out_dir,
            variant,
            train_indices,
            test_indices,
            y0_train,
            y0_test,
            delta_train,
            delta_test,
        )
        result["variants"][variant] = {
            "train_embedding": str(train_path),
            "test_embedding": str(test_path),
            "train_residual_fit": residual_fit_summary(delta_train, teacher_residual),
            "delta_norm_summary": {
                "train": vector_norm_summary(delta_train),
                "test": vector_norm_summary(delta_test),
                "teacher_train": vector_norm_summary(teacher_residual),
            },
            "model_config": {
                "model": "closed_form_ridge_regression",
                "ridge_alpha": float(args.ridge_alpha),
                "feature_set": variant,
                "feature_dim": int(x_train_raw.shape[1]),
                "standardization": "mean/std fitted on train only; applied unchanged to test",
                "intercept_regularized": False,
                "seed": int(args.seed),
            },
            "feature_standardization": {
                "mean_shape": [int(x) for x in feature_mean.shape],
                "std_shape": [int(x) for x in feature_std.shape],
                "std_min": float(np.min(feature_std)),
                "std_max": float(np.max(feature_std)),
            },
        }

    result["elapsed_s"] = float(time.time() - started)
    write_json(out_dir / "result.json", result)
    print("RESULT", json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
