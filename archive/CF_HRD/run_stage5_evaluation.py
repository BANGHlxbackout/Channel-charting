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

from common_heldout import (
    apply_affine_transform,
    fit_affine_transform,
    heldout_stage_dir,
    localization_error_summary_aligned,
)
from common_metrics import chart_metrics, write_json


VARIANTS = ("y0_only", "q_only", "y0_q")


def load_npz(path: Path) -> np.lib.npyio.NpzFile:
    if not path.exists():
        raise FileNotFoundError(path)
    return np.load(path)


def assert_same_indices(name: str, left: np.ndarray, right: np.ndarray) -> None:
    if not np.array_equal(left, right):
        raise ValueError(f"{name} indices do not match")


def evaluate_embedding(
    train_positions: np.ndarray,
    test_positions: np.ndarray,
    train_embedding: np.ndarray,
    test_embedding: np.ndarray,
) -> dict[str, object]:
    transform = fit_affine_transform(train_positions, train_embedding)
    train_aligned = apply_affine_transform(train_embedding, transform)
    test_aligned = apply_affine_transform(test_embedding, transform)
    return {
        "train_localization_errors": localization_error_summary_aligned(
            train_positions,
            train_aligned,
        ),
        "test_localization_errors": localization_error_summary_aligned(
            test_positions,
            test_aligned,
        ),
        "test_metrics": chart_metrics(test_positions, test_embedding),
        "affine_transform_shape": [int(x) for x in transform.shape],
    }


def deltas_against_baseline(
    method_errors: dict[str, float],
    baseline_errors: dict[str, float],
) -> dict[str, float]:
    return {
        key: float(method_errors[key] - baseline_errors[key])
        for key in ("mean_m", "median_m", "p90_m", "p95_m", "max_m")
    }


def metric_deltas_against_baseline(
    method_metrics: dict[str, float],
    baseline_metrics: dict[str, float],
) -> dict[str, float]:
    return {
        key: float(method_metrics[key] - baseline_metrics[key])
        for key in ("ct", "tw", "ks")
    }


def summarize_verdict(results: dict[str, object]) -> dict[str, object]:
    fcf = results["methods"]["fcf"]["test_localization_errors"]
    y0_only = results["methods"]["y0_only"]["test_localization_errors"]
    y0_q = results["methods"]["y0_q"]["test_localization_errors"]
    y0_q_metrics = results["methods"]["y0_q"]["test_metrics"]
    fcf_metrics = results["methods"]["fcf"]["test_metrics"]

    localization_keys = ("mean_m", "p90_m", "p95_m", "max_m")
    improves_fcf = {
        key: bool(y0_q[key] < fcf[key])
        for key in localization_keys
    }
    better_than_y0_only = {
        key: bool(y0_q[key] < y0_only[key])
        for key in localization_keys
    }
    metric_damage = {
        "ct_delta": float(y0_q_metrics["ct"] - fcf_metrics["ct"]),
        "tw_delta": float(y0_q_metrics["tw"] - fcf_metrics["tw"]),
        "ks_delta": float(y0_q_metrics["ks"] - fcf_metrics["ks"]),
    }
    return {
        "y0_q_improves_fcf": improves_fcf,
        "y0_q_better_than_y0_only": better_than_y0_only,
        "y0_q_metric_delta_vs_fcf": metric_damage,
        "positive_stage5_signal": bool(
            any(improves_fcf.values())
            and (y0_q_metrics["ct"] >= fcf_metrics["ct"] - 1e-3)
            and (y0_q_metrics["tw"] >= fcf_metrics["tw"] - 1e-3)
            and (y0_q_metrics["ks"] <= fcf_metrics["ks"] + 1e-3)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate CF-HRD Stage 4 residual students on held-out test data."
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
    parser.add_argument(
        "--residual-student-run",
        type=Path,
        default=Path("results/CF_HRD/seed10/residual_student"),
    )
    parser.add_argument("--out-dir", type=Path, default=Path("results/CF_HRD"))
    parser.add_argument("--seed", type=int, default=10)
    args = parser.parse_args()

    started = time.time()
    fcf_train = load_npz(args.embedding_train)
    fcf_test = load_npz(args.embedding_test)
    train_indices = fcf_train["indices"].astype(np.int64)
    test_indices = fcf_test["indices"].astype(np.int64)
    train_positions = fcf_train["positions"].astype(np.float32)
    test_positions = fcf_test["positions"].astype(np.float32)

    methods: dict[str, dict[str, object]] = {}
    methods["fcf"] = evaluate_embedding(
        train_positions,
        test_positions,
        fcf_train["embedding"].astype(np.float32),
        fcf_test["embedding"].astype(np.float32),
    )
    methods["fcf"]["train_embedding"] = str(args.embedding_train)
    methods["fcf"]["test_embedding"] = str(args.embedding_test)

    for variant in VARIANTS:
        train_path = args.residual_student_run / f"{variant}_embedding_train.npz"
        test_path = args.residual_student_run / f"{variant}_embedding_test.npz"
        train_data = load_npz(train_path)
        test_data = load_npz(test_path)
        assert_same_indices(
            f"{variant} train",
            train_indices,
            train_data["indices"].astype(np.int64),
        )
        assert_same_indices(
            f"{variant} test",
            test_indices,
            test_data["indices"].astype(np.int64),
        )
        method_result = evaluate_embedding(
            train_positions,
            test_positions,
            train_data["embedding"].astype(np.float32),
            test_data["embedding"].astype(np.float32),
        )
        method_result["train_embedding"] = str(train_path)
        method_result["test_embedding"] = str(test_path)
        method_result["test_localization_delta_vs_fcf"] = deltas_against_baseline(
            method_result["test_localization_errors"],
            methods["fcf"]["test_localization_errors"],
        )
        method_result["test_metric_delta_vs_fcf"] = metric_deltas_against_baseline(
            method_result["test_metrics"],
            methods["fcf"]["test_metrics"],
        )
        methods[variant] = method_result

    result: dict[str, object] = {
        "protocol": "cf_hrd_stage5_heldout_evaluation",
        "description": (
            "Fit affine alignment on train embeddings only, apply the same transform "
            "to held-out test embeddings, and report held-out localization and chart "
            "metrics for FCF and CF-HRD residual-student variants."
        ),
        "embedding_train": str(args.embedding_train),
        "embedding_test": str(args.embedding_test),
        "residual_student_run": str(args.residual_student_run),
        "seed": int(args.seed),
        "num_train": int(len(train_indices)),
        "num_test": int(len(test_indices)),
        "uses_train_affine_for_test": True,
        "uses_test_positions_for_evaluation_only": True,
        "methods": methods,
        "elapsed_s": float(time.time() - started),
    }
    result["verdict"] = summarize_verdict(result)

    out_dir = heldout_stage_dir(args.out_dir, args.seed, "stage5_evaluation")
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "result.json", result)
    print("RESULT", json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
