from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
HELDOUT_DIR = ROOT / "held-out"
for path in (ROOT, HELDOUT_DIR):
    path_text = str(path)
    if path_text not in sys.path:
        sys.path.insert(0, path_text)

from common_heldout import deterministic_subset_indices, heldout_stage_dir
from common_metrics import chart_metrics, localization_error_summary, write_json
from run_cross_fitted_gated_refinement import compute_cross_fitted_predictions
from run_diffusion_refined_charting import build_local_scaled_graph, compute_neighbor_indices
from run_diffusion_refined_heldout import load_train_inputs


def save_teacher_outputs(
    out_dir: Path,
    train_indices: np.ndarray,
    train_positions: np.ndarray,
    y_train: np.ndarray,
    teacher_train: np.ndarray,
    fold_id: np.ndarray,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_dir / "teacher_train.npz",
        indices=train_indices,
        positions=train_positions,
        init_embedding=y_train,
        teacher_embedding=teacher_train,
        teacher_residual=teacher_train - y_train,
        fold_id=fold_id,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate CF-HRD cross-fitted no-self-anchor harmonic teacher "
            "targets on the held-out training split."
        )
    )
    parser.add_argument(
        "--distance-run",
        type=Path,
        default=Path("results/heldout/seed10/adp_fused_train"),
        help="Held-out train ADP/time graph stage containing distance_data.npz and geodesic_train.npy.",
    )
    parser.add_argument(
        "--init-embedding",
        type=Path,
        default=Path("results/heldout/seed10/adp_siamese/embedding_train.npz"),
        help="Held-out train FCF embedding npz.",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("results/CF_HRD"))
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--graph-k", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=512)
    parser.add_argument("--lambda-teacher", type=float, default=0.15)
    parser.add_argument("--num-folds", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--metric-subset-fraction", type=float, default=0.2)
    args = parser.parse_args()

    if args.graph_k > args.top_k:
        raise ValueError("--graph-k must be <= --top-k")

    started = time.time()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    train_indices, train_positions, y_train, geodesic_train = load_train_inputs(
        args.distance_run,
        args.init_embedding,
        seed=args.seed,
    )

    neighbor_cache = args.distance_run / f"geodesic_top{args.top_k}.npy"
    neighbors = compute_neighbor_indices(
        geodesic_train,
        args.top_k,
        device,
        neighbor_cache,
    )
    graph = build_local_scaled_graph(geodesic_train, neighbors, args.graph_k)

    teacher_train, fold_id = compute_cross_fitted_predictions(
        y_train,
        graph,
        lambda_smooth=args.lambda_teacher,
        num_folds=args.num_folds,
        iterations=args.iterations,
        seed=args.seed,
        device=device,
    )
    residual = teacher_train - y_train
    residual_norm = np.linalg.norm(residual, axis=1).astype(np.float32)

    subset = deterministic_subset_indices(
        len(train_positions),
        args.metric_subset_fraction,
        args.seed,
    )
    baseline_metrics = chart_metrics(train_positions[subset], y_train[subset])
    teacher_metrics = chart_metrics(train_positions[subset], teacher_train[subset])
    baseline_errors = localization_error_summary(train_positions, y_train)
    teacher_errors = localization_error_summary(train_positions, teacher_train)

    stage_dir = heldout_stage_dir(args.out_dir, args.seed, "cross_fitted_teacher")
    save_teacher_outputs(
        stage_dir,
        train_indices=train_indices,
        train_positions=train_positions,
        y_train=y_train,
        teacher_train=teacher_train,
        fold_id=fold_id,
    )

    result = {
        "protocol": "cf_hrd_cross_fitted_teacher_train_only",
        "description": (
            "Step 1-2 validation artifact for CF-HRD: load held-out train FCF outputs "
            "and ADP/time train graph, then generate out-of-fold harmonic teacher "
            "coordinates with each target node's own fidelity anchor removed."
        ),
        "distance_run": str(args.distance_run),
        "init_embedding": str(args.init_embedding),
        "output_npz": str(stage_dir / "teacher_train.npz"),
        "seed": int(args.seed),
        "num_train": int(len(train_indices)),
        "graph_k": int(args.graph_k),
        "top_k": int(args.top_k),
        "lambda_teacher": float(args.lambda_teacher),
        "num_folds": int(args.num_folds),
        "iterations": int(args.iterations),
        "graph_edges": int(graph.nnz),
        "metric_subset_fraction": float(args.metric_subset_fraction),
        "metric_subset_size": int(len(subset)),
        "baseline_train_metrics": baseline_metrics,
        "teacher_train_metrics": teacher_metrics,
        "baseline_train_localization_errors": baseline_errors,
        "teacher_train_localization_errors": teacher_errors,
        "teacher_residual_norm": {
            "mean": float(np.mean(residual_norm)),
            "median": float(np.median(residual_norm)),
            "p90": float(np.percentile(residual_norm, 90)),
            "p95": float(np.percentile(residual_norm, 95)),
            "max": float(np.max(residual_norm)),
        },
        "fold_sizes": [
            int(np.sum(fold_id == fold_index)) for fold_index in range(args.num_folds)
        ],
        "heldout_test_metrics_available": False,
        "elapsed_s": float(time.time() - started),
    }
    write_json(stage_dir / "result.json", result)
    print("RESULT", json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
