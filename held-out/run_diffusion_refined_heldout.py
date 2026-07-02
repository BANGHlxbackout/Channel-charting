from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from common_dichasus import p2_split_indices
from common_heldout import deterministic_subset_indices, heldout_stage_dir
from common_metrics import chart_metrics, localization_error_summary, write_json
from run_adp_siamese import plot_chart
from run_diffusion_refined_charting import (
    build_local_scaled_graph,
    compute_neighbor_indices,
    solve_graph_harmonic_refinement,
)


def load_train_inputs(
    distance_run: Path,
    init_embedding_path: Path,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    distance_data = np.load(distance_run / "distance_data.npz")
    init_data = np.load(init_embedding_path)
    distance_indices = distance_data["indices"]
    test_indices = distance_data["test_indices"]
    num_samples = len(distance_indices) + len(test_indices)
    expected_train, expected_test = p2_split_indices(num_samples, seed=seed)
    if not np.array_equal(distance_indices, expected_train) or not np.array_equal(
        test_indices,
        expected_test,
    ):
        raise ValueError("saved split does not match requested seed")

    init_indices = init_data["indices"]
    if not np.array_equal(distance_indices, init_indices):
        raise ValueError("distance and embedding training indices do not match")

    geodesic = np.load(distance_run / "geodesic_train.npy").astype(np.float32)
    expected_shape = (len(init_indices), len(init_indices))
    if geodesic.shape != expected_shape:
        raise ValueError("training geodesic shape does not match training indices")
    if not np.all(np.isfinite(geodesic)):
        raise ValueError("training geodesic contains non-finite values")

    return (
        init_indices,
        init_data["positions"].astype(np.float32),
        init_data["embedding"].astype(np.float32),
        geodesic,
    )


def save_train_refinement(
    out_dir: Path,
    indices: np.ndarray,
    positions: np.ndarray,
    embedding: np.ndarray,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_dir / "embedding_train.npz",
        indices=indices,
        positions=positions,
        embedding=embedding,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run train-only graph-harmonic refinement for held-out protocol."
    )
    parser.add_argument("--distance-run", type=Path, required=True)
    parser.add_argument("--init-embedding", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("results/heldout"))
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--graph-k", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=512)
    parser.add_argument("--lambda-smooth", type=float, default=0.15)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--metric-subset-fraction", type=float, default=0.2)
    args = parser.parse_args()

    if args.graph_k > args.top_k:
        raise ValueError("--graph-k must be <= --top-k")

    started = time.time()
    device = torch.device(args.device)
    torch.manual_seed(args.seed)

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
    z_train = solve_graph_harmonic_refinement(
        y_train,
        graph,
        lambda_smooth=args.lambda_smooth,
        iterations=args.iterations,
        device=device,
    )

    subset = deterministic_subset_indices(
        len(train_positions),
        args.metric_subset_fraction,
        args.seed,
    )
    baseline_metrics = chart_metrics(train_positions[subset], y_train[subset])
    metrics = chart_metrics(train_positions[subset], z_train[subset])
    baseline_errors = localization_error_summary(train_positions, y_train)
    errors = localization_error_summary(train_positions, z_train)

    out_dir = heldout_stage_dir(args.out_dir, args.seed, "diffusion_train")
    save_train_refinement(out_dir, train_indices, train_positions, z_train)

    result = {
        "protocol": "train_graph_diagnostic",
        "distance_run": str(args.distance_run),
        "init_embedding": str(args.init_embedding),
        "seed": int(args.seed),
        "num_train": int(len(train_indices)),
        "graph_k": int(args.graph_k),
        "top_k": int(args.top_k),
        "lambda_smooth": float(args.lambda_smooth),
        "iterations": int(args.iterations),
        "graph_edges": int(graph.nnz),
        "metric_subset_fraction": float(args.metric_subset_fraction),
        "metric_subset_size": int(len(subset)),
        "baseline_train_metrics": baseline_metrics,
        "refined_train_metrics": metrics,
        "baseline_train_localization_errors": baseline_errors,
        "refined_train_localization_errors": errors,
        "heldout_diffusion_metrics_available": False,
        "elapsed_s": float(time.time() - started),
    }
    write_json(out_dir / "result.json", result)
    plot_chart(
        out_dir / "chart.png",
        train_positions,
        z_train,
        f"Train Diffusion Diagnostic\nCT={metrics['ct']:.3f}, TW={metrics['tw']:.3f}",
    )
    print("RESULT", json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
