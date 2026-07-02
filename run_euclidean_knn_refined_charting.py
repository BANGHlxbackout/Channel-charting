from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.distance import cdist

from common_dichasus import DEFAULT_DATA1, load_data1
from common_metrics import chart_metrics, localization_error_summary, write_json
from run_adp_fused import select_indices
from run_adp_siamese import plot_chart
from run_diffusion_refined_charting import (
    build_local_scaled_graph,
    compute_neighbor_indices,
    solve_graph_harmonic_refinement,
)


def compute_embedding_distances(embedding: np.ndarray, block_size: int = 2048) -> np.ndarray:
    if block_size <= 0:
        raise ValueError("--distance-block-size must be positive")

    embedding = np.asarray(embedding, dtype=np.float32)
    n = embedding.shape[0]
    distances = np.empty((n, n), dtype=np.float32)
    for start in range(0, n, block_size):
        stop = min(start + block_size, n)
        distances[start:stop] = cdist(embedding[start:stop], embedding, metric="euclidean").astype(np.float32)
        print(f"Euclidean distance rows {stop}/{n}", flush=True)
    np.fill_diagonal(distances, 0.0)
    return distances


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Euclidean kNN graph ablation for graph-harmonic diffusion refined charting."
    )
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA1)
    parser.add_argument("--init-embedding", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("results/euclidean_knn_refined_charting"))
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--graph-k", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=512)
    parser.add_argument("--lambda-smooth", type=float, default=0.15)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--distance-block-size", type=int, default=2048)
    parser.add_argument("--metric-subset-fraction", type=float, default=0.2)
    args = parser.parse_args()

    if args.graph_k > args.top_k:
        raise ValueError("--graph-k must be <= --top-k")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    t0 = time.time()
    device = torch.device(args.device)
    data = load_data1(args.data_path, seed=args.seed)
    idx = select_indices(len(data.h_by_array), args.stride, args.max_samples)
    pos = data.ue_pos[idx]

    init = np.load(args.init_embedding)
    init_indices = init["indices"]
    if len(init_indices) != len(idx) or not np.array_equal(init_indices, idx):
        raise ValueError("init embedding indices do not match requested data selection")
    init_embedding = init["embedding"].astype(np.float32)

    distances = compute_embedding_distances(init_embedding, block_size=args.distance_block_size)
    neighbor_cache = args.out_dir / f"euclidean_top{args.top_k}_stride{args.stride}_n{len(idx)}_seed{args.seed}.npy"
    neighbors = compute_neighbor_indices(distances, args.top_k, device, neighbor_cache)
    graph = build_local_scaled_graph(distances, neighbors, args.graph_k)
    embedding = solve_graph_harmonic_refinement(
        init_embedding,
        graph,
        lambda_smooth=args.lambda_smooth,
        iterations=args.iterations,
        device=device,
    )

    subset_size = max(1, int(len(pos) * args.metric_subset_fraction))
    subset_indices = random.sample(range(len(pos)), subset_size)
    init_metrics = chart_metrics(pos[subset_indices], init_embedding[subset_indices])
    init_errors = localization_error_summary(pos, init_embedding)
    metrics = chart_metrics(pos[subset_indices], embedding[subset_indices])
    errors = localization_error_summary(pos, embedding)

    run_name = (
        f"stride{args.stride}_n{len(idx)}_euclidean_k{args.graph_k}_"
        f"lambda{args.lambda_smooth:g}_seed{args.seed}"
    )
    out_dir = args.out_dir / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "paper_context": "Euclidean graph ablation on the ADP-native DICHASUS channel charting dataset",
        "method": "Graph-harmonic diffusion refinement with Euclidean kNN graph on the initial chart",
        "graph_source": "init_embedding_euclidean",
        "dataset": str(args.data_path),
        "init_embedding": str(args.init_embedding),
        "run_name": run_name,
        "num_samples": int(len(idx)),
        "stride": int(args.stride),
        "seed": int(args.seed),
        "graph_k": int(args.graph_k),
        "top_k": int(args.top_k),
        "lambda_smooth": float(args.lambda_smooth),
        "iterations": int(args.iterations),
        "graph_edges": int(graph.nnz),
        "metric_subset_fraction": float(args.metric_subset_fraction),
        "metric_subset_size": int(subset_size),
        "baseline_metrics": init_metrics,
        "baseline_localization_errors_after_affine": init_errors,
        "metrics": metrics,
        "localization_errors_after_affine": errors,
        "elapsed_s": float(time.time() - t0),
    }
    np.savez(out_dir / "embedding.npz", indices=idx, positions=pos, embedding=embedding)
    write_json(out_dir / "result.json", result)
    plot_chart(
        out_dir / "chart.png",
        pos,
        embedding,
        f"Euclidean kNN Refined\nCT={metrics['ct']:.3f}, TW={metrics['tw']:.3f}, KS={metrics['ks']:.3f}",
    )
    print("RESULT", json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
