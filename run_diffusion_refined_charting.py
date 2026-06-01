from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
import scipy.sparse as sp
import torch

from common_dichasus import DEFAULT_DATA1, load_data1
from common_metrics import chart_metrics, localization_error_summary, write_json
from run_adp_fused import select_indices
from run_adp_siamese import plot_chart


def compute_neighbor_indices(distances: np.ndarray, top_k: int, device: torch.device, cache_path: Path) -> np.ndarray:
    if cache_path.exists():
        neighbors = np.load(cache_path)
        if neighbors.shape == (distances.shape[0], top_k):
            return neighbors

    print(f"computing geodesic top-{top_k} neighbors on {device}", flush=True)
    with torch.no_grad():
        d = torch.from_numpy(distances).to(device)
        _, idx = torch.topk(d, k=top_k + 1, largest=False, dim=1)
        neighbors = idx[:, 1:].to(torch.int32).cpu().numpy()
        del d
        if device.type == "cuda":
            torch.cuda.empty_cache()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, neighbors)
    return neighbors


def build_local_scaled_graph(distances: np.ndarray, neighbors: np.ndarray, graph_k: int) -> sp.coo_matrix:
    n = distances.shape[0]
    rows = np.repeat(np.arange(n), graph_k)
    cols = neighbors[:, :graph_k].reshape(-1)
    local_dist = np.asarray(distances[rows, cols], dtype=np.float32)
    sigma = np.asarray(distances[np.arange(n), neighbors[:, graph_k - 1]], dtype=np.float32)
    denom = np.sqrt(np.maximum(sigma[rows], 1e-6) * np.maximum(sigma[cols], 1e-6))
    weights = np.exp(-np.square(local_dist / denom)).astype(np.float32)
    directed = sp.coo_matrix((weights, (rows, cols)), shape=(n, n)).tocsr()
    return directed.maximum(directed.T).tocoo()


def solve_graph_harmonic_refinement(
    init_embedding: np.ndarray,
    graph: sp.coo_matrix,
    lambda_smooth: float,
    iterations: int,
    device: torch.device,
) -> np.ndarray:
    """Solve min_Z ||Z-Y||^2 + lambda * sum_ij w_ij ||Z_i-Z_j||^2 by GPU Jacobi iterations."""
    graph = graph.tocsr()
    deg = np.asarray(graph.sum(axis=1)).ravel().astype(np.float32)
    coo = graph.tocoo()

    row = torch.from_numpy(coo.row.astype(np.int64)).to(device)
    col = torch.from_numpy(coo.col.astype(np.int64)).to(device)
    weight = torch.from_numpy(coo.data.astype(np.float32)).to(device)
    degree = torch.from_numpy(deg).to(device)
    y0 = torch.from_numpy(init_embedding.astype(np.float32, copy=False)).to(device)
    z = y0.clone()

    for iteration in range(1, iterations + 1):
        acc = torch.zeros_like(z)
        acc.index_add_(0, row, weight[:, None] * z[col])
        z = (y0 + lambda_smooth * acc) / (1.0 + lambda_smooth * degree[:, None])
        if iteration in {1, 10, 50, 100, iterations}:
            print(f"Jacobi iteration {iteration}/{iterations}", flush=True)

    return z.detach().cpu().numpy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA1)
    parser.add_argument("--distance-run", type=Path, required=True)
    parser.add_argument("--init-embedding", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("results/diffusion_refined_charting"))
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--graph-k", type=int, default=64)
    parser.add_argument("--top-k", type=int, default=512)
    parser.add_argument("--lambda-smooth", type=float, default=0.3)
    parser.add_argument("--iterations", type=int, default=200)
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

    saved = np.load(args.distance_run / "distance_data.npz")
    distance_indices = saved["indices"]
    if len(distance_indices) != len(idx) or not np.array_equal(distance_indices, idx):
        raise ValueError("distance-run indices do not match requested data selection")

    init = np.load(args.init_embedding)
    init_indices = init["indices"]
    if len(init_indices) != len(idx) or not np.array_equal(init_indices, idx):
        raise ValueError("init embedding indices do not match requested data selection")
    init_embedding = init["embedding"].astype(np.float32)

    distances = np.load(args.distance_run / "geodesic.npy").astype(np.float32)
    if not np.all(np.isfinite(distances)):
        raise ValueError("distance matrix contains non-finite values")
    neighbor_cache = args.distance_run / f"geodesic_top{args.top_k}.npy"
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
        f"stride{args.stride}_n{len(idx)}_k{args.graph_k}_"
        f"lambda{args.lambda_smooth:g}_seed{args.seed}"
    )
    out_dir = args.out_dir / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "paper_context": "new method on the ADP-native DICHASUS channel charting dataset",
        "method": "Graph-harmonic diffusion refinement of ADP/time geodesic channel charts",
        "idea": (
            "Estimate the UE manifold Laplace-Beltrami operator from the ADP/time geodesic graph, "
            "then solve a graph-harmonic Tikhonov problem that denoises the neural FCF chart while "
            "preserving the intrinsic channel manifold."
        ),
        "dataset": str(args.data_path),
        "distance_run": str(args.distance_run),
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
    plot_chart(out_dir / "chart.png", pos, embedding, f"Diffusion Refined\nCT={metrics['ct']:.3f}, TW={metrics['tw']:.3f}, KS={metrics['ks']:.3f}")
    print("RESULT", json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
