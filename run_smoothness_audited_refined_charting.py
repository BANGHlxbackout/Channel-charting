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
from run_diffusion_refined_charting import (
    build_local_scaled_graph,
    compute_neighbor_indices,
)


def compute_smoothness_residual(
    init_embedding: np.ndarray,
    graph: sp.coo_matrix,
    eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray]:
    """Measure how strongly the initial chart violates physical-graph smoothness.

    The residual is unsupervised: it uses only Y0 and W_phys. UE positions are not
    an input, so this score can be used before evaluation without label leakage.
    """
    graph = graph.tocsr()
    embedding = np.asarray(init_embedding, dtype=np.float32)
    if embedding.ndim != 2:
        raise ValueError("init_embedding must be a 2D array")
    if graph.shape != (embedding.shape[0], embedding.shape[0]):
        raise ValueError("graph shape must match init_embedding rows")
    if eps <= 0:
        raise ValueError("eps must be positive")

    degree = np.asarray(graph.sum(axis=1)).ravel().astype(np.float32)
    laplacian_response = degree[:, None] * embedding - graph @ embedding
    residual = np.linalg.norm(laplacian_response, axis=1).astype(np.float32)

    # Normalize by the local chart scale so large coordinates alone do not
    # dominate the smoothness audit.
    coo = graph.tocoo()
    edge_lengths = np.linalg.norm(embedding[coo.row] - embedding[coo.col], axis=1)
    local_scale_sum = np.bincount(
        coo.row,
        weights=edge_lengths.astype(np.float64),
        minlength=embedding.shape[0],
    )
    local_scale_count = np.bincount(coo.row, minlength=embedding.shape[0])
    local_scale = (
        local_scale_sum / np.maximum(local_scale_count, 1)
    ).astype(np.float32)

    normalized = residual / np.maximum(local_scale, eps)
    median = float(np.median(normalized))
    residual_hat = normalized / max(median, eps)
    return normalized.astype(np.float32), residual_hat.astype(np.float32)


def compute_node_fidelity(
    residual_hat: np.ndarray,
    eta: float,
    alpha_min: float,
    alpha_max: float,
) -> np.ndarray:
    """Convert residual into node-wise fidelity weights.

    High residual means Y0 is less consistent with W_phys, so the solver should
    be allowed to move that point farther from its initial coordinate.
    """
    if eta < 0:
        raise ValueError("--eta must be non-negative")
    if not 0 < alpha_min <= alpha_max:
        raise ValueError("--alpha-min must be in (0, alpha_max]")

    residual_hat = np.asarray(residual_hat, dtype=np.float32)
    alpha = 1.0 / (1.0 + eta * residual_hat)
    return np.clip(alpha, alpha_min, alpha_max).astype(np.float32)


def compute_edge_gate(
    graph: sp.coo_matrix,
    residual_hat: np.ndarray,
    use_beta: bool,
    beta_min: float,
    tau_min: float,
    eps: float = 1e-8,
) -> tuple[sp.coo_matrix, np.ndarray]:
    """Optionally downweight diffusion across residual-incompatible edges."""
    if not 0 < beta_min <= 1.0:
        raise ValueError("--beta-min must be in (0, 1]")
    if tau_min <= 0:
        raise ValueError("--tau-min must be positive")
    if eps <= 0:
        raise ValueError("eps must be positive")

    coo = graph.tocoo()
    residual_hat = np.asarray(residual_hat, dtype=np.float32)
    if residual_hat.shape != (coo.shape[0],):
        raise ValueError("residual_hat length must match graph size")

    if not use_beta:
        beta = np.ones_like(coo.data, dtype=np.float32)
        return graph.astype(np.float32).tocoo(), beta

    delta = np.abs(residual_hat[coo.row] - residual_hat[coo.col]).astype(np.float32)
    tau = max(float(np.median(delta)), tau_min)
    compatibility = np.exp(-delta / (tau + eps)).astype(np.float32)
    beta = beta_min + (1.0 - beta_min) * compatibility
    gated = sp.coo_matrix(
        (coo.data.astype(np.float32) * beta, (coo.row, coo.col)),
        shape=coo.shape,
    )
    return gated, beta.astype(np.float32)


def solve_adaptive_graph_harmonic_refinement(
    init_embedding: np.ndarray,
    graph: sp.coo_matrix,
    alpha: np.ndarray,
    lambda_smooth: float,
    iterations: int,
    device: torch.device,
) -> np.ndarray:
    """Solve the alpha-weighted graph-harmonic refinement with Jacobi updates."""
    if lambda_smooth < 0:
        raise ValueError("--lambda-smooth must be non-negative")
    if iterations < 1:
        raise ValueError("--iterations must be positive")

    graph = graph.tocsr()
    deg = np.asarray(graph.sum(axis=1)).ravel().astype(np.float32)
    coo = graph.tocoo()
    embedding = np.asarray(init_embedding, dtype=np.float32)
    alpha = np.asarray(alpha, dtype=np.float32)
    if graph.shape != (embedding.shape[0], embedding.shape[0]):
        raise ValueError("graph shape must match init_embedding rows")
    if alpha.shape != (embedding.shape[0],):
        raise ValueError("alpha length must match init_embedding rows")
    if np.any(alpha <= 0):
        raise ValueError("alpha weights must be positive")

    row = torch.from_numpy(coo.row.astype(np.int64)).to(device)
    col = torch.from_numpy(coo.col.astype(np.int64)).to(device)
    weight = torch.from_numpy(coo.data.astype(np.float32)).to(device)
    degree = torch.from_numpy(deg).to(device)
    alpha_t = torch.from_numpy(alpha.astype(np.float32, copy=False)).to(device)
    y0 = torch.from_numpy(embedding.astype(np.float32, copy=False)).to(device)
    z = y0.clone()

    for iteration in range(1, iterations + 1):
        acc = torch.zeros_like(z)
        acc.index_add_(0, row, weight[:, None] * z[col])
        z = (alpha_t[:, None] * y0 + lambda_smooth * acc) / (
            alpha_t[:, None] + lambda_smooth * degree[:, None]
        )
        if iteration in {1, 10, 50, 100, iterations}:
            print(f"Adaptive Jacobi iteration {iteration}/{iterations}", flush=True)

    return z.detach().cpu().numpy()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Smoothness-audited adaptive graph-harmonic diffusion refinement."
    )
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA1)
    parser.add_argument("--distance-run", type=Path, required=True)
    parser.add_argument("--init-embedding", type=Path, required=True)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/smoothness_audited_refined_charting"),
    )
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--graph-k", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=512)
    parser.add_argument("--lambda-smooth", type=float, default=0.15)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--metric-subset-fraction", type=float, default=0.2)
    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument("--alpha-min", type=float, default=0.25)
    parser.add_argument("--alpha-max", type=float, default=1.0)
    parser.add_argument("--use-beta", action="store_true")
    parser.add_argument("--beta-min", type=float, default=0.2)
    parser.add_argument("--tau-min", type=float, default=0.1)
    args = parser.parse_args()

    if args.graph_k > args.top_k:
        raise ValueError("--graph-k must be <= --top-k")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    started = time.time()
    device = torch.device(args.device)
    data = load_data1(args.data_path, seed=args.seed)
    idx = select_indices(len(data.h_by_array), args.stride, args.max_samples)
    # Positions are kept out of the refinement path and used only after Z exists.
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

    residual, residual_hat = compute_smoothness_residual(init_embedding, graph)
    # Alpha is the main mechanism under test: high residual lowers fidelity to Y0.
    alpha = compute_node_fidelity(
        residual_hat,
        eta=args.eta,
        alpha_min=args.alpha_min,
        alpha_max=args.alpha_max,
    )
    # Beta is optional and only changes edge diffusion strength, not graph support.
    adaptive_graph, beta = compute_edge_gate(
        graph,
        residual_hat,
        use_beta=args.use_beta,
        beta_min=args.beta_min,
        tau_min=args.tau_min,
    )
    embedding = solve_adaptive_graph_harmonic_refinement(
        init_embedding,
        adaptive_graph,
        alpha=alpha,
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

    displacement = np.linalg.norm(embedding - init_embedding, axis=1)
    high_cut = np.percentile(residual_hat, 80)
    low_cut = np.percentile(residual_hat, 20)
    high_mask = residual_hat >= high_cut
    low_mask = residual_hat <= low_cut

    mode = "alpha_beta" if args.use_beta else "alpha_only"
    run_name = (
        f"stride{args.stride}_n{len(idx)}_{mode}_k{args.graph_k}_"
        f"lambda{args.lambda_smooth:g}_eta{args.eta:g}_"
        f"amin{args.alpha_min:g}_bmin{args.beta_min:g}_seed{args.seed}"
    )
    out_dir = args.out_dir / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "protocol": "transductive_unsupervised_refinement",
        "method": "Smoothness-audited adaptive graph-harmonic diffusion refinement",
        "graph_source": "ADP/time fused geodesic physical graph",
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
        "eta": float(args.eta),
        "alpha_min": float(args.alpha_min),
        "alpha_max": float(args.alpha_max),
        "use_beta": bool(args.use_beta),
        "beta_min": float(args.beta_min),
        "tau_min": float(args.tau_min),
        "graph_edges": int(graph.nnz),
        "adaptive_graph_edges": int(adaptive_graph.nnz),
        "metric_subset_fraction": float(args.metric_subset_fraction),
        "metric_subset_size": int(subset_size),
        "uses_positions_for_refinement": False,
        "uses_positions_for_parameter_selection": False,
        "uses_positions_for_evaluation_only": True,
        "baseline_metrics": init_metrics,
        "baseline_localization_errors_after_affine": init_errors,
        "metrics": metrics,
        "localization_errors_after_affine": errors,
        "residual_summary": {
            "mean": float(np.mean(residual_hat)),
            "median": float(np.median(residual_hat)),
            "p80": float(high_cut),
            "p95": float(np.percentile(residual_hat, 95)),
            "max": float(np.max(residual_hat)),
        },
        "alpha_summary": {
            "mean": float(np.mean(alpha)),
            "median": float(np.median(alpha)),
            "min": float(np.min(alpha)),
            "max": float(np.max(alpha)),
        },
        "beta_summary": {
            "mean": float(np.mean(beta)),
            "median": float(np.median(beta)),
            "min": float(np.min(beta)),
            "max": float(np.max(beta)),
        },
        "mechanism_diagnostic": {
            "high_residual_displacement_mean": float(np.mean(displacement[high_mask])),
            "low_residual_displacement_mean": float(np.mean(displacement[low_mask])),
            "high_residual_count": int(np.sum(high_mask)),
            "low_residual_count": int(np.sum(low_mask)),
        },
        "elapsed_s": float(time.time() - started),
    }
    np.savez(
        out_dir / "embedding.npz",
        indices=idx,
        positions=pos,
        embedding=embedding,
        init_embedding=init_embedding,
        residual=residual,
        residual_hat=residual_hat,
        alpha=alpha,
    )
    write_json(out_dir / "result.json", result)
    title = (
        f"Smoothness-Audited {mode}\n"
        f"CT={metrics['ct']:.3f}, TW={metrics['tw']:.3f}, KS={metrics['ks']:.3f}"
    )
    plot_chart(out_dir / "chart.png", pos, embedding, title)
    print("RESULT", json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
