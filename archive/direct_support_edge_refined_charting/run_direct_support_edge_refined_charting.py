from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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
    solve_graph_harmonic_refinement,
)


def summarize_values(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        raise ValueError("cannot summarize an empty array")
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "min": float(np.min(values)),
        "p5": float(np.percentile(values, 5)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def build_direct_support_graph(
    physical_graph: sp.coo_matrix,
    geodesic_distances: np.ndarray,
    fused_distances: np.ndarray,
    rho: float = 0.6,
    eps: float = 1e-6,
    tau_min: float = 1e-6,
    preserve_energy: bool = False,
) -> tuple[sp.coo_matrix, dict[str, object]]:
    if not 0.0 <= rho <= 1.0:
        raise ValueError("rho must be in [0, 1]")
    if eps <= 0.0:
        raise ValueError("eps must be positive")
    if tau_min <= 0.0:
        raise ValueError("tau_min must be positive")

    coo = physical_graph.tocoo()
    d_geo = np.asarray(geodesic_distances[coo.row, coo.col], dtype=np.float32)
    d_fused = np.asarray(fused_distances[coo.row, coo.col], dtype=np.float32)
    q = np.maximum(np.log((d_fused + eps) / (d_geo + eps)), 0.0).astype(np.float32)

    tau_q_raw = float(np.median(q))
    tau_q_used = max(tau_q_raw, tau_min)
    reliability = (rho + (1.0 - rho) * np.exp(-q / tau_q_used)).astype(np.float32)

    original_weights = np.asarray(coo.data, dtype=np.float32)
    raw_weights = (original_weights * reliability).astype(np.float32)
    original_weight_sum = float(np.sum(original_weights, dtype=np.float64))
    raw_weight_sum = float(np.sum(raw_weights, dtype=np.float64))
    weight_ratio = raw_weight_sum / max(original_weight_sum, eps)

    energy_scale = 1.0
    new_weights = raw_weights
    if preserve_energy:
        energy_scale = original_weight_sum / max(raw_weight_sum, eps)
        new_weights = (raw_weights * energy_scale).astype(np.float32)

    refined_graph = sp.coo_matrix((new_weights, (coo.row, coo.col)), shape=coo.shape)

    original_degree = np.asarray(physical_graph.tocsr().sum(axis=1)).ravel()
    refined_degree = np.asarray(refined_graph.tocsr().sum(axis=1)).ravel()
    nonzero_degree = original_degree > 0.0
    degree_ratio = refined_degree[nonzero_degree] / original_degree[nonzero_degree]

    stats = {
        "num_edges": int(coo.nnz),
        "rho": float(rho),
        "eps": float(eps),
        "tau_min": float(tau_min),
        "tau_q_raw_median": tau_q_raw,
        "tau_q_used": float(tau_q_used),
        "q_summary": summarize_values(q),
        "reliability_summary": summarize_values(reliability),
        "original_weight_sum": original_weight_sum,
        "raw_weight_sum": raw_weight_sum,
        "raw_weight_ratio": float(weight_ratio),
        "energy_preserving": bool(preserve_energy),
        "energy_scale": float(energy_scale),
        "refined_weight_sum": float(np.sum(new_weights, dtype=np.float64)),
        "degree_ratio_summary": summarize_values(degree_ratio),
    }
    return refined_graph.tocoo(), stats


def evaluate_embedding(
    positions: np.ndarray,
    init_embedding: np.ndarray,
    embedding: np.ndarray,
    metric_subset_fraction: float,
) -> tuple[dict[str, float], dict[str, float], dict[str, float], dict[str, float], int]:
    subset_size = max(1, int(len(positions) * metric_subset_fraction))
    subset_indices = random.sample(range(len(positions)), subset_size)
    init_metrics = chart_metrics(positions[subset_indices], init_embedding[subset_indices])
    init_errors = localization_error_summary(positions, init_embedding)
    metrics = chart_metrics(positions[subset_indices], embedding[subset_indices])
    errors = localization_error_summary(positions, embedding)
    return init_metrics, init_errors, metrics, errors, subset_size


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Direct-support-aware physical edge filtering for graph-harmonic channel chart refinement."
    )
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA1)
    parser.add_argument("--distance-run", type=Path, required=True)
    parser.add_argument("--init-embedding", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("results/direct_support_edge_refined_charting"))
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--graph-k", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=512)
    parser.add_argument("--lambda-smooth", type=float, default=0.15)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--rho", type=float, default=0.6)
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--tau-min", type=float, default=1e-6)
    parser.add_argument(
        "--mode",
        choices=("diagnostic", "raw", "energy_preserving", "lower_lambda"),
        default="raw",
    )
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

    geodesic = np.load(args.distance_run / "geodesic.npy", mmap_mode="r")
    fused = np.load(args.distance_run / "fused.npy", mmap_mode="r")
    neighbor_cache = args.distance_run / f"geodesic_top{args.top_k}.npy"
    neighbors = compute_neighbor_indices(geodesic, args.top_k, device, neighbor_cache)
    physical_graph = build_local_scaled_graph(geodesic, neighbors, args.graph_k)

    raw_graph, raw_stats = build_direct_support_graph(
        physical_graph,
        geodesic,
        fused,
        rho=args.rho,
        eps=args.eps,
        tau_min=args.tau_min,
        preserve_energy=False,
    )

    lambda_used = args.lambda_smooth
    solver_graph = raw_graph
    mode_description = "Direct-support reliability weights with original lambda"
    if args.mode == "energy_preserving":
        solver_graph, raw_stats = build_direct_support_graph(
            physical_graph,
            geodesic,
            fused,
            rho=args.rho,
            eps=args.eps,
            tau_min=args.tau_min,
            preserve_energy=True,
        )
        mode_description = "Direct-support reliability weights rescaled to preserve total graph energy"
    elif args.mode == "lower_lambda":
        solver_graph = physical_graph
        lambda_used = args.lambda_smooth * float(raw_stats["raw_weight_ratio"])
        mode_description = "Original physical graph with lambda reduced by the raw reliability weight ratio"

    run_name = (
        f"stride{args.stride}_n{len(idx)}_direct_support_{args.mode}_"
        f"k{args.graph_k}_lambda{args.lambda_smooth:g}_rho{args.rho:g}_seed{args.seed}"
    )
    out_dir = args.out_dir / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    result: dict[str, object] = {
        "paper_context": "Direct-support edge reliability feasibility test on ADP/time channel chart refinement",
        "method": "Direct-Support-Aware Physical Edge Filtering",
        "mode": args.mode,
        "mode_description": mode_description,
        "dataset": str(args.data_path),
        "distance_run": str(args.distance_run),
        "init_embedding": str(args.init_embedding),
        "run_name": run_name,
        "num_samples": int(len(idx)),
        "stride": int(args.stride),
        "seed": int(args.seed),
        "graph_k": int(args.graph_k),
        "top_k": int(args.top_k),
        "lambda_smooth_requested": float(args.lambda_smooth),
        "lambda_smooth_used": float(lambda_used),
        "iterations": int(args.iterations),
        "graph_edges": int(solver_graph.nnz),
        "direct_support_stats": raw_stats,
        "uses_positions_for_refinement": False,
        "uses_positions_for_parameter_selection": False,
        "uses_positions_for_evaluation_only": True,
        "metric_subset_fraction": float(args.metric_subset_fraction),
        "elapsed_s": None,
    }

    if args.mode != "diagnostic":
        embedding = solve_graph_harmonic_refinement(
            init_embedding,
            solver_graph,
            lambda_smooth=lambda_used,
            iterations=args.iterations,
            device=device,
        )
        init_metrics, init_errors, metrics, errors, subset_size = evaluate_embedding(
            pos,
            init_embedding,
            embedding,
            args.metric_subset_fraction,
        )
        result.update(
            {
                "metric_subset_size": int(subset_size),
                "baseline_metrics": init_metrics,
                "baseline_localization_errors_after_affine": init_errors,
                "metrics": metrics,
                "localization_errors_after_affine": errors,
            }
        )
        np.savez(out_dir / "embedding.npz", indices=idx, positions=pos, embedding=embedding)
        plot_chart(
            out_dir / "chart.png",
            pos,
            embedding,
            f"Direct Support {args.mode}\nCT={metrics['ct']:.3f}, TW={metrics['tw']:.3f}, KS={metrics['ks']:.3f}",
        )

    result["elapsed_s"] = float(time.time() - t0)
    write_json(out_dir / "result.json", result)
    print("RESULT", json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
