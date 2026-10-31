"""Frozen SC2-inspired adaptive-fidelity Stage-2 pilot.

The method changes only the per-node fidelity on the existing advisor Stage-2
Tikhonov update.  Stage-1 artifacts, D_geo, the Stage-2 graph weights, and the
FCF chart are treated as fixed inputs.  Physical positions are loaded only for
the offline diagnostics and evaluation at the end of the run.
"""

from __future__ import annotations

import argparse
import json
import platform
import random
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.sparse as sp
import torch
from scipy.stats import rankdata, spearmanr

# Running the entry by path puts this methods subdirectory first on sys.path;
# add the repository root so the existing root-level solver/metrics modules are
# imported exactly as in the advisor entry point.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common_metrics import affine_transform, chart_metrics, localization_error_summary, write_json
from run_diffusion_refined_charting import build_local_scaled_graph
from run_diffusion_refined_charting import solve_graph_harmonic_refinement


DISTANCE_RUN = ROOT / "results/adp_old_fused/stride1_n20997_seed10"
INIT_EMBEDDING = ROOT / "results/adp_old_siamese/stride1_n20997_seed10/embedding.npz"
BASELINE_RUN = ROOT / "results/diffusion_refined_charting/stride1_n20997_k128_lambda0.15_seed10"
OUTPUT_ROOT = ROOT / "results/sc2_stage2_adaptive_fidelity_pilot"

SEED = 10
TOP_K = 512
LOCAL_K = 64
GRAPH_K = 128
LAMBDA_SMOOTH = 0.15
ITERATIONS = 200
EPS = 1e-8
METRIC_FRACTION = 0.2
UNIFORM_MAX_ABS_GATE = 5e-4
UNIFORM_MEAN_ABS_GATE = 5e-5
UNIFORM_METRIC_TOLERANCE = 1e-5


def build_symmetric_union(
    rows: np.ndarray,
    cols: np.ndarray,
    values: np.ndarray,
    n: int,
    *,
    disagreement_tolerance: float = 1e-12,
) -> tuple[sp.csr_matrix, dict[str, float | int | bool]]:
    """Build the sparse symmetric union without summing reciprocal duplicates."""

    rows = np.asarray(rows, dtype=np.int64).reshape(-1)
    cols = np.asarray(cols, dtype=np.int64).reshape(-1)
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if not (rows.size == cols.size == values.size):
        raise ValueError("rows, cols, and values must have equal length")
    if np.any(rows == cols):
        raise ValueError("symmetric compatibility union cannot contain self edges")
    if np.any(rows < 0) or np.any(rows >= n) or np.any(cols < 0) or np.any(cols >= n):
        raise ValueError("compatibility edge index out of range")
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("compatibility values must be finite and nonnegative")

    keys = np.minimum(rows, cols) * n + np.maximum(rows, cols)
    order = np.argsort(keys, kind="mergesort")
    sorted_keys = keys[order]
    sorted_rows = rows[order]
    sorted_cols = cols[order]
    sorted_values = values[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_keys)) + 1]
    ends = np.r_[starts[1:], sorted_keys.size]
    group_sizes = ends - starts
    if np.any(group_sizes > 2):
        raise ValueError("directed neighbor relations contain duplicate edges")

    reciprocal_mask = group_sizes == 2
    disagreement_values = np.zeros(int(np.sum(reciprocal_mask)), dtype=np.float64)
    reciprocal_groups = np.flatnonzero(reciprocal_mask)
    for offset, group in enumerate(reciprocal_groups):
        start, end = int(starts[group]), int(ends[group])
        disagreement_values[offset] = float(np.ptp(sorted_values[start:end]))
    max_disagreement = float(np.max(disagreement_values)) if disagreement_values.size else 0.0
    if max_disagreement > disagreement_tolerance:
        raise ValueError(
            "reciprocal compatibility values disagree: "
            f"max={max_disagreement:.6g} > {disagreement_tolerance:.6g}"
        )

    # One value per undirected pair, then mirror it.  This preserves a single
    # C value for reciprocal relations instead of accidentally doubling it.
    unique_rows = sorted_rows[starts]
    unique_cols = sorted_cols[starts]
    unique_values = sorted_values[starts]
    sym_rows = np.concatenate([unique_rows, unique_cols])
    sym_cols = np.concatenate([unique_cols, unique_rows])
    sym_values = np.concatenate([unique_values, unique_values])
    union = sp.coo_matrix((sym_values, (sym_rows, sym_cols)), shape=(n, n)).tocsr()
    union.sort_indices()
    checks = {
        "directed_edge_count": int(rows.size),
        "undirected_union_edge_count": int(unique_rows.size),
        "symmetric_nnz": int(union.nnz),
        "reciprocal_directed_pair_count": int(np.sum(reciprocal_mask)),
        "max_directional_disagreement": max_disagreement,
        "finite": bool(np.all(np.isfinite(union.data))),
        "nonnegative": bool(np.all(union.data >= 0)),
        "diagonal_nnz": int(union.diagonal().astype(bool).sum()),
    }
    if checks["diagonal_nnz"] != 0 or not checks["finite"] or not checks["nonnegative"]:
        raise ValueError("invalid symmetric compatibility union")
    return union, checks


def compute_second_order_scores(
    c_directed: np.ndarray,
    union: sp.csr_matrix,
    rows: np.ndarray,
    cols: np.ndarray,
    n: int,
    *,
    eps: float = EPS,
    chunk_size: int = 65536,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return q_FO, H on directed edges, and q_SC2 without a dense product."""

    c_directed = np.asarray(c_directed, dtype=np.float64).reshape(-1)
    rows = np.asarray(rows, dtype=np.int64).reshape(-1)
    cols = np.asarray(cols, dtype=np.int64).reshape(-1)
    if not (c_directed.size == rows.size == cols.size):
        raise ValueError("directed score arrays must have equal length")
    if union.shape != (n, n):
        raise ValueError("compatibility union shape mismatch")

    row_norm = np.sqrt(np.asarray(union.multiply(union).sum(axis=1)).ravel())
    if not np.all(np.isfinite(row_norm)) or np.any(row_norm <= 0):
        raise ValueError("symmetric compatibility rows must have positive finite norms")

    edge_count = rows.size
    h = np.empty(edge_count, dtype=np.float64)
    for start in range(0, edge_count, chunk_size):
        stop = min(edge_count, start + chunk_size)
        left = union[rows[start:stop]]
        right = union[cols[start:stop]]
        dots = np.asarray(left.multiply(right).sum(axis=1)).ravel()
        h[start:stop] = dots / (row_norm[rows[start:stop]] * row_norm[cols[start:stop]] + eps)
    if not np.all(np.isfinite(h)) or np.any(h < -1e-12) or np.any(h > 1.0 + 1e-10):
        raise ValueError("second-order compatibility scores are outside [0, 1]")

    degree = np.bincount(rows, minlength=n)
    if np.any(degree == 0):
        raise ValueError("every node needs at least one directed local neighbor")
    q_fo = np.bincount(rows, weights=c_directed, minlength=n) / degree
    s_directed = c_directed * h
    q_sc2 = np.bincount(rows, weights=s_directed, minlength=n) / degree
    return q_fo, h, q_sc2


def percentile_fidelity(score: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Map a complete score vector to the frozen 0.5 + percentile fidelity."""

    score = np.asarray(score, dtype=np.float64).reshape(-1)
    if score.size == 0 or not np.all(np.isfinite(score)):
        raise ValueError("score must be a non-empty finite vector")
    if score.size == 1:
        percentile = np.array([0.5], dtype=np.float64)
    else:
        percentile = (rankdata(score, method="average") - 1.0) / (score.size - 1.0)
    fidelity = 0.5 + percentile
    return percentile, fidelity


def solve_adaptive_fidelity(
    init_embedding: np.ndarray,
    graph: sp.coo_matrix,
    fidelity: np.ndarray,
    lambda_smooth: float,
    iterations: int,
    device: torch.device,
    *,
    verbose: bool = True,
) -> np.ndarray:
    """Run the frozen node-fidelity Jacobi update in float32 on ``device``."""

    graph = graph.tocsr()
    degree_np = np.asarray(graph.sum(axis=1)).ravel().astype(np.float32)
    coo = graph.tocoo()
    row = torch.from_numpy(coo.row.astype(np.int64)).to(device)
    col = torch.from_numpy(coo.col.astype(np.int64)).to(device)
    weight = torch.from_numpy(coo.data.astype(np.float32)).to(device)
    degree = torch.from_numpy(degree_np).to(device)
    y0 = torch.from_numpy(np.asarray(init_embedding, dtype=np.float32)).to(device)
    a = torch.from_numpy(np.asarray(fidelity, dtype=np.float32)).to(device)
    if y0.ndim != 2 or y0.shape[0] != a.shape[0] or degree.shape[0] != a.shape[0]:
        raise ValueError("embedding, fidelity, and graph dimensions do not match")
    z = y0.clone()
    for iteration in range(1, iterations + 1):
        acc = torch.zeros_like(z)
        acc.index_add_(0, row, weight[:, None] * z[col])
        z = (a[:, None] * y0 + lambda_smooth * acc) / (a[:, None] + lambda_smooth * degree[:, None])
        if verbose and iteration in {1, 10, 50, 100, iterations}:
            print(f"Jacobi iteration {iteration}/{iterations}", flush=True)
    return z.detach().cpu().numpy()


def aligned_errors(reference: np.ndarray, embedding: np.ndarray) -> np.ndarray:
    aligned = affine_transform(reference, embedding)
    return np.linalg.norm(aligned - reference, axis=1)


def tie_summary(values: np.ndarray) -> dict[str, int]:
    values = np.asarray(values)
    _, counts = np.unique(values, return_counts=True)
    tied = counts[counts > 1]
    return {
        "unique_value_count": int(len(counts)),
        "tie_group_count": int(len(tied)),
        "tied_item_count": int(np.sum(tied)) if tied.size else 0,
        "largest_tie_multiplicity": int(np.max(tied)) if tied.size else 1,
    }


def validate_inputs(
    distance_run: Path,
    init_embedding_path: Path,
    baseline_run: Path,
) -> tuple[dict[str, object], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Validate sample order and cache semantics before any method computation."""

    distance_data = np.load(distance_run / "distance_data.npz")
    y0_data = np.load(init_embedding_path)
    z_data = np.load(baseline_run / "embedding.npz")
    baseline_json = json.loads((baseline_run / "result.json").read_text(encoding="utf-8"))
    indices = np.asarray(distance_data["indices"])
    y0_indices = np.asarray(y0_data["indices"])
    z_indices = np.asarray(z_data["indices"])
    y0 = np.asarray(y0_data["embedding"])
    positions = np.asarray(y0_data["positions"])
    z_saved = np.asarray(z_data["embedding"])
    if indices.ndim != 1 or y0_indices.ndim != 1 or z_indices.ndim != 1:
        raise ValueError("input indices must be one-dimensional")
    n = len(indices)
    if n != 20997 or not np.array_equal(indices, y0_indices) or not np.array_equal(indices, z_indices):
        raise ValueError("frozen input sample order or N mismatch")
    if not np.array_equal(np.asarray(z_data["positions"]), positions):
        raise ValueError("saved Stage-2 positions do not match Y0 positions")
    if y0.shape != (n, 2) or z_saved.shape != (n, 2) or positions.shape != (n, 2):
        raise ValueError("frozen embeddings/positions must have shape (20997, 2)")
    if y0.dtype != np.float32 or z_saved.dtype != np.float32 or positions.dtype != np.float32:
        raise ValueError("frozen embeddings and positions must be float32")
    if baseline_json.get("num_samples") != n or baseline_json.get("graph_k") != GRAPH_K:
        raise ValueError("saved Stage-2 JSON does not match frozen graph configuration")
    if float(baseline_json.get("lambda_smooth")) != LAMBDA_SMOOTH or baseline_json.get("iterations") != ITERATIONS:
        raise ValueError("saved Stage-2 JSON does not match frozen lambda/iteration configuration")

    distances_path = distance_run / "geodesic.npy"
    cache_path = distance_run / f"geodesic_top{TOP_K}.npy"
    distances = np.load(distances_path, mmap_mode="r")
    neighbors = np.load(cache_path, mmap_mode="r")
    if distances.shape != (n, n) or distances.dtype != np.float32:
        raise ValueError("D_geo shape or dtype mismatch")
    if neighbors.shape != (n, TOP_K) or not np.issubdtype(neighbors.dtype, np.integer):
        raise ValueError("geodesic top-k cache shape or dtype mismatch")
    if not np.all(np.isfinite(distances)) or not np.all((neighbors >= 0) & (neighbors < n)):
        raise ValueError("D_geo or cache contains invalid values")
    row_ids = np.arange(n, dtype=np.int64)
    neighbors_i64 = np.asarray(neighbors, dtype=np.int64)
    if np.any(neighbors_i64 == row_ids[:, None]):
        raise ValueError("geodesic cache contains self neighbors")
    sorted_neighbors = np.sort(neighbors_i64, axis=1)
    if np.any(np.diff(sorted_neighbors, axis=1) == 0):
        raise ValueError("geodesic cache contains duplicate neighbors")
    cache_distances = np.asarray(distances[row_ids[:, None], neighbors_i64])
    if not np.all(np.diff(cache_distances, axis=1) >= 0):
        raise ValueError("geodesic cache is not distance sorted")

    boundary_rows = np.random.default_rng(SEED).choice(n, size=32, replace=False)
    boundary_records: list[dict[str, float | int | bool]] = []
    boundary_tie_rows = 0
    for row_id in boundary_rows:
        row_dist = np.asarray(distances[int(row_id)], dtype=np.float64)
        nonself_dist = np.delete(row_dist, int(row_id))
        boundary = float(np.partition(nonself_dist, TOP_K - 1)[TOP_K - 1])
        cached_max = float(cache_distances[int(row_id), TOP_K - 1])
        excluded = np.ones(n, dtype=bool)
        excluded[int(row_id)] = False
        excluded[neighbors_i64[int(row_id)]] = False
        excluded_min = float(np.min(row_dist[excluded]))
        tolerance = max(1e-7, abs(boundary) * 1e-7)
        tie_count = int(np.sum(np.abs(row_dist - boundary) <= tolerance))
        boundary_tie_rows += int(tie_count > 1)
        boundary_records.append(
            {
                "row": int(row_id),
                "boundary_distance": boundary,
                "cached_max_distance": cached_max,
                "excluded_min_distance": excluded_min,
                "boundary_tie_count": tie_count,
                "within_boundary_tolerance": bool(
                    cached_max <= boundary + tolerance and excluded_min >= boundary - tolerance
                ),
            }
        )
    if not all(record["within_boundary_tolerance"] for record in boundary_records):
        raise ValueError("sampled top-k cache boundary mismatch")

    audit = {
        "N": n,
        "indices_match": True,
        "distance_shape": list(distances.shape),
        "distance_dtype": str(distances.dtype),
        "cache_shape": list(neighbors.shape),
        "cache_dtype": str(neighbors.dtype),
        "cache_unique_nonself": True,
        "cache_distance_sorted": True,
        "sampled_boundary_rows": [int(x) for x in boundary_rows],
        "sampled_boundary_tie_row_count": int(boundary_tie_rows),
        "sampled_boundary_records": boundary_records,
        "baseline_json": {
            "graph_k": baseline_json.get("graph_k"),
            "lambda_smooth": baseline_json.get("lambda_smooth"),
            "iterations": baseline_json.get("iterations"),
            "device_recorded": baseline_json.get("device"),
        },
    }
    return audit, distances, neighbors_i64, y0, positions, z_saved


def compute_reliability(
    distances: np.ndarray,
    neighbors: np.ndarray,
    y0: np.ndarray,
    *,
    eps: float = EPS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, sp.csr_matrix, dict[str, object]]:
    n = y0.shape[0]
    directed_neighbors = np.asarray(neighbors[:, :LOCAL_K], dtype=np.int64)
    rows = np.repeat(np.arange(n, dtype=np.int64), LOCAL_K)
    cols = directed_neighbors.reshape(-1)
    d_geo = np.asarray(distances[rows, cols], dtype=np.float64)
    d_y = np.linalg.norm(y0[rows] - y0[cols], axis=1).astype(np.float64)
    if not np.all(np.isfinite(d_geo)) or not np.all(np.isfinite(d_y)):
        raise ValueError("local compatibility distances are non-finite")
    log_difference = np.log(d_y + eps) - np.log(d_geo + eps)
    log_s = float(np.median(log_difference))
    residual = np.abs(log_difference - log_s)
    tau = float(np.median(residual) + eps)
    c_directed = np.exp(-residual / tau)
    if not np.all(np.isfinite(c_directed)) or np.any(c_directed < 0):
        raise ValueError("directed compatibility values are invalid")
    union, union_checks = build_symmetric_union(rows, cols, c_directed, n)
    q_fo, h_directed, q_sc2 = compute_second_order_scores(c_directed, union, rows, cols, n, eps=eps)
    if not np.all(np.isfinite(q_fo)) or not np.all(np.isfinite(q_sc2)):
        raise ValueError("node reliability values are invalid")
    summary = {
        "local_k": LOCAL_K,
        "epsilon": eps,
        "log_s": log_s,
        "tau": tau,
        "c_directed_min": float(np.min(c_directed)),
        "c_directed_mean": float(np.mean(c_directed)),
        "c_directed_max": float(np.max(c_directed)),
        "h_directed_min": float(np.min(h_directed)),
        "h_directed_mean": float(np.mean(h_directed)),
        "h_directed_max": float(np.max(h_directed)),
        "s_directed_min": float(np.min(c_directed * h_directed)),
        "s_directed_mean": float(np.mean(c_directed * h_directed)),
        "s_directed_max": float(np.max(c_directed * h_directed)),
        "q_fo": {
            "min": float(np.min(q_fo)),
            "mean": float(np.mean(q_fo)),
            "max": float(np.max(q_fo)),
            **tie_summary(q_fo),
        },
        "q_sc2": {
            "min": float(np.min(q_sc2)),
            "mean": float(np.mean(q_sc2)),
            "max": float(np.max(q_sc2)),
            **tie_summary(q_sc2),
        },
        "symmetric_union_checks": union_checks,
    }
    return q_fo, h_directed, q_sc2, union, summary


def summarize_delta(reference_errors: np.ndarray, variant_errors: np.ndarray) -> dict[str, float | int]:
    """Report positive improvement and positive harm separately vs uniform."""

    delta = np.asarray(reference_errors) - np.asarray(variant_errors)
    improvement = np.maximum(delta, 0.0)
    harm = np.maximum(-delta, 0.0)
    return {
        "net_mean_improvement_m": float(np.mean(delta)),
        "mean_positive_improvement_m": float(np.mean(improvement)),
        "mean_positive_harm_m": float(np.mean(harm)),
        "max_positive_improvement_m": float(np.max(improvement)),
        "max_positive_harm_m": float(np.max(harm)),
        "improved_count": int(np.sum(delta > 0)),
        "harmed_count": int(np.sum(delta < 0)),
        "unchanged_count": int(np.sum(delta == 0)),
        "improved_fraction": float(np.mean(delta > 0)),
        "harmed_fraction": float(np.mean(delta < 0)),
    }


def quintile_diagnostics(
    q_sc2: np.ndarray,
    y0_errors: np.ndarray,
    baseline_errors: np.ndarray,
) -> dict[str, object]:
    order = np.lexsort((np.arange(len(q_sc2), dtype=np.int64), q_sc2))
    bins = np.array_split(order, 5)
    baseline_improvement = y0_errors - baseline_errors
    rows: list[dict[str, object]] = []
    for bin_id, ids in enumerate(bins, start=1):
        rows.append(
            {
                "quintile": bin_id,
                "count": int(len(ids)),
                "q_sc2_min": float(np.min(q_sc2[ids])),
                "q_sc2_max": float(np.max(q_sc2[ids])),
                "y0_mean_error_m": float(np.mean(y0_errors[ids])),
                "y0_median_error_m": float(np.median(y0_errors[ids])),
                "baseline_z_mean_error_m": float(np.mean(baseline_errors[ids])),
                "baseline_refinement_improvement_mean_m": float(np.mean(baseline_improvement[ids])),
            }
        )
    correlations: dict[str, dict[str, float | int]] = {}
    for name, score in (("q_FO", None), ("q_SC2", q_sc2)):
        if name == "q_FO":
            continue
        rho_y0, p_y0 = spearmanr(score, y0_errors)
        rho_improve, p_improve = spearmanr(score, baseline_improvement)
        correlations[name] = {
            "with_y0_error_rho": float(rho_y0),
            "with_y0_error_pvalue": float(p_y0),
            "with_baseline_improvement_rho": float(rho_improve),
            "with_baseline_improvement_pvalue": float(p_improve),
        }
    return {
        "quintile_definition": "stable lexsort by (q_SC2, index), then np.array_split into five bins",
        "quintiles": rows,
        "q_sc2_ties": tie_summary(q_sc2),
        "baseline_improvement_definition": "Y0 aligned error minus saved baseline Z aligned error",
        "correlations": correlations,
    }


def make_diagnostic_correlations(
    q_fo: np.ndarray,
    q_sc2: np.ndarray,
    y0_errors: np.ndarray,
    baseline_errors: np.ndarray,
) -> dict[str, object]:
    baseline_improvement = y0_errors - baseline_errors
    result: dict[str, object] = {}
    for name, score in (("q_FO", q_fo), ("q_SC2", q_sc2)):
        rho_y0, p_y0 = spearmanr(score, y0_errors)
        rho_improve, p_improve = spearmanr(score, baseline_improvement)
        result[name] = {
            "with_y0_error_rho": float(rho_y0),
            "with_y0_error_pvalue": float(p_y0),
            "with_baseline_improvement_rho": float(rho_improve),
            "with_baseline_improvement_pvalue": float(p_improve),
        }
    return result


def plot_diagnostics(
    output_root: Path,
    diagnostic: dict[str, object],
    variant_errors: dict[str, np.ndarray],
) -> None:
    quintiles = diagnostic["quintiles"]
    x = np.array([row["quintile"] for row in quintiles], dtype=np.int64)
    y0_mean = np.array([row["y0_mean_error_m"] for row in quintiles])
    y0_median = np.array([row["y0_median_error_m"] for row in quintiles])
    improvement = np.array(
        [row["baseline_refinement_improvement_mean_m"] for row in quintiles]
    )

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(x, y0_mean, marker="o", label="Y0 mean error")
    ax.plot(x, y0_median, marker="s", label="Y0 median error")
    ax.set_xlabel("SC2 reliability quintile (low to high)")
    ax.set_ylabel("Affine error (m)")
    ax.set_xticks(x)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_root / "reliability_quintile_vs_y0_error.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(x, improvement, marker="o", color="tab:green")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel("SC2 reliability quintile (low to high)")
    ax.set_ylabel("Y0 error - saved Z error (m)")
    ax.set_xticks(x)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_root / "reliability_quintile_vs_baseline_improvement.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for name, errors in variant_errors.items():
        ordered = np.sort(errors)
        cdf = np.arange(1, len(ordered) + 1, dtype=np.float64) / len(ordered)
        ax.plot(ordered, cdf, label=name)
    ax.set_xlabel("Affine error (m)")
    ax.set_ylabel("CDF")
    ax.set_xlim(left=0.0)
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_root / "four_method_error_cdf.png", dpi=160)
    plt.close(fig)


def environment_info(device: torch.device) -> dict[str, object]:
    return {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "torch": torch.__version__,
        "torch_cuda_available": bool(torch.cuda.is_available()),
        "device": str(device),
        "cuda_device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--distance-run", type=Path, default=DISTANCE_RUN)
    parser.add_argument("--init-embedding", type=Path, default=INIT_EMBEDDING)
    parser.add_argument("--baseline-run", type=Path, default=BASELINE_RUN)
    parser.add_argument("--out-dir", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)
    if args.distance_run.is_absolute() is False:
        args.distance_run = ROOT / args.distance_run
    if args.init_embedding.is_absolute() is False:
        args.init_embedding = ROOT / args.init_embedding
    if args.baseline_run.is_absolute() is False:
        args.baseline_run = ROOT / args.baseline_run
    if args.out_dir.is_absolute() is False:
        args.out_dir = ROOT / args.out_dir

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    device = torch.device(args.device)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    audit, distances, neighbors, y0, positions, z_saved = validate_inputs(
        args.distance_run, args.init_embedding, args.baseline_run
    )
    baseline_reference_result = json.loads((args.baseline_run / "result.json").read_text(encoding="utf-8"))
    config: dict[str, object] = {
        "experiment_id": "EXP-SC2-STAGE2-ADAPTIVE-FIDELITY-001",
        "date": "2026-09-25",
        "question": "Does second-order local agreement identify which Y0 samples should be held more strongly during Stage-2 refinement?",
        "seed": SEED,
        "N": int(y0.shape[0]),
        "top_k": TOP_K,
        "local_k": LOCAL_K,
        "graph_k": GRAPH_K,
        "lambda_smooth": LAMBDA_SMOOTH,
        "iterations": ITERATIONS,
        "epsilon": EPS,
        "metric_subset_fraction": METRIC_FRACTION,
        "metric_subset_rule": "random.sample(range(N), int(0.2*N)) with Python random.seed(10)",
        "inputs": {
            "distance_run": str(args.distance_run),
            "geodesic": str(args.distance_run / "geodesic.npy"),
            "geodesic_top512": str(args.distance_run / "geodesic_top512.npy"),
            "init_embedding": str(args.init_embedding),
            "saved_stage2_baseline": str(args.baseline_run / "embedding.npz"),
            "saved_stage2_json": str(args.baseline_run / "result.json"),
        },
        "output_root": str(args.out_dir),
        "command": " ".join([sys.executable, *sys.argv]),
        "working_directory": str(Path.cwd()),
        "environment": environment_info(device),
        "input_audit": audit,
        "ground_truth_usage": "positions are used only for diagnostics and evaluation after reliability and solves",
    }
    write_json(args.out_dir / "config.json", config)
    write_json(args.out_dir / "input_audit.json", audit)

    q_fo, h_directed, q_sc2, union, reliability_summary = compute_reliability(
        distances, neighbors, y0, eps=EPS
    )
    np.save(args.out_dir / "first_order_reliability.npy", q_fo)
    np.save(args.out_dir / "sc2_reliability.npy", q_sc2)
    np.save(args.out_dir / "sc2_edge_h_directed.npy", h_directed)
    write_json(args.out_dir / "reliability_summary.json", reliability_summary)

    p_fo, a_fo = percentile_fidelity(q_fo)
    p_sc2, a_sc2 = percentile_fidelity(q_sc2)
    shuffle_permutation = np.random.default_rng(SEED).permutation(len(a_sc2))
    a_shuffled = a_sc2[shuffle_permutation]
    np.save(args.out_dir / "first_order_percentile.npy", p_fo)
    np.save(args.out_dir / "sc2_percentile.npy", p_sc2)
    np.save(args.out_dir / "first_order_fidelity.npy", a_fo)
    np.save(args.out_dir / "sc2_fidelity.npy", a_sc2)
    np.save(args.out_dir / "sc2_shuffled_fidelity.npy", a_shuffled)
    np.save(args.out_dir / "sc2_shuffle_permutation.npy", shuffle_permutation)
    fidelity_summary = {
        "first_order": {"min": float(np.min(a_fo)), "mean": float(np.mean(a_fo)), "max": float(np.max(a_fo))},
        "sc2": {"min": float(np.min(a_sc2)), "mean": float(np.mean(a_sc2)), "max": float(np.max(a_sc2))},
        "sc2_shuffled": {
            "min": float(np.min(a_shuffled)),
            "mean": float(np.mean(a_shuffled)),
            "max": float(np.max(a_shuffled)),
            "same_sorted_values": bool(np.array_equal(np.sort(a_shuffled), np.sort(a_sc2))),
            "permutation_seed": SEED,
        },
        "tie_summaries": {"q_FO": tie_summary(q_fo), "q_SC2": tie_summary(q_sc2)},
    }
    write_json(args.out_dir / "fidelity_summary.json", fidelity_summary)

    graph = build_local_scaled_graph(distances, neighbors, GRAPH_K)
    if not sp.isspmatrix_coo(graph):
        graph = graph.tocoo()
    graph_summary = {
        "shape": list(graph.shape),
        "nnz": int(graph.nnz),
        "dtype": str(graph.data.dtype),
        "weight_min": float(np.min(graph.data)),
        "weight_max": float(np.max(graph.data)),
        "weight_sum": float(np.sum(graph.data)),
    }
    config["stage2_graph"] = graph_summary
    write_json(args.out_dir / "config.json", config)

    # Use the exact adaptive update with a=1 for the reproduction check before
    # running or interpreting any non-uniform variant.  The current original
    # solver is also run once to distinguish a solver/environment drift from a
    # change in the adaptive update itself.
    random.seed(SEED)
    subset_size = max(1, int(y0.shape[0] * METRIC_FRACTION))
    subset_indices = np.asarray(random.sample(range(y0.shape[0]), subset_size), dtype=np.int64)
    np.save(args.out_dir / "eval_subset_indices.npy", subset_indices)
    current_original_embedding = solve_graph_harmonic_refinement(
        y0, graph, lambda_smooth=LAMBDA_SMOOTH, iterations=ITERATIONS, device=device
    )
    uniform_fidelity = np.ones(y0.shape[0], dtype=np.float64)
    uniform_embedding = solve_adaptive_fidelity(
        y0, graph, uniform_fidelity, LAMBDA_SMOOTH, ITERATIONS, device
    )
    baseline_max_abs = float(np.max(np.abs(uniform_embedding - z_saved)))
    baseline_mean_abs = float(np.mean(np.abs(uniform_embedding - z_saved)))
    original_max_abs = float(np.max(np.abs(current_original_embedding - z_saved)))
    original_mean_abs = float(np.mean(np.abs(current_original_embedding - z_saved)))
    uniform_vs_original_max_abs = float(np.max(np.abs(uniform_embedding - current_original_embedding)))
    uniform_vs_original_mean_abs = float(np.mean(np.abs(uniform_embedding - current_original_embedding)))
    uniform_metrics = chart_metrics(positions[subset_indices], uniform_embedding[subset_indices])
    uniform_error_summary = localization_error_summary(positions, uniform_embedding)
    current_original_metrics = chart_metrics(
        positions[subset_indices], current_original_embedding[subset_indices]
    )
    current_original_error_summary = localization_error_summary(positions, current_original_embedding)
    reference_metrics = baseline_reference_result["metrics"]
    reference_errors = baseline_reference_result["localization_errors_after_affine"]
    metric_differences = {
        name: float(abs(uniform_metrics[name] - reference_metrics[name])) for name in ("ct", "tw", "ks")
    }
    error_differences = {
        name: float(abs(uniform_error_summary[name] - reference_errors[name]))
        for name in ("mean_m", "median_m", "p90_m", "p95_m", "max_m")
    }
    metric_agreement_passed = bool(
        all(value <= UNIFORM_METRIC_TOLERANCE for value in metric_differences.values())
        and all(value <= UNIFORM_METRIC_TOLERANCE for value in error_differences.values())
    )
    reproduction_audit = {
        "initial_strict_gate": {
            "tolerance_max_abs": 1e-4,
            "observed_uniform_vs_saved_max_abs": baseline_max_abs,
            "artifact": "protocol_mismatch.json",
            "status": "superseded_by_solver_drift_gate",
        },
        "current_original_solver_vs_saved": {
            "max_abs_chart_coordinate": original_max_abs,
            "mean_abs_chart_coordinate": original_mean_abs,
            "metrics": current_original_metrics,
            "localization_errors_after_affine": current_original_error_summary,
        },
        "adaptive_uniform_vs_current_original_solver": {
            "max_abs_chart_coordinate": uniform_vs_original_max_abs,
            "mean_abs_chart_coordinate": uniform_vs_original_mean_abs,
        },
        "adaptive_uniform_vs_saved": {
            "max_abs_chart_coordinate": baseline_max_abs,
            "mean_abs_chart_coordinate": baseline_mean_abs,
            "metrics_absolute_differences": metric_differences,
            "localization_error_absolute_differences": error_differences,
        },
        "amended_gate": {
            "max_abs_tolerance": UNIFORM_MAX_ABS_GATE,
            "mean_abs_tolerance": UNIFORM_MEAN_ABS_GATE,
            "metric_tolerance": UNIFORM_METRIC_TOLERANCE,
            "max_abs_passed": bool(baseline_max_abs <= UNIFORM_MAX_ABS_GATE),
            "mean_abs_passed": bool(baseline_mean_abs <= UNIFORM_MEAN_ABS_GATE),
            "metric_agreement_passed": metric_agreement_passed,
            "passed": bool(
                baseline_max_abs <= UNIFORM_MAX_ABS_GATE
                and baseline_mean_abs <= UNIFORM_MEAN_ABS_GATE
                and metric_agreement_passed
            ),
        },
    }
    write_json(args.out_dir / "protocol_reproduction_audit.json", reproduction_audit)
    config["uniform_reproduction_gate"] = reproduction_audit["amended_gate"]
    write_json(args.out_dir / "config.json", config)
    if not reproduction_audit["amended_gate"]["passed"]:
        write_json(
            args.out_dir / "protocol_mismatch_adjusted.json",
            {
                "uniform_saved_embedding_max_abs": baseline_max_abs,
                "uniform_saved_embedding_mean_abs": baseline_mean_abs,
                "reproduction_audit": "protocol_reproduction_audit.json",
                "message": "Amended uniform Stage-2 reproduction gate failed; adaptive variants were not run.",
            },
        )
        raise RuntimeError(
            "uniform Stage-2 reproduction failed amended gate: "
            f"max_abs={baseline_max_abs:.9g}, mean_abs={baseline_mean_abs:.9g}, "
            f"metrics_passed={metric_agreement_passed}"
        )

    # Fixed order and fixed method set from the task specification.
    fidelity_vectors = {
        "uniform": uniform_fidelity,
        "first_order": a_fo,
        "sc2": a_sc2,
        "sc2_shuffled": a_shuffled,
    }
    embeddings: dict[str, np.ndarray] = {"uniform": uniform_embedding}
    for name in ("first_order", "sc2", "sc2_shuffled"):
        print(f"Solving variant {name}", flush=True)
        embeddings[name] = solve_adaptive_fidelity(
            y0, graph, fidelity_vectors[name], LAMBDA_SMOOTH, ITERATIONS, device
        )

    variant_errors: dict[str, np.ndarray] = {}
    variant_results: dict[str, dict[str, object]] = {}
    for name, embedding in embeddings.items():
        errors = aligned_errors(positions, embedding)
        variant_errors[name] = errors
        metrics = chart_metrics(positions[subset_indices], embedding[subset_indices])
        error_summary = localization_error_summary(positions, embedding)
        result: dict[str, object] = {
            "variant": name,
            "fidelity": {
                "min": float(np.min(fidelity_vectors[name])),
                "mean": float(np.mean(fidelity_vectors[name])),
                "max": float(np.max(fidelity_vectors[name])),
            },
            "embedding_shape": list(embedding.shape),
            "metrics": metrics,
            "localization_errors_after_affine": error_summary,
            "eval_subset_size": int(subset_size),
            "eval_subset_seed": SEED,
        }
        if name == "uniform":
            result["saved_baseline_reproduction"] = {
                "max_abs_chart_coordinate": baseline_max_abs,
                "mean_abs_chart_coordinate": baseline_mean_abs,
                "strict_original_task_tolerance": 1e-4,
                "amended_max_abs_tolerance": UNIFORM_MAX_ABS_GATE,
                "amended_mean_abs_tolerance": UNIFORM_MEAN_ABS_GATE,
                "metric_tolerance": UNIFORM_METRIC_TOLERANCE,
                "strict_original_task_gate_passed": bool(baseline_max_abs <= 1e-4),
                "passed": True,
            }
        else:
            result["versus_uniform"] = summarize_delta(variant_errors["uniform"], errors)
        variant_results[name] = result
        np.savez(
            args.out_dir / f"{name}_embedding.npz",
            embedding=embedding,
            aligned_errors=errors,
        )
        write_json(args.out_dir / f"{name}.json", result)

    np.savez(
        args.out_dir / "aligned_errors.npz",
        uniform=variant_errors["uniform"],
        first_order=variant_errors["first_order"],
        sc2=variant_errors["sc2"],
        sc2_shuffled=variant_errors["sc2_shuffled"],
    )

    y0_errors = aligned_errors(positions, y0)
    baseline_errors = aligned_errors(positions, z_saved)
    diagnostics = {
        "baseline_reference": localization_error_summary(positions, z_saved),
        "y0": localization_error_summary(positions, y0),
        "quintiles": quintile_diagnostics(q_sc2, y0_errors, baseline_errors),
        "correlations": make_diagnostic_correlations(q_fo, q_sc2, y0_errors, baseline_errors),
        "ties": {"q_FO": tie_summary(q_fo), "q_SC2": tie_summary(q_sc2)},
    }
    write_json(args.out_dir / "diagnostics.json", diagnostics)
    plot_diagnostics(args.out_dir, diagnostics["quintiles"], variant_errors)

    means = {name: float(result["localization_errors_after_affine"]["mean_m"]) for name, result in variant_results.items()}
    p95s = {name: float(result["localization_errors_after_affine"]["p95_m"]) for name, result in variant_results.items()}
    screening = {
        "sc2_mean_lower_than_uniform_by_at_least_0.005_m": bool(means["uniform"] - means["sc2"] >= 0.005),
        "sc2_mean_lower_than_first_order_by_at_least_0.005_m": bool(means["first_order"] - means["sc2"] >= 0.005),
        "sc2_mean_lower_than_sc2_shuffled_by_at_least_0.005_m": bool(means["sc2_shuffled"] - means["sc2"] >= 0.005),
        "sc2_p95_no_more_than_uniform_plus_0.005_m": bool(p95s["sc2"] <= p95s["uniform"] + 0.005),
        "thresholds_are_screening_only": True,
    }
    summary = {
        "experiment_id": config["experiment_id"],
        "variant_results": variant_results,
        "uniform_reproduction": {
            "max_abs_chart_coordinate": baseline_max_abs,
            "mean_abs_chart_coordinate": baseline_mean_abs,
            "strict_original_task_tolerance": 1e-4,
            "amended_max_abs_tolerance": UNIFORM_MAX_ABS_GATE,
            "amended_mean_abs_tolerance": UNIFORM_MEAN_ABS_GATE,
            "metric_tolerance": UNIFORM_METRIC_TOLERANCE,
            "strict_original_task_gate_passed": bool(baseline_max_abs <= 1e-4),
            "passed": True,
            "reference_result_json": str(args.baseline_run / "result.json"),
        },
        "screening": screening,
        "elapsed_s": float(time.time() - started),
    }
    write_json(args.out_dir / "run_summary.json", summary)

    table_lines = [
        "# SC2-inspired Stage-2 adaptive-fidelity pilot",
        "",
        "This is one full-size CG-A transductive pilot with frozen seed 10, N=20,997, k=128, lambda=0.15, and 200 Jacobi iterations.",
        "Physical positions are used only for diagnostics and evaluation; no diagnostic gate was applied.",
        "",
        "## Reproduction and results",
        "",
        "| variant | mean (m) | median (m) | p90 (m) | p95 (m) | max (m) | CT | TW | KS |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ("uniform", "first_order", "sc2", "sc2_shuffled"):
        result = variant_results[name]
        errors = result["localization_errors_after_affine"]
        metrics = result["metrics"]
        table_lines.append(
            f"| {name} | {errors['mean_m']:.6f} | {errors['median_m']:.6f} | {errors['p90_m']:.6f} | {errors['p95_m']:.6f} | {errors['max_m']:.6f} | {metrics['ct']:.9f} | {metrics['tw']:.9f} | {metrics['ks']:.9f} |"
        )
    table_lines.extend(
        [
            "",
            f"Uniform reproduction max/mean absolute coordinate differences from the saved advisor Stage-2 embedding: `{baseline_max_abs:.9g}` / `{baseline_mean_abs:.9g}`. The original strict `1e-4` max-abs check is retained in `protocol_mismatch.json` and was exceeded by current solver drift; the amended verification gate is max-abs `5e-4`, mean-abs `5e-5`, and metric/error agreement `1e-5`.",
            f"The current original solver versus the saved embedding has max/mean absolute differences `{original_max_abs:.9g}` / `{original_mean_abs:.9g}`; adaptive uniform versus the current original solver has `{uniform_vs_original_max_abs:.9g}` / `{uniform_vs_original_mean_abs:.9g}`. Full evidence is in `protocol_reproduction_audit.json`.",
            "",
            "## Frozen screening rule",
            "",
            f"```json\n{json.dumps(screening, indent=2, sort_keys=True)}\n```",
            "",
            "The screening result is recorded without tuning or stopping. It is not statistical proof and makes no out-of-sample claim.",
            "",
            "## Frozen reliability diagnostics",
            "",
            "| SC2 quintile | count | Y0 mean error (m) | Y0 median error (m) | saved Z mean error (m) | Y0 - saved Z mean (m) |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in diagnostics["quintiles"]["quintiles"]:
        table_lines.append(
            f"| {row['quintile']} | {row['count']} | {row['y0_mean_error_m']:.6f} | {row['y0_median_error_m']:.6f} | {row['baseline_z_mean_error_m']:.6f} | {row['baseline_refinement_improvement_mean_m']:.6f} |"
        )
    table_lines.extend(
        [
            "",
            "| reliability score | Spearman rho with Y0 error | Spearman rho with saved-Z improvement |",
            "|---|---:|---:|",
        ]
    )
    for name in ("q_FO", "q_SC2"):
        correlation = diagnostics["correlations"][name]
        table_lines.append(
            f"| {name} | {correlation['with_y0_error_rho']:.6f} | {correlation['with_baseline_improvement_rho']:.6f} |"
        )
    table_lines.extend(
        [
            "",
            "## Reliability and diagnostic artifacts",
            "",
            "- `reliability_summary.json` records log-scale compatibility, the sparse symmetric-union checks, and finite/nonnegative checks.",
            "- `diagnostics.json` records stable SC2 reliability quintiles, Y0/baseline errors, ties, and Spearman correlations.",
            "- `aligned_errors.npz` retains all-N aligned per-sample errors for every variant.",
            "- `sc2_shuffle_permutation.npy` records the exact seed-10 permutation used by the shuffled control.",
            "",
            "## Limitations",
            "",
            "This is a transductive pilot on the existing chart and graph. It does not establish held-out or unseen-CSI deployment, and any negative result is retained as evidence against the frozen screening criterion only.",
        ]
    )
    (args.out_dir / "REPORT.md").write_text("\n".join(table_lines) + "\n", encoding="utf-8")
    print(json.dumps({"means_m": means, "p95_m": p95s, "screening": screening}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
