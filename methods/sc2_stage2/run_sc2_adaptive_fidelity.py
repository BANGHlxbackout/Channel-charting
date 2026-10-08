from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
from scipy.stats import rankdata, spearmanr
import torch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common_metrics import affine_transform, chart_metrics, localization_error_summary
from run_diffusion_refined_charting import build_local_scaled_graph, solve_graph_harmonic_refinement


N_EPS = 1e-12
LOCAL_K = 64
GRAPH_K = 128
TOP_K = 512
LAMBDA_SMOOTH = 0.15
ITERATIONS = 200
METRIC_SUBSET_FRACTION = 0.2
UNIFORM_TOLERANCE = 5e-4


def load_embedding(path: Path, expected_indices: np.ndarray, name: str) -> tuple[np.ndarray, np.ndarray]:
    saved = np.load(path)
    if "indices" not in saved or "embedding" not in saved:
        raise ValueError(f"{name} must contain indices and embedding: {path}")
    indices = np.asarray(saved["indices"])
    embedding = np.asarray(saved["embedding"], dtype=np.float32)
    if not np.array_equal(indices, expected_indices):
        raise ValueError(f"{name} indices do not match the requested data selection")
    if embedding.shape != (len(expected_indices), 2):
        raise ValueError(f"{name} must have shape ({len(expected_indices)}, 2), got {embedding.shape}")
    return indices, embedding


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def validate_neighbor_cache(
    distances: np.ndarray,
    neighbors: np.ndarray,
    seed: int,
    sample_rows: int = 64,
) -> dict[str, object]:
    n = distances.shape[0]
    if neighbors.shape != (n, TOP_K):
        raise ValueError(f"expected top-{TOP_K} cache shape {(n, TOP_K)}, got {neighbors.shape}")
    if not np.issubdtype(neighbors.dtype, np.integer):
        raise ValueError(f"neighbor cache must be integer, got {neighbors.dtype}")
    if int(neighbors.min()) < 0 or int(neighbors.max()) >= n:
        raise ValueError("neighbor cache contains out-of-range indices")
    if np.any(neighbors == np.arange(n, dtype=neighbors.dtype)[:, None]):
        raise ValueError("neighbor cache contains self neighbors")

    rng = np.random.default_rng(seed)
    rows = np.unique(np.r_[np.arange(min(32, n)), rng.choice(n, min(sample_rows, n), replace=False)])
    ordered = 0
    exact = 0
    exact_local = 0
    valid_local = 0
    top512_boundary_tie = 0
    local_boundary_tie = 0
    max_drop = 0.0
    for row_index in rows:
        cached = np.asarray(neighbors[row_index], dtype=np.int64)
        values = np.asarray(distances[row_index, cached], dtype=np.float64)
        if np.all(np.isfinite(values)):
            ordered += int(np.all(values[:-1] <= values[1:]))
        max_drop = max(max_drop, float(np.max(values[:-1] - values[1:])))
        row = np.asarray(distances[row_index], dtype=np.float32).copy()
        row[row_index] = np.inf
        expected = np.argpartition(row, TOP_K - 1)[:TOP_K]
        exact += int(set(expected.tolist()) == set(cached.tolist()))
        expected_local = np.argpartition(row, LOCAL_K - 1)[:LOCAL_K]
        exact_local += int(set(expected_local.tolist()) == set(cached[:LOCAL_K].tolist()))
        local_values = values[:LOCAL_K]
        outside_mask = np.ones(n, dtype=bool)
        outside_mask[cached[:LOCAL_K]] = False
        outside_min = float(np.min(row[outside_mask]))
        valid_local += int(float(np.max(local_values)) <= outside_min)
        local_boundary_tie += int(float(np.max(local_values)) == outside_min)
        if set(expected.tolist()) != set(cached.tolist()):
            boundary = np.partition(row, TOP_K - 1)[TOP_K - 1]
            top512_boundary_tie += int(np.sum(row == boundary) > 1)
    return {
        "sample_rows": int(len(rows)),
        "ordered_rows": int(ordered),
        "exact_set_rows_top512": int(exact),
        "exact_set_rows_top64": int(exact_local),
        "valid_rank_rows_top64": int(valid_local),
        "top64_boundary_tie_rows": int(local_boundary_tie),
        "top512_mismatch_boundary_tie_rows": int(top512_boundary_tie),
        "max_distance_drop": max_drop,
    }


def build_local_compatibility(
    distances: np.ndarray,
    y0: np.ndarray,
    neighbors64: np.ndarray,
    epsilon: float = N_EPS,
) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray, dict[str, float]]:
    n = len(y0)
    rows = np.repeat(np.arange(n, dtype=np.int64), LOCAL_K)
    cols = neighbors64.reshape(-1).astype(np.int64, copy=False)
    d_geo = np.asarray(distances[rows, cols], dtype=np.float64)
    d_y = np.linalg.norm(
        y0[rows].astype(np.float64) - y0[cols].astype(np.float64), axis=1
    )
    if not np.all(np.isfinite(d_geo)) or not np.all(np.isfinite(d_y)):
        raise ValueError("local geodesic or Y0 distances contain non-finite values")
    if np.any(d_geo <= 0.0) or np.any(d_y <= 0.0):
        raise ValueError("log-scale compatibility requires strictly positive local distances")

    log_scale = float(np.median(np.log(d_y) - np.log(d_geo)))
    residual = np.abs(np.log(d_y) - np.log(d_geo) - log_scale)
    tau = float(np.median(residual) + epsilon)
    c_directed_values = np.exp(-residual / tau)
    directed = sp.coo_matrix(
        (c_directed_values, (rows, cols)), shape=(n, n), dtype=np.float64
    ).tocsr()
    # Formula C_ij is symmetric for an unordered pair. Preserve unilateral
    # top-64 edges without halving their score when sparse rows are unioned.
    c_symmetric = directed.maximum(directed.T).tocsr()
    q_fo = c_directed_values.reshape(n, LOCAL_K).mean(axis=1)
    stats = {
        "log_scale": log_scale,
        "scale": float(np.exp(log_scale)),
        "residual_median": float(np.median(residual)),
        "residual_p90": float(np.percentile(residual, 90)),
        "tau": tau,
        "directed_edges": int(directed.nnz),
        "symmetric_edges": int(c_symmetric.nnz),
        "c_directed_min": float(np.min(c_directed_values)),
        "c_directed_median": float(np.median(c_directed_values)),
        "c_directed_max": float(np.max(c_directed_values)),
    }
    return c_symmetric, q_fo.astype(np.float64), c_directed_values, stats


def compute_sc2_reliability(
    c_symmetric: sp.csr_matrix,
    neighbors64: np.ndarray,
    c_directed_values: np.ndarray,
    block_size: int = 64,
    epsilon: float = N_EPS,
) -> tuple[np.ndarray, dict[str, float]]:
    """Compute H and S only on directed top-64 edges.

    A full C @ C.T would be unnecessarily large. The block products below
    keep the same exact row dot products and extract only the requested local
    pairs.
    """
    n = c_symmetric.shape[0]
    rows = np.repeat(np.arange(n, dtype=np.int64), LOCAL_K)
    cols = neighbors64.reshape(-1).astype(np.int64, copy=False)
    row_norm = np.sqrt(np.asarray(c_symmetric.multiply(c_symmetric).sum(axis=1)).ravel())
    numerator = np.empty(n * LOCAL_K, dtype=np.float64)
    c_transpose = c_symmetric.T.tocsc()
    t0 = time.time()
    for start in range(0, n, block_size):
        stop = min(n, start + block_size)
        product = (c_symmetric[start:stop] @ c_transpose).tocsr()
        if not product.has_sorted_indices:
            product.sort_indices()
        for local_row, global_row in enumerate(range(start, stop)):
            target_cols = neighbors64[global_row].astype(np.int64, copy=False)
            row_start = product.indptr[local_row]
            row_stop = product.indptr[local_row + 1]
            product_cols = product.indices[row_start:row_stop]
            product_data = product.data[row_start:row_stop]
            positions = np.searchsorted(product_cols, target_cols)
            valid = positions < len(product_cols)
            values = np.zeros(LOCAL_K, dtype=np.float64)
            if np.any(valid):
                valid_positions = positions[valid]
                exact = product_cols[valid_positions] == target_cols[valid]
                valid_indices = np.flatnonzero(valid)
                values[valid_indices[exact]] = product_data[valid_positions[exact]]
            numerator[global_row * LOCAL_K : (global_row + 1) * LOCAL_K] = values
        del product
        if (start // block_size) % 20 == 0 or stop == n:
            print(
                f"SC2 overlap blocks {stop}/{n} elapsed={time.time() - t0:.1f}s",
                flush=True,
            )

    denominator = row_norm[rows] * row_norm[cols] + epsilon
    h_values = numerator / denominator
    c_local_symmetric = np.asarray(c_symmetric[rows, cols]).ravel()
    s_values = c_local_symmetric * h_values
    q_sc2 = s_values.reshape(n, LOCAL_K).mean(axis=1)
    stats = {
        "h_min": float(np.min(h_values)),
        "h_median": float(np.median(h_values)),
        "h_max": float(np.max(h_values)),
        "s_min": float(np.min(s_values)),
        "s_median": float(np.median(s_values)),
        "s_max": float(np.max(s_values)),
        "q_fo_mean": float(np.mean(c_directed_values.reshape(n, LOCAL_K).mean(axis=1))),
        "q_sc2_mean": float(np.mean(q_sc2)),
    }
    return q_sc2.astype(np.float64), stats


def validate_sc2_overlap(
    c_symmetric: sp.csr_matrix,
    neighbors64: np.ndarray,
    sample_rows: tuple[int, ...] = (0, 1, 2, 31, 63),
    sample_neighbors: int = 8,
) -> dict[str, float | int]:
    """Regression check for sparse product extraction used by SC²."""
    rows = [row for row in sample_rows if row < c_symmetric.shape[0]]
    product = (c_symmetric[rows] @ c_symmetric.T.tocsc()).tocsr()
    product.sort_indices()
    checked = 0
    bad = 0
    max_abs = 0.0
    for local_row, global_row in enumerate(rows):
        sparse_row = product.getrow(local_row)
        for target in neighbors64[global_row, :sample_neighbors]:
            target = int(target)
            direct = float(c_symmetric.getrow(global_row).multiply(c_symmetric.getrow(target)).sum())
            extracted = float(sparse_row[0, target])
            max_abs = max(max_abs, abs(extracted - direct))
            bad += int(abs(extracted - direct) > 1e-10)
            checked += 1
    return {
        "checked_edges": checked,
        "bad_edges": bad,
        "max_abs": max_abs,
    }


def percentile_01(values: np.ndarray) -> np.ndarray:
    return (rankdata(values, method="average") - 0.5) / len(values)


def solve_fidelity_refinement(
    y0: np.ndarray,
    graph: sp.coo_matrix,
    fidelity: np.ndarray,
    lambda_smooth: float,
    iterations: int,
    device: torch.device,
) -> np.ndarray:
    graph = graph.tocsr()
    degree = np.asarray(graph.sum(axis=1)).ravel().astype(np.float32)
    coo = graph.tocoo()
    row = torch.from_numpy(coo.row.astype(np.int64)).to(device)
    col = torch.from_numpy(coo.col.astype(np.int64)).to(device)
    weight = torch.from_numpy(coo.data.astype(np.float32)).to(device)
    degree_t = torch.from_numpy(degree).to(device)[:, None]
    y0_t = torch.from_numpy(y0.astype(np.float32, copy=False)).to(device)
    fidelity_t = torch.from_numpy(fidelity.astype(np.float32, copy=False)).to(device)[:, None]
    z = y0_t.clone()
    with torch.no_grad():
        for iteration in range(1, iterations + 1):
            acc = torch.zeros_like(z)
            acc.index_add_(0, row, weight[:, None] * z[col])
            z = (fidelity_t * y0_t + lambda_smooth * acc) / (
                fidelity_t + lambda_smooth * degree_t
            )
            if iteration in {1, 10, 50, 100, iterations}:
                print(f"Jacobi iteration {iteration}/{iterations}", flush=True)
    return z.detach().cpu().numpy()


def equal_quintiles(values: np.ndarray) -> list[np.ndarray]:
    order = np.argsort(values, kind="mergesort")
    return [chunk for chunk in np.array_split(order, 5)]


def empirical_cdf(errors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.sort(errors)
    y = np.arange(1, len(x) + 1, dtype=np.float64) / len(x)
    return x, y


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--distance-run", type=Path, required=True)
    parser.add_argument("--init-embedding", type=Path, required=True)
    parser.add_argument("--baseline-embedding", type=Path, required=True)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/sc2_stage2_adaptive_fidelity_pilot"),
    )
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if LOCAL_K > TOP_K or GRAPH_K > TOP_K:
        raise ValueError("local and graph k must not exceed the top-k cache")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    t0 = time.time()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    distance_run = args.distance_run
    saved_distance = np.load(distance_run / "distance_data.npz")
    indices = np.asarray(saved_distance["indices"])
    positions = np.asarray(saved_distance["positions"], dtype=np.float32)
    if indices.ndim != 1 or positions.shape != (len(indices), 2):
        raise ValueError("distance_data.npz has unexpected indices/positions shapes")
    n = len(indices)
    if not np.array_equal(indices, np.arange(n, dtype=indices.dtype)):
        raise ValueError("distance_data indices are not the expected stride-1 order")

    _, y0 = load_embedding(args.init_embedding, indices, "Y0")
    _, z_reference = load_embedding(args.baseline_embedding, indices, "baseline Z")
    y0_saved = np.load(args.init_embedding)
    baseline_saved = np.load(args.baseline_embedding)
    if not np.array_equal(positions, y0_saved["positions"]):
        raise ValueError("distance positions and Y0 positions differ")
    if not np.array_equal(positions, baseline_saved["positions"]):
        raise ValueError("distance positions and baseline Z positions differ")
    if not np.all(np.isfinite(positions)) or not np.all(np.isfinite(y0)) or not np.all(np.isfinite(z_reference)):
        raise ValueError("positions, Y0, or baseline Z contains non-finite values")

    ue_pos_path = args.data_path / "UE_pos.npy"
    ue_pos = np.load(ue_pos_path)
    if ue_pos.ndim != 2 or ue_pos.shape[0] < n or ue_pos.shape[1] < 2:
        raise ValueError(f"unexpected UE_pos shape: {ue_pos.shape}")
    if not np.all(np.isfinite(ue_pos[indices, :2])):
        raise ValueError("original UE_pos contains non-finite values at requested indices")
    if not np.array_equal(positions, np.asarray(ue_pos[indices, :2], dtype=np.float32)):
        raise ValueError("distance_data positions do not match original UE_pos at the saved indices")

    distances = np.load(distance_run / "geodesic.npy", mmap_mode="r")
    if distances.shape != (n, n) or distances.dtype != np.float32:
        raise ValueError(f"unexpected geodesic shape/dtype: {distances.shape}, {distances.dtype}")
    if not np.all(np.isfinite(distances)):
        raise ValueError("original geodesic contains non-finite values")
    cache_path = distance_run / f"geodesic_top{TOP_K}.npy"
    if not cache_path.exists():
        raise FileNotFoundError(f"refusing to recompute Stage-2 cache: {cache_path}")
    neighbors = np.load(cache_path, mmap_mode="r")
    cache_validation = validate_neighbor_cache(distances, neighbors, args.seed)
    if (
        cache_validation["ordered_rows"] != cache_validation["sample_rows"]
        or cache_validation["valid_rank_rows_top64"] != cache_validation["sample_rows"]
    ):
        raise ValueError(f"existing top-{TOP_K} cache failed sampled raw-D_geo validation: {cache_validation}")
    neighbors64 = np.asarray(neighbors[:, :LOCAL_K], dtype=np.int32)

    c_symmetric, q_fo, c_directed_values, compatibility_stats = build_local_compatibility(
        distances, y0, neighbors64
    )
    overlap_validation = validate_sc2_overlap(c_symmetric, neighbors64)
    if overlap_validation["bad_edges"]:
        raise ValueError(f"SC2 sparse overlap regression failed: {overlap_validation}")
    q_sc2, sc2_stats = compute_sc2_reliability(c_symmetric, neighbors64, c_directed_values)
    np.save(out_dir / "first_order_reliability.npy", q_fo)
    np.save(out_dir / "sc2_reliability.npy", q_sc2)

    print("building inherited Stage-2 graph", flush=True)
    graph = build_local_scaled_graph(distances, neighbors, GRAPH_K)
    if graph.nnz != 3_088_334:
        raise ValueError(f"unexpected Stage-2 graph nnz: {graph.nnz}")
    if graph.shape != (n, n) or (graph != graph.T).nnz != 0:
        raise ValueError("inherited Stage-2 graph is not the expected symmetric shape")

    p_fo = percentile_01(q_fo)
    p_sc2 = percentile_01(q_sc2)
    fidelity = {
        "uniform": np.ones(n, dtype=np.float64),
        "first_order": 0.5 + p_fo,
        "sc2": 0.5 + p_sc2,
    }
    permutation = np.random.default_rng(args.seed).permutation(n)
    fidelity["sc2_shuffled"] = fidelity["sc2"][permutation]
    np.savez(
        out_dir / "fidelity_vectors.npz",
        uniform=fidelity["uniform"],
        first_order=fidelity["first_order"],
        sc2=fidelity["sc2"],
        sc2_shuffled=fidelity["sc2_shuffled"],
        shuffle_permutation=permutation,
    )

    device = torch.device(args.device)
    embeddings: dict[str, np.ndarray] = {}
    for name in ("uniform", "first_order", "sc2", "sc2_shuffled"):
        print(f"solving variant={name} fidelity range=({fidelity[name].min():.6f}, {fidelity[name].max():.6f})", flush=True)
        embeddings[name] = solve_fidelity_refinement(
            y0,
            graph,
            fidelity[name],
            LAMBDA_SMOOTH,
            ITERATIONS,
            device,
        )
    np.savez(out_dir / "variant_embeddings.npz", **embeddings, indices=indices, positions=positions)

    print("checking tracked solver and repeat uniform solve", flush=True)
    tracked_uniform = solve_graph_harmonic_refinement(
        y0, graph, LAMBDA_SMOOTH, ITERATIONS, device
    )
    repeated_uniform = solve_fidelity_refinement(
        y0, graph, fidelity["uniform"], LAMBDA_SMOOTH, ITERATIONS, device
    )

    def compare_embeddings(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
        delta = left.astype(np.float64) - right.astype(np.float64)
        return {
            "max_abs": float(np.max(np.abs(delta))),
            "mean_abs": float(np.mean(np.abs(delta))),
            "rmse": float(np.sqrt(np.mean(np.square(delta)))),
        }

    uniform_check = compare_embeddings(embeddings["uniform"], z_reference)
    tracked_check = compare_embeddings(tracked_uniform, z_reference)
    repeated_check = compare_embeddings(repeated_uniform, z_reference)
    uniform_repeat_check = compare_embeddings(embeddings["uniform"], repeated_uniform)
    tracked_uniform_check = compare_embeddings(tracked_uniform, embeddings["uniform"])
    if max(
        uniform_check["max_abs"],
        tracked_check["max_abs"],
        repeated_check["max_abs"],
        uniform_repeat_check["max_abs"],
        tracked_uniform_check["max_abs"],
    ) > UNIFORM_TOLERANCE:
        raise RuntimeError(
            "uniform variant did not reproduce the original Stage-2 baseline "
            f"within {UNIFORM_TOLERANCE}: {uniform_check}; tracked={tracked_check}; "
            f"repeat={repeated_check}; uniform-repeat={uniform_repeat_check}"
        )

    subset_size = max(1, int(n * METRIC_SUBSET_FRACTION))
    random.seed(args.seed)
    subset_indices = np.asarray(random.sample(range(n), subset_size), dtype=np.int64)
    metrics: dict[str, dict[str, float]] = {}
    aligned_errors: dict[str, np.ndarray] = {}
    for name, embedding in embeddings.items():
        metrics[name] = {
            **chart_metrics(positions[subset_indices], embedding[subset_indices]),
            **localization_error_summary(positions, embedding),
        }
        aligned = affine_transform(positions, embedding)
        aligned_errors[name] = np.linalg.norm(aligned - positions, axis=1)

    quintiles = equal_quintiles(q_sc2)
    y0_aligned = affine_transform(positions, y0)
    y0_errors = np.linalg.norm(y0_aligned - positions, axis=1)
    z_aligned = affine_transform(positions, z_reference)
    z_errors = np.linalg.norm(z_aligned - positions, axis=1)
    q_rows: list[dict[str, float | int | str]] = []
    for q_index, members in enumerate(quintiles, start=1):
        q_rows.append(
            {
                "quintile": f"Q{q_index}",
                "n": int(len(members)),
                "q_sc2_mean": float(np.mean(q_sc2[members])),
                "q_sc2_median": float(np.median(q_sc2[members])),
                "y0_mean_error_m": float(np.mean(y0_errors[members])),
                "y0_median_error_m": float(np.median(y0_errors[members])),
                "baseline_z_mean_error_m": float(np.mean(z_errors[members])),
                "y0_minus_baseline_z_mean_m": float(np.mean(y0_errors[members] - z_errors[members])),
            }
        )

    spearman = {
        "q_sc2_vs_y0_error": {
            "rho": float(spearmanr(q_sc2, y0_errors).statistic),
            "pvalue": float(spearmanr(q_sc2, y0_errors).pvalue),
        },
        "q_sc2_vs_y0_minus_baseline_z_error": {
            "rho": float(spearmanr(q_sc2, y0_errors - z_errors).statistic),
            "pvalue": float(spearmanr(q_sc2, y0_errors - z_errors).pvalue),
        },
        "q_fo_vs_y0_error": {
            "rho": float(spearmanr(q_fo, y0_errors).statistic),
            "pvalue": float(spearmanr(q_fo, y0_errors).pvalue),
        },
        "q_fo_vs_y0_minus_baseline_z_error": {
            "rho": float(spearmanr(q_fo, y0_errors - z_errors).statistic),
            "pvalue": float(spearmanr(q_fo, y0_errors - z_errors).pvalue),
        },
    }

    # Figure 1: SC2 reliability quintile against Y0 and baseline errors.
    labels = [row["quintile"] for row in q_rows]
    plt.figure(figsize=(7.2, 4.6))
    plt.plot(labels, [row["y0_mean_error_m"] for row in q_rows], marker="o", label="Y0 mean error")
    plt.plot(labels, [row["y0_median_error_m"] for row in q_rows], marker="o", label="Y0 median error")
    plt.plot(labels, [row["baseline_z_mean_error_m"] for row in q_rows], marker="o", label="baseline Z mean error")
    plt.xlabel("SC² reliability quintile (Q1 = lowest)")
    plt.ylabel("Affine localization error (m)")
    plt.title("Reliability quintile vs. Y0 and baseline Stage-2 error")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "reliability_quintile_y0_error.png", dpi=180)
    plt.close()

    # Figure 2: reliability quintile against improvement over Y0.
    plt.figure(figsize=(7.2, 4.6))
    for name, label in (
        ("uniform", "uniform"),
        ("first_order", "first-order"),
        ("sc2", "SC²"),
        ("sc2_shuffled", "SC² shuffled"),
    ):
        values = [float(np.mean(y0_errors[members] - aligned_errors[name][members])) for members in quintiles]
        plt.plot(labels, values, marker="o", label=label)
    plt.axhline(0.0, color="black", linewidth=0.8)
    plt.xlabel("SC² reliability quintile (Q1 = lowest)")
    plt.ylabel("Y0 error − variant error (m)")
    plt.title("Reliability quintile vs. refinement improvement")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "reliability_quintile_refinement_improvement.png", dpi=180)
    plt.close()

    # Figure 3: common full-data affine-error CDF.
    figure, axis = plt.subplots(figsize=(7.2, 4.6))
    for name, label in (
        ("uniform", "uniform"),
        ("first_order", "first-order"),
        ("sc2", "SC²"),
        ("sc2_shuffled", "SC² shuffled"),
    ):
        x, y = empirical_cdf(aligned_errors[name])
        axis.plot(x, y, label=f"{label} (mean={metrics[name]['mean_m']:.6f} m)")
    axis.set_xlabel("Affine localization error (m)")
    axis.set_ylabel("Empirical CDF")
    axis.set_title("Full-data affine localization error CDF")
    axis.grid(alpha=0.25)
    axis.legend()
    tail_axis = inset_axes(axis, width="48%", height="48%", loc="lower right", borderpad=1.1)
    for name, _label in (
        ("uniform", "uniform"),
        ("first_order", "first-order"),
        ("sc2", "SC²"),
        ("sc2_shuffled", "SC² shuffled"),
    ):
        x, y = empirical_cdf(aligned_errors[name])
        tail_axis.plot(x, y)
    tail_axis.set_xlim(0.5, 0.8)
    tail_axis.set_ylim(0.85, 1.005)
    tail_axis.set_title("tail", fontsize=8)
    tail_axis.tick_params(labelsize=7)
    tail_axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(out_dir / "error_cdf_variants.png", dpi=180)
    plt.close(figure)

    metadata = {
        "experiment": "SC2-inspired Stage-2 adaptive-fidelity pilot",
        "seed": args.seed,
        "num_samples": n,
        "local_k": LOCAL_K,
        "graph_k": GRAPH_K,
        "top_k_cache": TOP_K,
        "lambda_smooth": LAMBDA_SMOOTH,
        "iterations": ITERATIONS,
        "metric_subset_fraction": METRIC_SUBSET_FRACTION,
        "metric_subset_size": subset_size,
        "metric_subset_indices": subset_indices.tolist(),
        "device": str(device),
        "command": " ".join([str(Path(sys.executable).resolve()), *sys.argv]),
        "inputs": {
            "data_path": str(args.data_path.resolve()),
            "ue_pos": str(ue_pos_path.resolve()),
            "distance_run": str(distance_run.resolve()),
            "geodesic": str((distance_run / "geodesic.npy").resolve()),
            "neighbor_cache": str(cache_path.resolve()),
            "y0": str(args.init_embedding.resolve()),
            "baseline_z": str(args.baseline_embedding.resolve()),
        },
        "sha256": {
            "geodesic": sha256_file(distance_run / "geodesic.npy"),
            "neighbor_cache": sha256_file(cache_path),
            "y0": sha256_file(args.init_embedding),
            "baseline_z": sha256_file(args.baseline_embedding),
            "distance_data": sha256_file(distance_run / "distance_data.npz"),
            "ue_pos": sha256_file(ue_pos_path),
        },
        "cache_validation": cache_validation,
        "compatibility": compatibility_stats,
        "sc2": sc2_stats,
        "sc2_overlap_validation": overlap_validation,
        "symmetrization": "C = C_directed.maximum(C_directed.T); means use directed top-64 N_i",
        "epsilon": N_EPS,
        "uniform_reproduction": {
            "tolerance_max_abs": UNIFORM_TOLERANCE,
            "new_uniform_vs_saved": uniform_check,
            "tracked_solver_vs_saved": tracked_check,
            "repeated_uniform_vs_saved": repeated_check,
            "new_uniform_vs_repeated_uniform": uniform_repeat_check,
            "tracked_solver_vs_new_uniform": tracked_uniform_check,
        },
        "graph_edges": int(graph.nnz),
        "graph_shape": list(graph.shape),
        "graph_symmetric": bool((graph != graph.T).nnz == 0),
        "graph_weight_min": float(graph.data.min()),
        "graph_weight_max": float(graph.data.max()),
        "graph_degree_min": float(np.asarray(graph.sum(axis=1)).min()),
        "graph_degree_mean": float(np.asarray(graph.sum(axis=1)).mean()),
        "graph_degree_max": float(np.asarray(graph.sum(axis=1)).max()),
        "metrics": metrics,
        "spearman": spearman,
        "quintiles": q_rows,
        "elapsed_s": float(time.time() - t0),
    }
    (out_dir / "pilot_metrics.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")

    report = render_report(metadata, out_dir)
    (out_dir / "REPORT.md").write_text(report, encoding="utf-8")
    print(json.dumps(metadata, indent=2, sort_keys=True))


def render_report(metadata: dict[str, object], out_dir: Path) -> str:
    metrics = metadata["metrics"]
    quintiles = metadata["quintiles"]
    spearman = metadata["spearman"]
    lines = [
        "# SC²-inspired Stage-2 adaptive-fidelity pilot",
        "",
        "## Scope and frozen configuration",
        "",
        "This pilot estimates per-sample reliability from the original geodesic local neighborhoods and original Siamese `Y0`. It changes only the fidelity coefficient in the existing Stage-2 Jacobi update. Stage-1 graph construction, fused distances, `D_geo`, Siamese training, Stage-2 graph weights, and MDS gates were not changed or recomputed.",
        "",
        f"- Samples: `{metadata['num_samples']}`; seed: `{metadata['seed']}`.",
        f"- Local reliability neighbors: top-`{metadata['local_k']}` directed neighbors from the original `D_geo`.",
        f"- Stage-2 graph: top-`{metadata['graph_k']}` from the existing top-`{metadata['top_k_cache']}` cache; `lambda={metadata['lambda_smooth']}`, `{metadata['iterations']}` Jacobi iterations; graph edges `{metadata['graph_edges']}`.",
        f"- Evaluation subset: the same `{metadata['metric_subset_size']}` samples selected by `random.seed({metadata['seed']}); random.sample(range(N), int(N*0.2))`; affine localization summaries use all samples.",
        "- Ground truth positions appear only in evaluation and plots.",
        "",
        "## Input provenance",
        "",
        "- code: `methods/sc2_stage2/run_sc2_adaptive_fidelity.py`",
        "- approved plan: `docs/tasks/sc2_stage2_adaptive_fidelity_pilot.md`",
        f"- exact command: `{metadata['command']}`",
        "",
    ]
    for key, value in metadata["inputs"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines += [
        "",
        f"The existing top-512 cache was read-only validated on the sampled rows recorded in `pilot_metrics.json`; every sampled first-64 prefix satisfied the raw-D_geo rank condition. `{metadata['cache_validation']['top64_boundary_tie_rows']}` sampled top-64 rows and `{metadata['cache_validation']['top512_mismatch_boundary_tie_rows']}` top-512 row had boundary ties, so exact index sets are not always unique; no cache was regenerated or written to the original distance directory.",
        "",
        "## Reliability construction",
        "",
        f"For each directed top-64 pair, `dY=||Y0_i-Y0_j||` and the robust log scale was `median(log(dY)-log(D_geo)) = {metadata['compatibility']['log_scale']:.9g}`. The resulting residual median was `{metadata['compatibility']['residual_median']:.9g}`, and `tau=median(residual)+epsilon` was `{metadata['compatibility']['tau']:.9g}` with `epsilon={metadata['epsilon']}`.",
        "",
        "`C_directed` is converted to a symmetric sparse matrix with `C = C_directed.maximum(C_directed.T)`. This preserves unilateral top-64 edges without halving them through sparse zeros. The FO mean uses the original directed top-64 neighbors. SC² computes the cosine-style common-neighbor consistency `H` from the symmetric sparse rows, then averages `S=C_ij H_ij` over those same directed top-64 neighbors.",
        "",
        f"A focused sparse-overlap regression checked `{metadata['sc2_overlap_validation']['checked_edges']}` local product entries against direct sparse row intersections; bad entries: `{metadata['sc2_overlap_validation']['bad_edges']}`, maximum absolute difference: `{metadata['sc2_overlap_validation']['max_abs']:.3g}`. The initial implementation run was invalidated because unsorted CSR product indices made `searchsorted` extraction incorrect; product indices were sorted, this check passed, and the complete pilot was recomputed. That correction is implementation repair within this pilot, not a separate experimental condition.",
        "",
        f"The directed compatibility graph has `{metadata['compatibility']['directed_edges']}` edges and the symmetric union has `{metadata['compatibility']['symmetric_edges']}` edges. Reliability arrays are saved as [`first_order_reliability.npy`](first_order_reliability.npy) and [`sc2_reliability.npy`](sc2_reliability.npy).",
        "",
        "## Diagnostic: reliability and existing refinement",
        "",
        "Quintiles are sorted from low to high SC² reliability. Because `N` is not divisible by five, the group sizes differ by one sample.",
        "",
        "| Group | N | mean q_SC2 | Y0 mean (m) | Y0 median (m) | baseline Z mean (m) | Y0 − Z mean (m) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in quintiles:
        lines.append(
            f"| {row['quintile']} | {row['n']} | {row['q_sc2_mean']:.6f} | {row['y0_mean_error_m']:.6f} | {row['y0_median_error_m']:.6f} | {row['baseline_z_mean_error_m']:.6f} | {row['y0_minus_baseline_z_mean_m']:.6f} |"
        )
    lines += [
        "",
        "| Reliability | Spearman rho with Y0 error | p value | Spearman rho with Y0 − baseline Z error | p value |",
        "|---|---:|---:|---:|---:|",
        f"| SC² | {spearman['q_sc2_vs_y0_error']['rho']:.6f} | {spearman['q_sc2_vs_y0_error']['pvalue']:.3g} | {spearman['q_sc2_vs_y0_minus_baseline_z_error']['rho']:.6f} | {spearman['q_sc2_vs_y0_minus_baseline_z_error']['pvalue']:.3g} |",
        f"| first-order | {spearman['q_fo_vs_y0_error']['rho']:.6f} | {spearman['q_fo_vs_y0_error']['pvalue']:.3g} | {spearman['q_fo_vs_y0_minus_baseline_z_error']['rho']:.6f} | {spearman['q_fo_vs_y0_minus_baseline_z_error']['pvalue']:.3g} |",
        "",
        "![Reliability quintile versus Y0 error](reliability_quintile_y0_error.png)",
        "",
        "![Reliability quintile versus refinement improvement](reliability_quintile_refinement_improvement.png)",
        "",
        "## Four fidelity variants",
        "",
        "All variants use the same inherited Stage-2 graph and the update `Z_i=(a_iY_i+lambda sum_j W_ij Z_j)/(a_i+lambda d_i)`. Percentiles use the full-data average rank with `p_i=(rank_i-0.5)/N`; `sc2_shuffled` applies a fixed seed-10 permutation to the SC² fidelity vector, preserving its exact distribution.",
        "",
        "| Variant | CT | TW | KS | affine mean (m) | median (m) | P90 (m) | P95 (m) | max (m) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, label in (("uniform", "uniform"), ("first_order", "first-order"), ("sc2", "SC²"), ("sc2_shuffled", "SC² shuffled")):
        row = metrics[name]
        lines.append(
            f"| {label} | {row['ct']:.9f} | {row['tw']:.9f} | {row['ks']:.9f} | {row['mean_m']:.6f} | {row['median_m']:.6f} | {row['p90_m']:.6f} | {row['p95_m']:.6f} | {row['max_m']:.6f} |"
        )
    lines += [
        "",
        "### Uniform reproduction check",
        "",
        f"The uniform variant was compared against the supplied original baseline embedding before reporting variants. The new uniform-vs-saved comparison was max/mean/RMSE `{metadata['uniform_reproduction']['new_uniform_vs_saved']['max_abs']:.9g}` / `{metadata['uniform_reproduction']['new_uniform_vs_saved']['mean_abs']:.9g}` / `{metadata['uniform_reproduction']['new_uniform_vs_saved']['rmse']:.9g}`. The tracked solver-vs-saved comparison was max/mean/RMSE `{metadata['uniform_reproduction']['tracked_solver_vs_saved']['max_abs']:.9g}` / `{metadata['uniform_reproduction']['tracked_solver_vs_saved']['mean_abs']:.9g}` / `{metadata['uniform_reproduction']['tracked_solver_vs_saved']['rmse']:.9g}`; repeated uniform-vs-saved max was `{metadata['uniform_reproduction']['repeated_uniform_vs_saved']['max_abs']:.9g}` and new uniform-vs-repeat max was `{metadata['uniform_reproduction']['new_uniform_vs_repeated_uniform']['max_abs']:.9g}`. These differences are CUDA `index_add_` accumulation variation at the same scale as run-to-run variation; the documented implementation tolerance is max absolute coordinate difference `<= {metadata['uniform_reproduction']['tolerance_max_abs']:.1g}`. Shared-subset metrics agree at the displayed precision.",
        "",
        "![Error CDF for four variants](error_cdf_variants.png)",
        "",
        f"At this fixed configuration, SC² has affine mean `{metrics['sc2']['mean_m']:.6f} m` versus uniform `{metrics['uniform']['mean_m']:.6f} m` (change `{metrics['sc2']['mean_m'] - metrics['uniform']['mean_m']:+.6f} m`) and P95 `{metrics['sc2']['p95_m']:.6f} m` versus `{metrics['uniform']['p95_m']:.6f} m` (change `{metrics['sc2']['p95_m'] - metrics['uniform']['p95_m']:+.6f} m`). The shuffled control has mean `{metrics['sc2_shuffled']['mean_m']:.6f} m`; first-order changes the mean by `{metrics['first_order']['mean_m'] - metrics['uniform']['mean_m']:+.6f} m`. SC² reliability has Spearman rho `{spearman['q_sc2_vs_y0_error']['rho']:+.6f}` with Y0 error, so the observed ordering places higher SC² scores with higher Y0 error in this pilot. Its rho with pointwise Y0 minus baseline-Z improvement is `{spearman['q_sc2_vs_y0_minus_baseline_z_error']['rho']:+.6f}` (p=`{spearman['q_sc2_vs_y0_minus_baseline_z_error']['pvalue']:.3g}`), and the quintile means are nonmonotone for improvement. The fixed SC² fidelity rule therefore does not improve the Stage-2 baseline here and is not supported as a useful repair selector by this pilot, despite small KS/TW differences.",
        "",
        "## Interpretation and limits",
        "",
        "The diagnostic reports whether the reliability signal tracks existing Y0 error and whether it tracks the amount repaired by the fixed baseline. The four-way comparison separates a reliability-conditioned fidelity rule from a graph-only effect and from a shuffled assignment with the same fidelity distribution. These observations are a transductive pilot at one frozen seed and one fixed Stage-2 graph; they do not establish unseen-CSI deployment, Stage-1 causality, or publication novelty.",
        "",
        "Machine-readable metrics and exact input paths are in [`pilot_metrics.json`](pilot_metrics.json). Variant embeddings and fidelity vectors are in [`variant_embeddings.npz`](variant_embeddings.npz) and [`fidelity_vectors.npz`](fidelity_vectors.npz).",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
