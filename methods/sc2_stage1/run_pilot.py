"""SC2-inspired common-support Stage-1 pilot.

This entry point builds the four frozen Stage-1 graph variants and, unless
``--build-only`` is given, runs the existing Siamese and diffusion scripts for
each variant.  The support score is only a graph diagnostic/control variable;
ground truth is loaded after graph construction for evaluation.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components, shortest_path
from scipy.stats import spearmanr
from sklearn.isotonic import IsotonicRegression


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
GRAPH_PATH = ROOT / "results/phase2c_h1_h2/stage1_graph.npz"
DISTANCE_PATH = ROOT / "results/adp_old_fused/stride1_n20997_seed10"
DATA_PATH = ROOT / "datasets/adp_old_shard4"
OUTPUT_ROOT = ROOT / "results/sc2_stage1_consensus_pilot"
N_EXPECTED = 20_997
K_EXPECTED = 20
SEED = 10


def _jsonable(value: object) -> object:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def write_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(value), indent=2, sort_keys=True), encoding="utf-8")


def load_saved_csr(path: Path) -> sp.csr_matrix:
    saved = np.load(path, allow_pickle=True)
    required = {"indices", "indptr", "shape", "data"}
    if not required.issubset(saved.files):
        raise ValueError(f"Stage-1 graph is missing fields: {sorted(required - set(saved.files))}")
    shape = tuple(int(x) for x in saved["shape"])
    graph = sp.csr_matrix((saved["data"], saved["indices"], saved["indptr"]), shape=shape)
    graph.sum_duplicates()
    return graph


def validate_stage1_graph(graph: sp.csr_matrix, fused: np.ndarray, expected_n: int = N_EXPECTED) -> dict[str, object]:
    if graph.shape != (expected_n, expected_n):
        raise ValueError(f"unexpected graph shape: {graph.shape}")
    if fused.shape != graph.shape:
        raise ValueError(f"fused shape {fused.shape} does not match graph {graph.shape}")
    if not sp.isspmatrix_csr(graph):
        raise TypeError(f"expected CSR graph, got {graph.getformat()}")
    if graph.diagonal().any():
        raise ValueError("Stage-1 graph contains self edges")
    degrees = np.diff(graph.indptr).astype(np.int64, copy=False)
    if not np.all(degrees == K_EXPECTED):
        raise ValueError(f"expected exactly {K_EXPECTED} outgoing edges per row")
    rows = np.repeat(np.arange(expected_n, dtype=np.int64), degrees)
    cols = graph.indices.astype(np.int64, copy=False)
    if np.any(rows == cols):
        raise ValueError("Stage-1 graph contains a self edge")
    if not np.all(np.isfinite(graph.data)) or np.any(graph.data <= 0):
        raise ValueError("Stage-1 graph has non-positive or non-finite edge weights")
    if not np.array_equal(graph.data, np.asarray(fused[rows, cols])):
        raise ValueError("saved Stage-1 edge weights do not equal fused distances")
    components = int(connected_components(graph, directed=False, return_labels=False))
    if components != 1:
        raise ValueError(f"Stage-1 graph is disconnected: {components} components")
    return {
        "shape": [expected_n, expected_n],
        "format": graph.getformat(),
        "dtype": str(graph.dtype),
        "directed_nnz": int(graph.nnz),
        "outgoing_degree_min": int(degrees.min()),
        "outgoing_degree_median": float(np.median(degrees)),
        "outgoing_degree_max": int(degrees.max()),
        "self_edge_count": int(np.count_nonzero(graph.diagonal())),
        "connected_components_directed_false": components,
        "edge_weight_sum": float(np.sum(graph.data, dtype=np.float64)),
    }


def edge_arrays(graph: sp.csr_matrix) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    degree = np.diff(graph.indptr).astype(np.int64, copy=False)
    rows = np.repeat(np.arange(graph.shape[0], dtype=np.int64), degree)
    return rows, graph.indices.astype(np.int64, copy=False), graph.data.astype(np.float32, copy=False)


def common_neighbor_support(graph: sp.csr_matrix) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return c_ij, s_ij and the undirected endpoint degrees for graph edges.

    This is the frozen pilot definition: A is the binary undirected union of
    the directed Stage-1 graph, and c is read from A@A.  No GT quantities are
    used here.
    """
    rows, cols, _ = edge_arrays(graph)
    adjacency = graph.copy().astype(np.int32)
    adjacency.data[:] = 1
    adjacency = (adjacency + adjacency.T).tocsr()
    adjacency.data[:] = 1
    adjacency.setdiag(0)
    adjacency.eliminate_zeros()
    undirected_degree = np.asarray(adjacency.sum(axis=1)).ravel().astype(np.int32)
    two_hop = (adjacency @ adjacency).tocsr()
    counts = np.asarray(two_hop[rows, cols]).ravel().astype(np.int32, copy=False)
    denominator = np.maximum(1, np.minimum(undirected_degree[rows] - 1, undirected_degree[cols] - 1))
    scores = np.clip(counts.astype(np.float64) / denominator, 0.0, 1.0).astype(np.float32)
    return counts, scores, undirected_degree, np.asarray(adjacency.sum(axis=1)).ravel().astype(np.int32)


def quantile_labels(values: np.ndarray, bins: int) -> tuple[np.ndarray, np.ndarray]:
    if bins < 1:
        raise ValueError("bins must be positive")
    boundaries = np.quantile(values.astype(np.float64), np.arange(1, bins) / bins)
    labels = np.searchsorted(boundaries, values, side="right").astype(np.int16)
    return labels, boundaries.astype(np.float64)


def shuffled_multipliers(
    edge_weights: np.ndarray,
    multipliers: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    undirected_degree: np.ndarray,
    seed: int = SEED,
) -> tuple[np.ndarray, dict[str, object]]:
    """Shuffle treatment multipliers within length-decile/degree-pair bins."""
    length_bins, length_boundaries = quantile_labels(edge_weights, 10)
    degree_labels, degree_boundaries = quantile_labels(undirected_degree, 4)
    endpoint_lo = np.minimum(degree_labels[rows], degree_labels[cols])
    endpoint_hi = np.maximum(degree_labels[rows], degree_labels[cols])
    keys = np.column_stack((length_bins, endpoint_lo, endpoint_hi))
    groups: dict[tuple[int, int, int], list[int]] = {}
    for edge_index, key in enumerate(map(tuple, keys.tolist())):
        groups.setdefault(key, []).append(edge_index)

    rng = np.random.default_rng(seed)
    shuffled = np.empty_like(multipliers)
    singleton_edges = 0
    group_sizes: dict[str, int] = {}
    for key, edge_indices in sorted(groups.items()):
        index = np.asarray(edge_indices, dtype=np.int64)
        group_sizes["%d-%d-%d" % key] = int(index.size)
        if index.size < 2:
            singleton_edges += int(index.size)
            shuffled[index] = multipliers[index]
        else:
            shuffled[index] = multipliers[index[rng.permutation(index.size)]]
    if not np.array_equal(np.sort(shuffled), np.sort(multipliers)):
        raise AssertionError("shuffling changed the multiplier distribution")
    return shuffled, {
        "seed": int(seed),
        "length_bins": 10,
        "length_quantile_boundaries": length_boundaries,
        "endpoint_degree_bins": 4,
        "endpoint_degree_quantile_boundaries": degree_boundaries,
        "group_key": "(edge_length_decile, min(endpoint_degree_bin), max(endpoint_degree_bin))",
        "group_count": int(len(groups)),
        "singleton_group_count": int(sum(size < 2 for size in (len(v) for v in groups.values()))),
        "singleton_edge_count": int(singleton_edges),
        "singleton_edge_fraction": float(singleton_edges / len(edge_weights)),
        "group_sizes": group_sizes,
    }


def build_variant_graph(graph: sp.csr_matrix, edge_weights: np.ndarray) -> sp.csr_matrix:
    rows, cols, _ = edge_arrays(graph)
    if edge_weights.shape != (graph.nnz,):
        raise ValueError("variant edge weights have the wrong shape")
    variant = sp.csr_matrix((edge_weights.astype(np.float32), (rows, cols)), shape=graph.shape)
    if not np.array_equal(variant.indices, graph.indices) or not np.array_equal(variant.indptr, graph.indptr):
        raise AssertionError("variant graph changed the directed CSR support")
    return variant


def _symmetric_max_abs(matrix: np.ndarray, block_rows: int = 512) -> float:
    result = 0.0
    for start in range(0, matrix.shape[0], block_rows):
        stop = min(start + block_rows, matrix.shape[0])
        result = max(result, float(np.max(np.abs(matrix[start:stop] - matrix[:, start:stop].T))))
    return result


def sampled_geodesic_check(graph: sp.csr_matrix, saved: np.ndarray, seed: int = SEED, count: int = 32) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    sources = rng.choice(graph.shape[0], size=count, replace=False)
    computed = shortest_path(graph, method="auto", directed=False, indices=sources).astype(np.float32)
    reference = np.asarray(saved[sources], dtype=np.float32)
    exact = computed == reference
    close = np.isclose(computed, reference, rtol=1e-6, atol=1e-5)
    return {
        "seed": int(seed),
        "source_count": int(count),
        "source_ids": sources.astype(int).tolist(),
        "distance_count": int(exact.size),
        "exact_match_count": int(np.count_nonzero(exact)),
        "near_exact_match_count": int(np.count_nonzero(close)),
        "max_abs_error": float(np.max(np.abs(computed - reference))),
    }


def summarize_array(values: np.ndarray) -> dict[str, float]:
    values64 = values.astype(np.float64, copy=False)
    return {
        "min": float(np.min(values64)),
        "p01": float(np.percentile(values64, 1)),
        "p25": float(np.percentile(values64, 25)),
        "median": float(np.median(values64)),
        "p75": float(np.percentile(values64, 75)),
        "p99": float(np.percentile(values64, 99)),
        "max": float(np.max(values64)),
        "mean": float(np.mean(values64)),
    }


def save_graph(path: Path, graph: sp.csr_matrix) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        indices=graph.indices,
        indptr=graph.indptr,
        format=np.asarray("csr"),
        shape=np.asarray(graph.shape, dtype=np.int64),
        data=graph.data.astype(np.float32, copy=False),
    )


def build_variants(output_root: Path = OUTPUT_ROOT) -> dict[str, object]:
    output_root.mkdir(parents=True, exist_ok=True)
    graph = load_saved_csr(GRAPH_PATH)
    fused = np.load(DISTANCE_PATH / "fused.npy", mmap_mode="r")
    saved_geo = np.load(DISTANCE_PATH / "geodesic.npy", mmap_mode="r")
    distance_data = np.load(DISTANCE_PATH / "distance_data.npz")
    indices = np.asarray(distance_data["indices"])
    positions = np.asarray(distance_data["positions"])
    if not np.array_equal(indices, np.arange(N_EXPECTED, dtype=indices.dtype)):
        raise ValueError("frozen distance data does not contain the expected sample indices")
    base_graph_summary = validate_stage1_graph(graph, fused)
    baseline_check = sampled_geodesic_check(graph, saved_geo)
    if baseline_check["near_exact_match_count"] != baseline_check["distance_count"]:
        raise ValueError("original Stage-1 graph no longer reproduces saved geodesic distances")

    rows, cols, original_weights = edge_arrays(graph)
    counts, support, undirected_degree, _ = common_neighbor_support(graph)
    treatment_multiplier = 1.0 + 0.5 * (1.0 - support.astype(np.float64))
    uniform_factor = float(np.sum(original_weights * treatment_multiplier, dtype=np.float64) / np.sum(original_weights, dtype=np.float64))
    shuffled_multiplier, shuffle_info = shuffled_multipliers(
        original_weights, treatment_multiplier, rows, cols, undirected_degree, seed=SEED
    )
    edge_weights = {
        "baseline": original_weights,
        "uniform": original_weights * uniform_factor,
        "shuffled": original_weights * shuffled_multiplier,
        "consensus": original_weights * treatment_multiplier,
    }
    multipliers = {
        "baseline": np.ones_like(treatment_multiplier),
        "uniform": np.full_like(treatment_multiplier, uniform_factor),
        "shuffled": shuffled_multiplier,
        "consensus": treatment_multiplier,
    }
    intervention_summary = {
        "support_definition": "c_ij=(A^2)_ij; s_ij=c_ij/max(1,min(deg_i-1,deg_j-1)); clip [0,1]",
        "multiplier_definition": "m_ij=1+0.5*(1-s_ij)",
        "coefficient": 0.5,
        "support_summary": summarize_array(support),
        "common_neighbor_count_summary": summarize_array(counts),
        "undirected_degree_summary": summarize_array(undirected_degree),
        "near_zero_support_count_exact_zero": int(np.count_nonzero(support == 0)),
        "near_zero_support_count_le_0_05": int(np.count_nonzero(support <= 0.05)),
        "treatment_multiplier_summary": summarize_array(treatment_multiplier),
        "uniform_factor": uniform_factor,
        "shuffle": shuffle_info,
    }
    write_json(output_root / "config.json", {
        "experiment": "SC2-inspired Stage-1 consensus pilot",
        "date": "2026-09-25",
        "seed": SEED,
        "n": N_EXPECTED,
        "k": K_EXPECTED,
        "graph_path": str(GRAPH_PATH.relative_to(ROOT)),
        "distance_path": str(DISTANCE_PATH.relative_to(ROOT)),
        "gt_used_in_graph": False,
        "shortest_path_directed": False,
        "base_graph_summary": base_graph_summary,
        "original_geodesic_reproduction": baseline_check,
        "intervention": intervention_summary,
    })
    np.savez_compressed(
        output_root / "edge_scores.npz",
        rows=rows,
        cols=cols,
        original_weights=original_weights,
        common_neighbor_count=counts,
        support=support,
        treatment_multiplier=treatment_multiplier.astype(np.float32),
        shuffled_multiplier=shuffled_multiplier.astype(np.float32),
        undirected_degree=undirected_degree,
    )

    variant_summaries: dict[str, object] = {}
    for name in ("baseline", "uniform", "shuffled", "consensus"):
        variant_dir = output_root / name
        variant_dir.mkdir(parents=True, exist_ok=True)
        variant_graph = build_variant_graph(graph, edge_weights[name])
        components = int(connected_components(variant_graph, directed=False, return_labels=False))
        if components != 1:
            raise ValueError(f"{name} graph is disconnected: {components} components")
        graph_path = variant_dir / "stage1_graph.npz"
        save_graph(graph_path, variant_graph)
        distance_data_path = variant_dir / "distance_data.npz"
        np.savez(distance_data_path, indices=indices, positions=positions)
        geodesic_path = variant_dir / "geodesic.npy"
        t0 = time.time()
        if geodesic_path.exists():
            geodesic = np.load(geodesic_path, mmap_mode="r")
            if geodesic.shape != graph.shape or geodesic.dtype != np.float32:
                raise ValueError(f"existing {geodesic_path} is not the expected float32 matrix")
            elapsed = 0.0
        else:
            geodesic64 = shortest_path(variant_graph, method="auto", directed=False)
            if not np.all(np.isfinite(geodesic64)):
                raise ValueError(f"{name} geodesic contains non-finite values")
            geodesic32 = geodesic64.astype(np.float32)
            np.fill_diagonal(geodesic32, 0.0)
            np.save(geodesic_path, geodesic32)
            geodesic = np.load(geodesic_path, mmap_mode="r")
            del geodesic64, geodesic32
            elapsed = time.time() - t0
        if not np.all(np.isfinite(geodesic)) or not np.allclose(np.diag(geodesic), 0.0, atol=0.0):
            raise ValueError(f"{name} geodesic failed finiteness/diagonal check")
        symmetry_error = _symmetric_max_abs(geodesic)
        if symmetry_error > 1e-5:
            raise ValueError(f"{name} geodesic is not symmetric: {symmetry_error}")
        sample_check = sampled_geodesic_check(variant_graph, saved_geo)
        delta_sample = np.asarray(geodesic[np.random.default_rng(SEED).choice(N_EXPECTED, size=32, replace=False)]) - np.asarray(saved_geo[np.random.default_rng(SEED).choice(N_EXPECTED, size=32, replace=False)])
        # The two RNG expressions above are intentionally seeded identically;
        # keep this diagnostic independent of any evaluation or GT selection.
        delta_sample = delta_sample.astype(np.float64, copy=False)
        graph_summary = {
            "variant": name,
            "directed_nnz": int(variant_graph.nnz),
            "connected_components_directed_false": components,
            "edge_weight_sum": float(np.sum(variant_graph.data, dtype=np.float64)),
            "edge_weight_summary": summarize_array(variant_graph.data),
            "multiplier_summary": summarize_array(multipliers[name]),
            "geodesic_dtype": str(geodesic.dtype),
            "geodesic_shape": list(geodesic.shape),
            "geodesic_symmetry_max_abs": symmetry_error,
            "geodesic_diagonal_max_abs": float(np.max(np.abs(np.diag(geodesic)))),
            "geodesic_sampled_vs_saved": sample_check,
            "geodesic_change_sample_summary": summarize_array(delta_sample.reshape(-1)),
            "geodesic_build_elapsed_s": float(elapsed),
        }
        write_json(variant_dir / "graph_summary.json", graph_summary)
        variant_summaries[name] = graph_summary
        del geodesic
    write_json(output_root / "variant_summaries.json", variant_summaries)
    return {
        "base_graph_summary": base_graph_summary,
        "original_geodesic_reproduction": baseline_check,
        "intervention": intervention_summary,
        "variants": variant_summaries,
    }


def _pair_keys(left: np.ndarray, right: np.ndarray, n: int = N_EXPECTED) -> np.ndarray:
    low = np.minimum(left, right).astype(np.int64, copy=False)
    high = np.maximum(left, right).astype(np.int64, copy=False)
    return low * n + high


def _decode_pair_keys(keys: np.ndarray, n: int = N_EXPECTED) -> tuple[np.ndarray, np.ndarray]:
    return (keys // n).astype(np.int32), (keys % n).astype(np.int32)


def _sample_unique_uniform_pairs(
    rng: np.random.Generator,
    count: int,
    used: set[int],
    n: int = N_EXPECTED,
) -> tuple[np.ndarray, np.ndarray]:
    accepted: list[int] = []
    while len(accepted) < count:
        batch_size = max(4096, 4 * (count - len(accepted)))
        left = rng.integers(0, n, size=batch_size, dtype=np.int32)
        right = rng.integers(0, n, size=batch_size, dtype=np.int32)
        candidate = np.unique(_pair_keys(left[left != right], right[left != right], n))
        for key in candidate.tolist():
            key_int = int(key)
            if key_int not in used:
                used.add(key_int)
                accepted.append(key_int)
                if len(accepted) == count:
                    break
    return _decode_pair_keys(np.asarray(accepted, dtype=np.int64), n)


def _sample_unique_local_pairs(
    rng: np.random.Generator,
    neighbors128: np.ndarray,
    count: int,
    used: set[int],
    n: int = N_EXPECTED,
) -> tuple[np.ndarray, np.ndarray]:
    accepted: list[int] = []
    while len(accepted) < count:
        batch_size = max(4096, 4 * (count - len(accepted)))
        left = rng.integers(0, n, size=batch_size, dtype=np.int32)
        target_column = rng.integers(0, neighbors128.shape[1], size=batch_size)
        right = neighbors128[left, target_column].astype(np.int32, copy=False)
        valid = left != right
        candidate = np.unique(_pair_keys(left[valid], right[valid], n))
        for key in candidate.tolist():
            key_int = int(key)
            if key_int not in used:
                used.add(key_int)
                accepted.append(key_int)
                if len(accepted) == count:
                    break
    return _decode_pair_keys(np.asarray(accepted, dtype=np.int64), n)


def _gate_metrics(
    model: IsotonicRegression,
    distance_values: np.ndarray,
    physical_values: np.ndarray,
) -> dict[str, float]:
    predicted = np.asarray(model.predict(distance_values), dtype=np.float64)
    residual = predicted - physical_values
    abs_residual = np.abs(residual)
    under = residual < 0
    over = residual > 0
    rho = spearmanr(distance_values, physical_values).statistic
    return {
        "mae_m": float(np.mean(abs_residual)),
        "p95_abs_residual_m": float(np.percentile(abs_residual, 95)),
        "mean_signed_residual_m_pred_minus_physical": float(np.mean(residual)),
        "mean_under_residual_m_pred_minus_physical": float(np.mean(residual[under])) if np.any(under) else 0.0,
        "mean_over_residual_m_pred_minus_physical": float(np.mean(residual[over])) if np.any(over) else 0.0,
        "under_fraction": float(np.mean(under)),
        "over_fraction": float(np.mean(over)),
        "spearman_rho": float(rho),
    }


def run_geodesic_gate(output_root: Path = OUTPUT_ROOT) -> dict[str, object]:
    """Run the frozen GT-assisted held-out D_geo gate before any FCF training."""
    seed = 20260925
    calibration_count = 100_000
    uniform_count = 50_000
    local_count = 50_000
    rng = np.random.default_rng(seed)
    used: set[int] = set()
    calibration_left, calibration_right = _sample_unique_uniform_pairs(rng, calibration_count, used)
    uniform_left, uniform_right = _sample_unique_uniform_pairs(rng, uniform_count, used)
    top_path = DISTANCE_PATH / "geodesic_top512.npy"
    if not top_path.exists():
        raise FileNotFoundError(f"required local sampling frame missing: {top_path}")
    top_neighbors = np.load(top_path, mmap_mode="r")
    if top_neighbors.shape != (N_EXPECTED, 512):
        raise ValueError(f"unexpected local sampling frame shape: {top_neighbors.shape}")
    local_left, local_right = _sample_unique_local_pairs(rng, np.asarray(top_neighbors[:, :128]), local_count, used)
    all_pairs = np.concatenate([
        _pair_keys(calibration_left, calibration_right),
        _pair_keys(uniform_left, uniform_right),
        _pair_keys(local_left, local_right),
    ])
    if len(np.unique(all_pairs)) != len(all_pairs):
        raise AssertionError("gate pair sets are not disjoint")
    np.savez_compressed(
        output_root / "gate_pairs.npz",
        calibration_left=calibration_left,
        calibration_right=calibration_right,
        uniform_left=uniform_left,
        uniform_right=uniform_right,
        local_left=local_left,
        local_right=local_right,
    )
    distance_data = np.load(DISTANCE_PATH / "distance_data.npz")
    positions = np.asarray(distance_data["positions"], dtype=np.float64)
    physical_calibration = np.linalg.norm(positions[calibration_left] - positions[calibration_right], axis=1)
    physical_uniform = np.linalg.norm(positions[uniform_left] - positions[uniform_right], axis=1)
    physical_local = np.linalg.norm(positions[local_left] - positions[local_right], axis=1)
    split_info = {
        "seed": seed,
        "calibration_count": calibration_count,
        "uniform_test_count": uniform_count,
        "local_test_count": local_count,
        "pair_sampling": "uniform unordered IDs for calibration/uniform; source uniform and target uniform among original geodesic_top512 first 128 columns for local",
        "local_sampling_frame": str(top_path.relative_to(ROOT)),
        "local_sampling_frame_columns": 128,
        "pair_sets_disjoint": True,
        "self_pairs_rejected": True,
    }
    variant_metrics: dict[str, object] = {}
    for name in ("baseline", "uniform", "shuffled", "consensus"):
        geo_path = output_root / name / "geodesic.npy"
        geodesic = np.load(geo_path, mmap_mode="r")
        calibration_distance = np.asarray(geodesic[calibration_left, calibration_right], dtype=np.float64)
        uniform_distance = np.asarray(geodesic[uniform_left, uniform_right], dtype=np.float64)
        local_distance = np.asarray(geodesic[local_left, local_right], dtype=np.float64)
        model = IsotonicRegression(increasing=True, out_of_bounds="clip")
        model.fit(calibration_distance, physical_calibration)
        np.savez_compressed(
            output_root / name / "gate_isotonic.npz",
            x_thresholds=np.asarray(model.X_thresholds_, dtype=np.float64),
            y_thresholds=np.asarray(model.y_thresholds_, dtype=np.float64),
        )
        variant_metrics[name] = {
            "uniform_test": _gate_metrics(model, uniform_distance, physical_uniform),
            "local_test": _gate_metrics(model, local_distance, physical_local),
            "calibration": {
                "distance_summary": summarize_array(calibration_distance),
                "physical_distance_summary_m": summarize_array(physical_calibration),
                "isotonic_knot_count": int(len(model.X_thresholds_)),
            },
        }
    base_uniform = variant_metrics["baseline"]["uniform_test"]
    base_local = variant_metrics["baseline"]["local_test"]
    cons_uniform = variant_metrics["consensus"]["uniform_test"]
    cons_local = variant_metrics["consensus"]["local_test"]
    uniform_comparisons = {
        name: float(variant_metrics[name]["uniform_test"]["mae_m"]) for name in ("baseline", "uniform", "shuffled", "consensus")
    }
    uniform_improvements = {
        name: float(variant_metrics[name]["uniform_test"]["mae_m"] - cons_uniform["mae_m"])
        for name in ("baseline", "uniform", "shuffled")
    }
    if base_local["mae_m"] == 0:
        local_mae_ok = cons_local["mae_m"] <= 0
    else:
        local_mae_ok = cons_local["mae_m"] <= 1.01 * base_local["mae_m"]
    if base_local["p95_abs_residual_m"] == 0:
        local_p95_ok = cons_local["p95_abs_residual_m"] <= 0
    else:
        local_p95_ok = cons_local["p95_abs_residual_m"] <= 1.01 * base_local["p95_abs_residual_m"]
    decision = {
        "consensus_uniform_mae_at_least_1pct_lower_vs_each_control": bool(all(cons_uniform["mae_m"] <= 0.99 * uniform_comparisons[name] for name in ("baseline", "uniform", "shuffled"))),
        "consensus_local_mae_within_1pct_of_baseline": bool(local_mae_ok),
        "consensus_uniform_p95_within_1pct_of_baseline": bool(cons_uniform["p95_abs_residual_m"] <= 1.01 * base_uniform["p95_abs_residual_m"] if base_uniform["p95_abs_residual_m"] != 0 else cons_uniform["p95_abs_residual_m"] <= 0),
        "consensus_local_p95_within_1pct_of_baseline": bool(local_p95_ok),
        "uniform_mae_m": uniform_comparisons,
        "consensus_uniform_mae_improvement_m_vs_controls": uniform_improvements,
    }
    decision["gate_positive"] = bool(all(decision[key] for key in (
        "consensus_uniform_mae_at_least_1pct_lower_vs_each_control",
        "consensus_local_mae_within_1pct_of_baseline",
        "consensus_uniform_p95_within_1pct_of_baseline",
        "consensus_local_p95_within_1pct_of_baseline",
    )))
    result = {
        "experiment": "SC2-inspired Stage-1 consensus pilot",
        "gate": "independent held-out physical-distance isotonic gate",
        "split": split_info,
        "variants": variant_metrics,
        "decision": decision,
        "gt_role": "physical positions used only for calibration and held-out evaluation; no GT entered graph construction, shortest paths, or FCF targets",
    }
    write_json(output_root / "geodesic_gate.json", result)
    return result


def run_mds_gate(output_root: Path = OUTPUT_ROOT) -> dict[str, object]:
    """Run the fixed stride-10 MDS gate before any FCF training."""
    from common_metrics import chart_metrics, localization_error_summary
    from run_adp_fused import run_mds

    mds_indices = np.arange(0, N_EXPECTED, 10, dtype=np.int64)
    distance_data = np.load(DISTANCE_PATH / "distance_data.npz")
    positions = np.asarray(distance_data["positions"], dtype=np.float32)
    subset_positions = positions[mds_indices]
    variant_metrics: dict[str, object] = {}
    for name in ("baseline", "uniform", "shuffled", "consensus"):
        geodesic = np.load(output_root / name / "geodesic.npy", mmap_mode="r")
        subset_distances = np.asarray(geodesic[np.ix_(mds_indices, mds_indices)], dtype=np.float32)
        t0 = time.time()
        embedding = run_mds(subset_distances, seed=SEED, max_iter=80, n_init=4)
        errors = localization_error_summary(subset_positions, embedding)
        metrics = chart_metrics(subset_positions, embedding)
        variant_dir = output_root / name
        np.savez(variant_dir / "mds.npz", indices=mds_indices, positions=subset_positions, embedding=embedding)
        record = {
            "variant": name,
            "indices_definition": "np.arange(0,20997,10)",
            "num_samples": int(len(mds_indices)),
            "seed": SEED,
            "max_iter": 80,
            "n_init": 4,
            "metrics": metrics,
            "localization_errors_after_affine": errors,
            "elapsed_s": float(time.time() - t0),
        }
        write_json(variant_dir / "mds_result.json", record)
        variant_metrics[name] = record
        del subset_distances, geodesic, embedding
    baseline = variant_metrics["baseline"]
    consensus = variant_metrics["consensus"]
    controls = {name: variant_metrics[name] for name in ("baseline", "uniform", "shuffled")}
    decision = {
        "consensus_mean_lower_by_at_least_0_005_m_vs_each_control": bool(all(
            consensus["localization_errors_after_affine"]["mean_m"]
            <= controls[name]["localization_errors_after_affine"]["mean_m"] - 0.005
            for name in controls
        )),
        "consensus_p95_not_more_than_0_01_m_above_baseline": bool(
            consensus["localization_errors_after_affine"]["p95_m"]
            <= baseline["localization_errors_after_affine"]["p95_m"] + 0.01
        ),
        "consensus_ct_not_more_than_0_005_below_baseline": bool(
            consensus["metrics"]["ct"] >= baseline["metrics"]["ct"] - 0.005
        ),
        "consensus_tw_not_more_than_0_005_below_baseline": bool(
            consensus["metrics"]["tw"] >= baseline["metrics"]["tw"] - 0.005
        ),
        "consensus_ks_not_more_than_0_01_above_baseline": bool(
            consensus["metrics"]["ks"] <= baseline["metrics"]["ks"] + 0.01
        ),
    }
    decision["gate_positive"] = bool(all(decision.values()))
    result = {
        "experiment": "SC2-inspired Stage-1 consensus pilot",
        "gate": "fixed stride-10 MDS pre-FCF gate",
        "mds_indices_definition": "np.arange(0,20997,10)",
        "mds_sample_count": int(len(mds_indices)),
        "seed": SEED,
        "max_iter": 80,
        "n_init": 4,
        "variants": variant_metrics,
        "decision": decision,
        "gt_role": "positions used only for affine evaluation and CT/TW/KS; MDS receives only the candidate D_geo submatrix",
    }
    write_json(output_root / "mds_gate.json", result)
    return result


def write_gate_failure_report(output_root: Path, mds_gate: dict[str, object], isotonic_gate: dict[str, object]) -> None:
    decision = mds_gate["decision"]
    iso_decision = isotonic_gate["decision"]
    variant_summaries = json.loads((output_root / "variant_summaries.json").read_text(encoding="utf-8"))
    consensus_sum = variant_summaries["consensus"]["edge_weight_sum"]
    shuffled_sum = variant_summaries["shuffled"]["edge_weight_sum"]
    shuffled_gap_fraction = (consensus_sum - shuffled_sum) / consensus_sum
    mds_rows = []
    for name in ("baseline", "uniform", "shuffled", "consensus"):
        item = mds_gate["variants"][name]
        mds_rows.append(
            f"| {name} | {item['localization_errors_after_affine']['mean_m']:.6f} | {item['localization_errors_after_affine']['p95_m']:.6f} | {item['metrics']['ct']:.6f} | {item['metrics']['tw']:.6f} | {item['metrics']['ks']:.6f} |"
        )
    rows = []
    for name in ("baseline", "uniform", "shuffled", "consensus"):
        item = isotonic_gate["variants"][name]
        rows.append(
            f"| {name} | {item['uniform_test']['mae_m']:.6f} | {item['uniform_test']['p95_abs_residual_m']:.6f} | {item['local_test']['mae_m']:.6f} | {item['local_test']['p95_abs_residual_m']:.6f} | {item['uniform_test']['spearman_rho']:.6f} | {item['local_test']['spearman_rho']:.6f} |"
        )
    text = f"""# SC2-inspired Stage-1 consensus pilot

The frozen independent Stage-1 distance gates were evaluated before any FCF training. Because the primary MDS gate result is `{decision['gate_positive']}`, this run stopped before all Siamese and Stage-2 training pipelines.

## Primary fixed-subset MDS gate

The existing repository MDS implementation was run on exactly `np.arange(0,20997,10)` (2,100 samples) from each full-graph D_geo matrix, with seed 10, max_iter 80, and n_init 4. The same physical positions were used only for affine evaluation.

The uniform graph matches the consensus total edge weight to float32 summation precision. The shuffled graph preserves the multiplier distribution within bins but its realized edge-weight sum is {shuffled_sum:.6f}, versus consensus {consensus_sum:.6f}; this is a {shuffled_gap_fraction:.4%} lower total and is recorded as a control limitation.

| variant | MDS mean (m) | MDS p95 (m) | CT | TW | KS |
|---|---:|---:|---:|---:|---:|
{chr(10).join(mds_rows)}

Frozen MDS conditions: consensus mean at least 0.005 m lower than every control; p95 no more than 0.01 m above baseline; CT/TW no more than 0.005 below baseline; KS no more than 0.01 above baseline.

- Consensus mean condition: `{decision['consensus_mean_lower_by_at_least_0_005_m_vs_each_control']}`
- Consensus p95 condition: `{decision['consensus_p95_not_more_than_0_01_m_above_baseline']}`
- Consensus CT condition: `{decision['consensus_ct_not_more_than_0_005_below_baseline']}`
- Consensus TW condition: `{decision['consensus_tw_not_more_than_0_005_below_baseline']}`
- Consensus KS condition: `{decision['consensus_ks_not_more_than_0_01_above_baseline']}`
- MDS gate positive: `{decision['gate_positive']}`

## Secondary held-out physical-distance diagnostic

Each variant was independently calibrated with monotone isotonic regression on 100,000 disjoint unordered pairs. The uniform test has 50,000 uniformly sampled unordered pairs; the local test has 50,000 source/target pairs drawn from the saved original `geodesic_top512.npy` first 128 columns. Pair generation used seed 20260925 and is retained in `gate_pairs.npz`. Ground truth positions were used only for calibration and held-out scoring.

| variant | uniform MAE (m) | uniform p95 abs (m) | local MAE (m) | local p95 abs (m) | uniform Spearman | local Spearman |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

## Frozen decision

- Consensus uniform-test MAE at least 1% lower than baseline, uniform, and shuffled: `{iso_decision['consensus_uniform_mae_at_least_1pct_lower_vs_each_control']}`
- Consensus local-test MAE within 1% of baseline: `{iso_decision['consensus_local_mae_within_1pct_of_baseline']}`
- Consensus uniform-test p95 within 1% of baseline: `{iso_decision['consensus_uniform_p95_within_1pct_of_baseline']}`
- Consensus local-test p95 within 1% of baseline: `{iso_decision['consensus_local_p95_within_1pct_of_baseline']}`

The isotonic pair test is a secondary diagnostic proxy for physical-distance fidelity after per-variant monotone recalibration; it does not decide whether to train. Since the primary MDS gate failed, the budgeted FCF and Stage-2 runs were not started. No coefficient, endpoint, or threshold was tuned after observing outcomes.

Artifacts: `config.json`, `variant_summaries.json`, `mds_gate.json`, per-variant `mds.npz`/`mds_result.json`, `gate_pairs.npz`, `geodesic_gate.json`, per-variant `geodesic.npy`, `graph_summary.json`, and `gate_isotonic.npz`.

Executed commands: `D:\\anaconda3\\Scripts\\conda.exe run -n diffusion_charting python methods/sc2_stage1/run_pilot.py --build-only --device cuda:0`; then `D:\\anaconda3\\Scripts\\conda.exe run -n diffusion_charting python methods/sc2_stage1/run_pilot.py --skip-build --device cuda:0`. Environment: Python 3.11.15, torch 2.11.0+cu128, NVIDIA GeForce RTX 3050 Laptop GPU. Focused semantic tests: `D:\\anaconda3\\Scripts\\conda.exe run -n diffusion_charting python -m unittest tests.test_sc2_stage1 -v` (3/3 passed).
"""
    (output_root / "REPORT.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--skip-build", action="store_true")
    args = parser.parse_args()
    output_root = args.output_root if args.output_root.is_absolute() else ROOT / args.output_root
    if not args.skip_build:
        build_variants(output_root)
    if args.build_only:
        return
    mds_gate = run_mds_gate(output_root)
    isotonic_gate = run_geodesic_gate(output_root)
    write_gate_failure_report(output_root, mds_gate, isotonic_gate)
    print(json.dumps(mds_gate["decision"], indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
