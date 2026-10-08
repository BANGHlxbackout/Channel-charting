from __future__ import annotations

import json
import math
import platform
import time
from array import array
from importlib.metadata import version
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import shortest_path
from scipy.stats import fisher_exact, rankdata
from sklearn.isotonic import IsotonicRegression
from sklearn.neighbors import NearestNeighbors


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_DIR = REPO_ROOT / "results/phase2c_h1_h2"
FIGURE_DIR = RESULT_DIR / "figures"

GRAPH_PATH = RESULT_DIR / "stage1_graph.npz"
GATE_AUDIT_PATH = RESULT_DIR / "audit.json"
DISTANCE_RUN = REPO_ROOT / "results/adp_old_fused/stride1_n20997_seed10"
ADP_PATH = DISTANCE_RUN / "adp.npy"
D_GEO_PATH = DISTANCE_RUN / "geodesic.npy"
DISTANCE_DATA_PATH = DISTANCE_RUN / "distance_data.npz"
BASELINE_RESULT_PATH = DISTANCE_RUN / "result.json"
TIMESTAMP_PATH = REPO_ROOT / "datasets/adp_old_shard4/timestamps.npy"

SEED = 10
PHYSICAL_Q = 16
SOURCES_PER_DENSITY_QUARTILE = 32
DISCOVERY_SOURCE_COUNT = 128
VALIDATION_SOURCE_COUNT = 128
TARGETS_PER_SOURCE = 512
CALIBRATION_PAIR_COUNT = 100_000
SEVERE_CALIBRATION_QUANTILE = 0.05
CONTROL_REPLICATES = 50
TOP_FRACTION_FOR_DECISION = 0.01
EPS = 1e-12


def package_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "numpy": version("numpy"),
        "scipy": version("scipy"),
        "scikit_learn": version("scikit-learn"),
        "pandas": version("pandas"),
        "pyarrow": version("pyarrow"),
        "matplotlib": version("matplotlib"),
    }


def read_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def extract_undirected_edges(
    directed_graph: sp.csr_matrix,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Return the unordered union without changing the verified directed graph."""
    coo = directed_graph.tocoo()
    n = directed_graph.shape[0]
    low = np.minimum(coo.row, coo.col).astype(np.int32, copy=False)
    high = np.maximum(coo.row, coo.col).astype(np.int32, copy=False)
    keys = low.astype(np.int64) * n + high.astype(np.int64)
    order = np.argsort(keys, kind="stable")
    keys = keys[order]
    low = low[order]
    high = high[order]
    rows = coo.row[order]
    data = coo.data[order]

    starts = np.r_[0, np.flatnonzero(keys[1:] != keys[:-1]) + 1]
    is_low_to_high = rows == low
    support_low_to_high = np.logical_or.reduceat(is_low_to_high, starts)
    support_high_to_low = np.logical_or.reduceat(~is_low_to_high, starts)
    weight_low_to_high = np.minimum.reduceat(
        np.where(is_low_to_high, data, np.inf),
        starts,
    ).astype(np.float32)
    weight_high_to_low = np.minimum.reduceat(
        np.where(~is_low_to_high, data, np.inf),
        starts,
    ).astype(np.float32)
    effective_weight = np.minimum(weight_low_to_high, weight_high_to_low)
    return (
        low[starts],
        high[starts],
        effective_weight,
        support_low_to_high,
        support_high_to_low,
        weight_low_to_high,
        weight_high_to_low,
    )


def incident_median_scale(
    n: int,
    edge_u: np.ndarray,
    edge_v: np.ndarray,
    edge_weight: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    rows = np.concatenate((edge_u, edge_v))
    cols = np.concatenate((edge_v, edge_u))
    weights = np.concatenate((edge_weight, edge_weight))
    graph = sp.csr_matrix((weights, (rows, cols)), shape=(n, n))
    degree = np.diff(graph.indptr).astype(np.int32, copy=False)
    scale = np.empty(n, dtype=np.float32)
    for node in range(n):
        start, stop = graph.indptr[node : node + 2]
        scale[node] = np.median(graph.data[start:stop])
    if np.any(scale <= 0):
        raise ValueError("wireless incident-median scale contains non-positive values")
    return scale, degree


def physical_local_scale(positions: np.ndarray, q: int) -> tuple[np.ndarray, int]:
    model = NearestNeighbors(
        n_neighbors=q,
        algorithm="kd_tree",
        metric="euclidean",
        n_jobs=-1,
    ).fit(positions)
    distances, _ = model.kneighbors(X=None, n_neighbors=q, return_distance=True)
    scale = np.median(distances, axis=1).astype(np.float32)
    fallback_rows = np.flatnonzero(scale <= 0)
    for row in fallback_rows:
        positive = distances[row, distances[row] > 0]
        if len(positive) == 0:
            raise ValueError(f"physical q-neighbor set for node {int(row)} has no positive distance")
        scale[row] = np.median(positive)
    return scale, int(len(fallback_rows))


def robust_mismatch_scores(
    wireless_normalized: np.ndarray,
    physical_normalized: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    log_ratio = np.log(
        (physical_normalized.astype(np.float64) + EPS)
        / (wireless_normalized.astype(np.float64) + EPS)
    )
    center = float(np.median(log_ratio))
    mad = float(np.median(np.abs(log_ratio - center)))
    robust_scale = 1.4826 * mad
    if robust_scale <= 0:
        raise ValueError("robust mismatch scale is zero")
    score = (log_ratio - center) / robust_scale
    return log_ratio, score, center, robust_scale


def quantile_bins(values: np.ndarray, bin_count: int) -> tuple[np.ndarray, np.ndarray]:
    boundaries = np.quantile(
        values.astype(np.float64, copy=False),
        np.arange(1, bin_count) / bin_count,
    )
    bins = np.searchsorted(boundaries, values, side="right").astype(np.int16)
    return bins, boundaries


def stratified_source_split(
    physical_scale: np.ndarray,
    rng: np.random.Generator,
    per_quartile_per_split: int = SOURCES_PER_DENSITY_QUARTILE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    quartiles, boundaries = quantile_bins(physical_scale, 4)
    discovery = []
    validation = []
    required = 2 * per_quartile_per_split
    for quartile in range(4):
        members = np.flatnonzero(quartiles == quartile)
        if len(members) < required:
            raise ValueError(f"density quartile {quartile} has only {len(members)} nodes")
        selected = rng.permutation(members)[:required]
        discovery.extend(selected[:per_quartile_per_split])
        validation.extend(selected[per_quartile_per_split:])
    return (
        np.asarray(discovery, dtype=np.int32),
        np.asarray(validation, dtype=np.int32),
        quartiles,
        boundaries,
    )


def pair_key(left: int, right: int, n: int) -> int:
    low, high = (left, right) if left < right else (right, left)
    return low * n + high


def sample_targets(
    sources: np.ndarray,
    target_count: int,
    n: int,
    rng: np.random.Generator,
    used_pair_keys: set[int],
    positions: np.ndarray,
    d_geo: np.ndarray,
) -> np.ndarray:
    targets = np.empty((len(sources), target_count), dtype=np.int32)
    for source_row, source_value in enumerate(sources):
        source = int(source_value)
        local_targets: set[int] = set()
        ordered_targets: list[int] = []
        while len(ordered_targets) < target_count:
            target = int(rng.integers(0, n - 1))
            if target >= source:
                target += 1
            key = pair_key(source, target, n)
            if key in used_pair_keys or target in local_targets:
                continue
            if np.linalg.norm(positions[source] - positions[target]) <= 0:
                continue
            if float(d_geo[source, target]) <= 0:
                continue
            local_targets.add(target)
            ordered_targets.append(target)
            used_pair_keys.add(key)
        targets[source_row] = np.asarray(ordered_targets, dtype=np.int32)
    return targets


def sample_calibration_pairs(
    count: int,
    n: int,
    rng: np.random.Generator,
    used_pair_keys: set[int],
    positions: np.ndarray,
    d_geo: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    left = np.empty(count, dtype=np.int32)
    right = np.empty(count, dtype=np.int32)
    selected = 0
    while selected < count:
        a = int(rng.integers(0, n))
        b = int(rng.integers(0, n - 1))
        if b >= a:
            b += 1
        key = pair_key(a, b, n)
        if key in used_pair_keys:
            continue
        if np.linalg.norm(positions[a] - positions[b]) <= 0:
            continue
        if float(d_geo[a, b]) <= 0:
            continue
        used_pair_keys.add(key)
        left[selected] = a
        right[selected] = b
        selected += 1
    return left, right


def pair_distances(
    left: np.ndarray,
    right: np.ndarray,
    positions: np.ndarray,
    d_geo: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    physical = np.linalg.norm(positions[left] - positions[right], axis=1).astype(np.float32)
    geodesic = np.asarray(d_geo[left, right], dtype=np.float32)
    if np.any(physical <= 0) or np.any(geodesic <= 0):
        raise ValueError("sampled pairs must have positive physical and geodesic distances")
    return physical, geodesic


def fit_calibration(
    physical: np.ndarray,
    geodesic: np.ndarray,
) -> tuple[IsotonicRegression, np.ndarray, np.ndarray, float]:
    log_physical = np.log(physical.astype(np.float64))
    log_geodesic = np.log(geodesic.astype(np.float64))
    model = IsotonicRegression(increasing=True, out_of_bounds="clip")
    predicted = model.fit_transform(log_physical, log_geodesic)
    residual = log_geodesic - predicted
    threshold = float(np.quantile(residual, SEVERE_CALIBRATION_QUANTILE))
    return model, predicted, residual, threshold


def calibrated_pair_values(
    physical: np.ndarray,
    geodesic: np.ndarray,
    calibration: IsotonicRegression,
    severe_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    predicted = calibration.predict(np.log(physical.astype(np.float64)))
    residual = np.log(geodesic.astype(np.float64)) - predicted
    severe = residual <= severe_threshold
    return predicted, residual, severe


def trace_sampled_paths(
    graph: sp.csr_matrix,
    sources: np.ndarray,
    targets: np.ndarray,
    severe_paths: np.ndarray,
    sorted_edge_keys: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    distances, predecessors = shortest_path(
        graph,
        method="auto",
        directed=False,
        indices=sources,
        return_predecessors=True,
    )
    n = graph.shape[0]
    flat_edges = array("i")
    path_count = targets.size
    offsets = np.empty(path_count + 1, dtype=np.int64)
    hops = np.empty(path_count, dtype=np.int32)
    offsets[0] = 0
    path_index = 0

    for source_row, source_value in enumerate(sources):
        source = int(source_value)
        for target_value in targets[source_row]:
            current = int(target_value)
            hop_count = 0
            while current != source:
                predecessor = int(predecessors[source_row, current])
                if predecessor < 0:
                    raise RuntimeError(f"no predecessor from {source} to {int(target_value)}")
                key = pair_key(predecessor, current, n)
                edge_id = int(np.searchsorted(sorted_edge_keys, key))
                if edge_id == len(sorted_edge_keys) or int(sorted_edge_keys[edge_id]) != key:
                    raise RuntimeError(f"shortest path used unknown Stage-1 edge {predecessor}-{current}")
                flat_edges.append(edge_id)
                current = predecessor
                hop_count += 1
                if hop_count > n:
                    raise RuntimeError("predecessor cycle detected")
            hops[path_index] = hop_count
            path_index += 1
            offsets[path_index] = len(flat_edges)

    flat = np.frombuffer(flat_edges, dtype=np.int32).copy()
    per_path_lengths = np.diff(offsets)
    path_ids = np.repeat(np.arange(path_count, dtype=np.int32), per_path_lengths)
    usage = np.bincount(flat, minlength=len(sorted_edge_keys)).astype(np.int32)
    severe_usage = np.bincount(
        flat[severe_paths[path_ids]],
        minlength=len(sorted_edge_keys),
    ).astype(np.int32)
    target_distances = distances[
        np.arange(len(sources), dtype=np.int32)[:, None],
        targets,
    ].astype(np.float32)
    return flat, offsets, hops, usage, severe_usage, target_distances


def sequence_concentration(
    flat_path_edges: np.ndarray,
    offsets: np.ndarray,
    severe_paths: np.ndarray,
    edge_sequence: np.ndarray,
    cutoffs: np.ndarray,
    edge_count: int,
) -> np.ndarray:
    max_cutoff = int(np.max(cutoffs))
    ranks = np.full(edge_count, max_cutoff, dtype=np.int32)
    ranks[edge_sequence[:max_cutoff]] = np.arange(max_cutoff, dtype=np.int32)
    path_min_rank = np.minimum.reduceat(ranks[flat_path_edges], offsets[:-1])
    severe_min_rank = path_min_rank[severe_paths]
    if len(severe_min_rank) == 0:
        raise ValueError("no severe paths available for concentration analysis")
    return np.asarray(
        [np.mean(severe_min_rank < cutoff) for cutoff in cutoffs],
        dtype=np.float64,
    )


def edge_set_path_hits(
    flat_path_edges: np.ndarray,
    offsets: np.ndarray,
    selected_edges: np.ndarray,
) -> np.ndarray:
    selected_on_steps = selected_edges[flat_path_edges]
    return np.logical_or.reduceat(selected_on_steps, offsets[:-1])


def matched_random_sequence(
    ranked_edges: np.ndarray,
    edge_strata: np.ndarray,
    max_count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    target = ranked_edges[:max_count]
    target_strata = edge_strata[target]
    excluded = np.zeros(len(edge_strata), dtype=bool)
    excluded[target] = True
    sequence = np.empty(max_count, dtype=np.int32)
    for stratum in np.unique(target_strata):
        positions = np.flatnonzero(target_strata == stratum)
        pool = np.flatnonzero((edge_strata == stratum) & ~excluded)
        if len(pool) < len(positions):
            raise RuntimeError(
                f"matched-control stratum {int(stratum)} has {len(pool)} candidates "
                f"for {len(positions)} requested edges"
            )
        sequence[positions] = rng.choice(pool, size=len(positions), replace=False)
    return sequence


def branch_comparison(branch: np.ndarray, top_mask: np.ndarray) -> dict[str, object]:
    result: dict[str, object] = {}
    for label in ("adp", "timestamp", "tie"):
        overall_count = int(np.count_nonzero(branch == label))
        top_count = int(np.count_nonzero((branch == label) & top_mask))
        overall_fraction = overall_count / len(branch)
        top_fraction = top_count / int(np.count_nonzero(top_mask))
        table = np.array(
            [
                [top_count, int(np.count_nonzero(top_mask)) - top_count],
                [overall_count - top_count, len(branch) - overall_count - int(np.count_nonzero(top_mask)) + top_count],
            ],
            dtype=np.int64,
        )
        odds_ratio, p_value = fisher_exact(table)
        result[label] = {
            "overall_count": overall_count,
            "overall_fraction": float(overall_fraction),
            "top1pct_count": top_count,
            "top1pct_fraction": float(top_fraction),
            "fraction_enrichment": float(top_fraction / overall_fraction) if overall_fraction else None,
            "fisher_odds_ratio": float(odds_ratio),
            "fisher_p_value": float(p_value),
        }
    return result


def distribution_summary(values: np.ndarray) -> dict[str, float]:
    values64 = values.astype(np.float64, copy=False)
    return {
        "min": float(np.min(values64)),
        "p01": float(np.percentile(values64, 1)),
        "p05": float(np.percentile(values64, 5)),
        "median": float(np.median(values64)),
        "p95": float(np.percentile(values64, 95)),
        "p99": float(np.percentile(values64, 99)),
        "max": float(np.max(values64)),
    }


def control_summary(samples: np.ndarray) -> dict[str, list[float]]:
    return {
        "mean": np.mean(samples, axis=0).astype(float).tolist(),
        "p05": np.percentile(samples, 5, axis=0).astype(float).tolist(),
        "p95": np.percentile(samples, 95, axis=0).astype(float).tolist(),
    }


def save_figures(
    edge_frame: pd.DataFrame,
    positions: np.ndarray,
    ranked_edges: np.ndarray,
    cutoffs: np.ndarray,
    validation_curve: np.ndarray,
    uniform_controls: np.ndarray,
    matched_controls: np.ndarray,
    branch_stats: dict[str, object],
) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    plot_rng = np.random.default_rng(SEED)
    sample_count = min(50_000, len(edge_frame))
    sample = plot_rng.choice(len(edge_frame), size=sample_count, replace=False)

    fig, ax = plt.subplots(figsize=(7, 5.5))
    points = ax.scatter(
        edge_frame.loc[sample, "wireless_normalized"],
        edge_frame.loc[sample, "physical_normalized"],
        c=edge_frame.loc[sample, "robust_mismatch_score"],
        s=5,
        alpha=0.35,
        cmap="viridis",
    )
    ax.set_xlabel("Stage-1 wireless normalized edge length")
    ax.set_ylabel("Physical normalized edge length (GT diagnostic only)")
    ax.set_title("Stage-1 edge mismatch (50k deterministic sample)")
    fig.colorbar(points, ax=ax, label="Robust mismatch score")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "h1_edge_mismatch.png", dpi=180)
    plt.close(fig)

    edge_fractions = cutoffs / len(edge_frame)
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.plot(edge_fractions, validation_curve, marker="o", label="Discovery-ranked edges")
    uniform_mean = np.mean(uniform_controls, axis=0)
    matched_mean = np.mean(matched_controls, axis=0)
    ax.plot(edge_fractions, uniform_mean, linestyle="--", label="Uniform random mean")
    ax.fill_between(
        edge_fractions,
        np.percentile(uniform_controls, 5, axis=0),
        np.percentile(uniform_controls, 95, axis=0),
        alpha=0.15,
    )
    ax.plot(edge_fractions, matched_mean, linestyle="--", label="Matched random mean")
    ax.fill_between(
        edge_fractions,
        np.percentile(matched_controls, 5, axis=0),
        np.percentile(matched_controls, 95, axis=0),
        alpha=0.15,
    )
    ax.set_xscale("log")
    ax.set_xlabel("Top discovery-ranked edge fraction")
    ax.set_ylabel("Fraction of severe validation paths explained")
    ax.set_title("H1 severe-path concentration")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "h1_concentration.png", dpi=180)
    plt.close(fig)

    top_count = int(math.ceil(TOP_FRACTION_FOR_DECISION * len(edge_frame)))
    top_edges = ranked_edges[:top_count]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(positions[:, 0], positions[:, 1], s=2, alpha=0.12, color="gray")
    for edge_id in top_edges:
        u = int(edge_frame.at[int(edge_id), "u"])
        v = int(edge_frame.at[int(edge_id), "v"])
        ax.plot(
            positions[[u, v], 0],
            positions[[u, v], 1],
            color="crimson",
            alpha=0.08,
            linewidth=0.5,
        )
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Top-1% discovery-ranked edges (GT diagnostic overlay)")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "h1_top1pct_spatial_overlay.png", dpi=180)
    plt.close(fig)

    labels = ["adp", "timestamp", "tie"]
    overall = [branch_stats[label]["overall_fraction"] for label in labels]
    top = [branch_stats[label]["top1pct_fraction"] for label in labels]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(x - 0.18, overall, width=0.36, label="All Stage-1 edges")
    ax.bar(x + 0.18, top, width=0.36, label="Top-1% candidates")
    ax.set_xticks(x, labels)
    ax.set_ylabel("Edge fraction")
    ax.set_title("Wireless evidence branch attribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "h1_branch_attribution.png", dpi=180)
    plt.close(fig)


def main() -> None:
    started = time.perf_counter()
    rng = np.random.default_rng(SEED)

    gate_audit = read_json(GATE_AUDIT_PATH)
    if gate_audit["provenance_result"] != "consistent":
        raise RuntimeError("Gate 0.5 provenance result is not consistent")

    directed_graph = sp.load_npz(GRAPH_PATH)
    if not sp.isspmatrix_csr(directed_graph):
        raise TypeError("Stage-1 graph must remain directed CSR")
    adp = np.load(ADP_PATH, mmap_mode="r")
    d_geo = np.load(D_GEO_PATH, mmap_mode="r")
    timestamps = np.load(TIMESTAMP_PATH, mmap_mode="r")
    distance_data = np.load(DISTANCE_DATA_PATH)
    positions = distance_data["positions"].astype(np.float32, copy=False)
    gamma = float(read_json(BASELINE_RESULT_PATH)["gamma"])
    n = directed_graph.shape[0]
    if not (
        directed_graph.shape == (n, n)
        and adp.shape == (n, n)
        and d_geo.shape == (n, n)
        and positions.shape == (n, 2)
        and timestamps.shape == (n,)
    ):
        raise ValueError("Stage-1 graph and input artifact shapes do not align")

    timing: dict[str, float] = {}
    stage_started = time.perf_counter()
    (
        edge_u,
        edge_v,
        fused_weight,
        support_uv,
        support_vu,
        directed_weight_uv,
        directed_weight_vu,
    ) = extract_undirected_edges(directed_graph)
    edge_count = len(edge_u)
    edge_keys = edge_u.astype(np.int64) * n + edge_v.astype(np.int64)
    wireless_scale, degree = incident_median_scale(n, edge_u, edge_v, fused_weight)
    physical_scale, physical_scale_fallback_count = physical_local_scale(positions, PHYSICAL_Q)
    timing["edge_and_local_scale"] = time.perf_counter() - stage_started

    physical_distance = np.linalg.norm(positions[edge_u] - positions[edge_v], axis=1).astype(np.float32)
    wireless_normalized = fused_weight / np.sqrt(wireless_scale[edge_u] * wireless_scale[edge_v])
    physical_normalized = physical_distance / np.sqrt(physical_scale[edge_u] * physical_scale[edge_v])
    mismatch_log_ratio, mismatch_score, mismatch_center, mismatch_robust_scale = robust_mismatch_scores(
        wireless_normalized,
        physical_normalized,
    )

    d_adp_uv = np.asarray(adp[edge_u, edge_v], dtype=np.float32)
    d_adp_vu = np.asarray(adp[edge_v, edge_u], dtype=np.float32)
    gamma_abs_dt = (
        np.abs(timestamps[edge_u] - timestamps[edge_v]) * gamma
    ).astype(np.float32)
    expected_uv = np.minimum(d_adp_uv, gamma_abs_dt).astype(np.float32)
    expected_vu = np.minimum(d_adp_vu, gamma_abs_dt).astype(np.float32)
    if not np.array_equal(directed_weight_uv[support_uv], expected_uv[support_uv]):
        raise RuntimeError("u-to-v Stage-1 weights do not equal saved fused values")
    if not np.array_equal(directed_weight_vu[support_vu], expected_vu[support_vu]):
        raise RuntimeError("v-to-u Stage-1 weights do not equal saved fused values")
    effective_from_uv = directed_weight_uv <= directed_weight_vu
    d_adp = np.where(effective_from_uv, d_adp_uv, d_adp_vu).astype(np.float32)
    if not np.array_equal(fused_weight, np.minimum(d_adp, gamma_abs_dt).astype(np.float32)):
        raise RuntimeError("effective undirected weights do not match directed=False semantics")
    branch = np.full(edge_count, "tie", dtype="<U9")
    branch[d_adp < gamma_abs_dt] = "adp"
    branch[gamma_abs_dt < d_adp] = "timestamp"

    boundary_tie_node = np.zeros(n, dtype=bool)
    boundary_tie_node[
        np.asarray(gate_audit["boundary_ties"]["boundary_tie_row_ids"], dtype=np.int32)
    ] = True
    boundary_tie_u = boundary_tie_node[edge_u]
    boundary_tie_v = boundary_tie_node[edge_v]
    boundary_ambiguous = boundary_tie_u | boundary_tie_v

    stage_started = time.perf_counter()
    discovery_sources, validation_sources, density_quartile, density_boundaries = stratified_source_split(
        physical_scale,
        rng,
    )
    used_pair_keys: set[int] = set()
    discovery_targets = sample_targets(
        discovery_sources,
        TARGETS_PER_SOURCE,
        n,
        rng,
        used_pair_keys,
        positions,
        d_geo,
    )
    validation_targets = sample_targets(
        validation_sources,
        TARGETS_PER_SOURCE,
        n,
        rng,
        used_pair_keys,
        positions,
        d_geo,
    )
    calibration_left, calibration_right = sample_calibration_pairs(
        CALIBRATION_PAIR_COUNT,
        n,
        rng,
        used_pair_keys,
        positions,
        d_geo,
    )
    timing["deterministic_pair_split"] = time.perf_counter() - stage_started

    stage_started = time.perf_counter()
    calibration_physical, calibration_d_geo = pair_distances(
        calibration_left,
        calibration_right,
        positions,
        d_geo,
    )
    calibration, calibration_predicted, calibration_residual, severe_threshold = fit_calibration(
        calibration_physical,
        calibration_d_geo,
    )
    discovery_left = np.repeat(discovery_sources, TARGETS_PER_SOURCE)
    discovery_right = discovery_targets.reshape(-1)
    validation_left = np.repeat(validation_sources, TARGETS_PER_SOURCE)
    validation_right = validation_targets.reshape(-1)
    discovery_physical, discovery_d_geo = pair_distances(
        discovery_left,
        discovery_right,
        positions,
        d_geo,
    )
    validation_physical, validation_d_geo = pair_distances(
        validation_left,
        validation_right,
        positions,
        d_geo,
    )
    discovery_predicted, discovery_residual, discovery_severe = calibrated_pair_values(
        discovery_physical,
        discovery_d_geo,
        calibration,
        severe_threshold,
    )
    validation_predicted, validation_residual, validation_severe = calibrated_pair_values(
        validation_physical,
        validation_d_geo,
        calibration,
        severe_threshold,
    )
    timing["isotonic_calibration"] = time.perf_counter() - stage_started

    stage_started = time.perf_counter()
    (
        discovery_flat_edges,
        discovery_offsets,
        discovery_hops,
        discovery_usage,
        discovery_severe_usage,
        discovery_reconstructed_distance,
    ) = trace_sampled_paths(
        directed_graph,
        discovery_sources,
        discovery_targets,
        discovery_severe,
        edge_keys,
    )
    if not np.array_equal(discovery_reconstructed_distance.reshape(-1), discovery_d_geo):
        raise RuntimeError("discovery sampled shortest paths differ from saved D_geo")
    (
        validation_flat_edges,
        validation_offsets,
        validation_hops,
        _,
        _,
        validation_reconstructed_distance,
    ) = trace_sampled_paths(
        directed_graph,
        validation_sources,
        validation_targets,
        validation_severe,
        edge_keys,
    )
    if not np.array_equal(validation_reconstructed_distance.reshape(-1), validation_d_geo):
        raise RuntimeError("validation sampled shortest paths differ from saved D_geo")
    timing["sampled_shortest_paths_and_tracing"] = time.perf_counter() - stage_started

    global_discovery_severe_fraction = float(np.mean(discovery_severe))
    bad_path_enrichment = np.full(edge_count, np.nan, dtype=np.float32)
    used = discovery_usage > 0
    bad_path_enrichment[used] = (
        discovery_severe_usage[used] / discovery_usage[used]
    ) / global_discovery_severe_fraction

    mismatch_percentile = rankdata(mismatch_score, method="average") / edge_count
    severe_usage_percentile = np.zeros(edge_count, dtype=np.float64)
    positive_impact = discovery_severe_usage > 0
    severe_usage_percentile[positive_impact] = rankdata(
        discovery_severe_usage[positive_impact],
        method="average",
    ) / np.count_nonzero(positive_impact)
    candidate_score = mismatch_percentile * severe_usage_percentile
    ranked_edges = np.argsort(-candidate_score, kind="stable").astype(np.int32)
    discovery_rank = np.empty(edge_count, dtype=np.int32)
    discovery_rank[ranked_edges] = np.arange(1, edge_count + 1, dtype=np.int32)
    top1_count = int(math.ceil(TOP_FRACTION_FOR_DECISION * edge_count))
    top1_mask = discovery_rank <= top1_count

    fixed_counts = [25, 50, 100, 250, 500, 1000]
    fixed_fractions = [0.001, 0.0025, 0.005, 0.01, 0.02, 0.05]
    cutoffs = np.asarray(
        sorted(
            {
                *fixed_counts,
                *(int(math.ceil(fraction * edge_count)) for fraction in fixed_fractions),
            }
        ),
        dtype=np.int32,
    )
    cutoffs = cutoffs[cutoffs <= edge_count]

    discovery_curve = sequence_concentration(
        discovery_flat_edges,
        discovery_offsets,
        discovery_severe,
        ranked_edges,
        cutoffs,
        edge_count,
    )
    validation_curve = sequence_concentration(
        validation_flat_edges,
        validation_offsets,
        validation_severe,
        ranked_edges,
        cutoffs,
        edge_count,
    )

    weight_bins, weight_boundaries = quantile_bins(fused_weight, 10)
    max_endpoint_degree = np.maximum(degree[edge_u], degree[edge_v])
    degree_bins, degree_boundaries = quantile_bins(max_endpoint_degree, 4)
    edge_strata = weight_bins * 4 + degree_bins
    max_cutoff = int(np.max(cutoffs))
    uniform_controls = np.empty((CONTROL_REPLICATES, len(cutoffs)), dtype=np.float64)
    matched_controls = np.empty_like(uniform_controls)
    for replicate in range(CONTROL_REPLICATES):
        uniform_sequence = rng.choice(edge_count, size=max_cutoff, replace=False).astype(np.int32)
        matched_sequence = matched_random_sequence(
            ranked_edges,
            edge_strata,
            max_cutoff,
            rng,
        )
        uniform_controls[replicate] = sequence_concentration(
            validation_flat_edges,
            validation_offsets,
            validation_severe,
            uniform_sequence,
            cutoffs,
            edge_count,
        )
        matched_controls[replicate] = sequence_concentration(
            validation_flat_edges,
            validation_offsets,
            validation_severe,
            matched_sequence,
            cutoffs,
            edge_count,
        )
    timing["ranking_and_concentration"] = time.perf_counter() - stage_started

    top1_index = int(np.flatnonzero(cutoffs == top1_count)[0])
    top1_validation_coverage = float(validation_curve[top1_index])
    top1_matched_mean = float(np.mean(matched_controls[:, top1_index]))
    top1_matched_p95 = float(np.percentile(matched_controls[:, top1_index], 95))
    top1_mismatch_median = float(np.median(mismatch_score[top1_mask]))

    nonambiguous_top1 = top1_mask & ~boundary_ambiguous
    nonambiguous_hits = edge_set_path_hits(
        validation_flat_edges,
        validation_offsets,
        nonambiguous_top1,
    )
    nonambiguous_coverage = float(np.mean(nonambiguous_hits[validation_severe]))
    retained_coverage_fraction = (
        nonambiguous_coverage / top1_validation_coverage
        if top1_validation_coverage > 0
        else 0.0
    )

    decision_checks = {
        "validation_top1pct_coverage_at_least_0p20": top1_validation_coverage >= 0.20,
        "validation_top1pct_lift_at_least_2x_matched_mean": (
            top1_validation_coverage >= 2.0 * top1_matched_mean
        ),
        "validation_top1pct_exceeds_matched_p95": top1_validation_coverage > top1_matched_p95,
        "top1pct_median_robust_mismatch_at_least_2": top1_mismatch_median >= 2.0,
        "tie_excluded_coverage_retains_at_least_80pct": retained_coverage_fraction >= 0.80,
    }
    proceed_to_counterfactual = bool(all(decision_checks.values()))
    branch_stats = branch_comparison(branch, top1_mask)

    edge_frame = pd.DataFrame(
        {
            "edge_id": np.arange(edge_count, dtype=np.int32),
            "u": edge_u,
            "v": edge_v,
            "directed_support_u_to_v": support_uv,
            "directed_support_v_to_u": support_vu,
            "directed_weight_u_to_v": directed_weight_uv,
            "directed_weight_v_to_u": directed_weight_vu,
            "effective_weight_from_u_to_v": effective_from_uv,
            "fused_distance": fused_weight,
            "physical_distance": physical_distance,
            "degree_u": degree[edge_u],
            "degree_v": degree[edge_v],
            "wireless_scale_u": wireless_scale[edge_u],
            "wireless_scale_v": wireless_scale[edge_v],
            "wireless_normalized": wireless_normalized.astype(np.float32),
            "physical_scale_u": physical_scale[edge_u],
            "physical_scale_v": physical_scale[edge_v],
            "physical_normalized": physical_normalized.astype(np.float32),
            "mismatch_log_ratio": mismatch_log_ratio.astype(np.float32),
            "robust_mismatch_score": mismatch_score.astype(np.float32),
            "d_adp": d_adp,
            "d_adp_u_to_v": d_adp_uv,
            "d_adp_v_to_u": d_adp_vu,
            "gamma_abs_dt": gamma_abs_dt,
            "branch": branch,
            "boundary_tie_u": boundary_tie_u,
            "boundary_tie_v": boundary_tie_v,
            "boundary_tie_ambiguity": boundary_ambiguous,
            "discovery_path_usage": discovery_usage,
            "discovery_severe_path_usage": discovery_severe_usage,
            "discovery_bad_path_enrichment": bad_path_enrichment,
            "mismatch_percentile": mismatch_percentile.astype(np.float32),
            "severe_usage_percentile": severe_usage_percentile.astype(np.float32),
            "candidate_score": candidate_score.astype(np.float32),
            "discovery_rank": discovery_rank,
            "candidate_top1pct": top1_mask,
        }
    )

    top1_hits_discovery = edge_set_path_hits(
        discovery_flat_edges,
        discovery_offsets,
        top1_mask,
    )
    top1_hits_validation = edge_set_path_hits(
        validation_flat_edges,
        validation_offsets,
        top1_mask,
    )
    pair_frame = pd.concat(
        [
            pd.DataFrame(
                {
                    "split": "calibration",
                    "source": calibration_left,
                    "target": calibration_right,
                    "physical_distance": calibration_physical,
                    "d_geo": calibration_d_geo,
                    "predicted_log_d_geo": calibration_predicted.astype(np.float32),
                    "r_geo": calibration_residual.astype(np.float32),
                    "severe_underestimation": calibration_residual <= severe_threshold,
                    "path_hops": np.full(CALIBRATION_PAIR_COUNT, -1, dtype=np.int32),
                    "traverses_top1pct_candidate": False,
                }
            ),
            pd.DataFrame(
                {
                    "split": "discovery",
                    "source": discovery_left,
                    "target": discovery_right,
                    "physical_distance": discovery_physical,
                    "d_geo": discovery_d_geo,
                    "predicted_log_d_geo": discovery_predicted.astype(np.float32),
                    "r_geo": discovery_residual.astype(np.float32),
                    "severe_underestimation": discovery_severe,
                    "path_hops": discovery_hops,
                    "traverses_top1pct_candidate": top1_hits_discovery,
                }
            ),
            pd.DataFrame(
                {
                    "split": "validation",
                    "source": validation_left,
                    "target": validation_right,
                    "physical_distance": validation_physical,
                    "d_geo": validation_d_geo,
                    "predicted_log_d_geo": validation_predicted.astype(np.float32),
                    "r_geo": validation_residual.astype(np.float32),
                    "severe_underestimation": validation_severe,
                    "path_hops": validation_hops,
                    "traverses_top1pct_candidate": top1_hits_validation,
                }
            ),
        ],
        ignore_index=True,
    )

    config = {
        "scope": "H1 first stage only; no edge removal and no H2",
        "seed": SEED,
        "N": n,
        "stage1_graph": str(GRAPH_PATH),
        "stage1_shortest_path_directed": False,
        "physical_positions": str(DISTANCE_DATA_PATH),
        "physical_dimensions": 2,
        "physical_q": PHYSICAL_Q,
        "wireless_node_scale": "median incident undirected Stage-1 edge weight",
        "wireless_normalization": "w_uv / sqrt(s_w[u] * s_w[v])",
        "physical_node_scale": (
            "median distance to q physical 2D nearest neighbors excluding self; "
            "if non-positive, median of strictly-positive distances in the same q-neighbor set"
        ),
        "physical_scale_zero_median_fallback_count": physical_scale_fallback_count,
        "physical_normalization": "d_phys_uv / sqrt(s_p[u] * s_p[v])",
        "robust_mismatch": "robust z-score of log(physical_normalized / wireless_normalized)",
        "calibration_pairs": CALIBRATION_PAIR_COUNT,
        "discovery_sources": DISCOVERY_SOURCE_COUNT,
        "validation_sources": VALIDATION_SOURCE_COUNT,
        "targets_per_source": TARGETS_PER_SOURCE,
        "source_stratification": "32 discovery and 32 validation sources per physical-scale quartile",
        "pair_disjointness": "unordered pair keys are disjoint across calibration, discovery, validation",
        "calibration_model": "IsotonicRegression(increasing=True, out_of_bounds='clip')",
        "severe_threshold": "calibration residual 5th percentile",
        "candidate_score": "mismatch_percentile * positive-severe-usage percentile; discovery only",
        "top1pct_edge_count": top1_count,
        "control_replicates": CONTROL_REPLICATES,
        "matched_control": "fused-weight decile x max-endpoint-degree quartile",
        "decision_rule": {
            "validation_top1pct_coverage": ">= 0.20",
            "matched_mean_lift": ">= 2.0",
            "matched_p95": "strictly exceeded",
            "top1pct_median_robust_mismatch": ">= 2.0",
            "tie_excluded_coverage_retention": ">= 0.80",
        },
        "discovery_source_ids": discovery_sources.astype(int).tolist(),
        "validation_source_ids": validation_sources.astype(int).tolist(),
        "density_quartile_boundaries": density_boundaries.astype(float).tolist(),
        "package_versions": package_versions(),
    }

    top_edge_ids = ranked_edges[:20]
    concentration_rows = []
    uniform_summary = control_summary(uniform_controls)
    matched_summary = control_summary(matched_controls)
    for index, cutoff in enumerate(cutoffs):
        concentration_rows.append(
            {
                "edge_count": int(cutoff),
                "edge_fraction": float(cutoff / edge_count),
                "discovery_severe_path_coverage": float(discovery_curve[index]),
                "validation_severe_path_coverage": float(validation_curve[index]),
                "uniform_random_mean": uniform_summary["mean"][index],
                "uniform_random_p05": uniform_summary["p05"][index],
                "uniform_random_p95": uniform_summary["p95"][index],
                "matched_random_mean": matched_summary["mean"][index],
                "matched_random_p05": matched_summary["p05"][index],
                "matched_random_p95": matched_summary["p95"][index],
            }
        )

    summary = {
        "scope": "H1 first stage only; no targeted removal and no H2",
        "H1_supported": None,
        "edge_count": edge_count,
        "directed_graph_nnz": int(directed_graph.nnz),
        "undirected_degree": distribution_summary(degree),
        "wireless_scale": distribution_summary(wireless_scale),
        "physical_scale_q16": distribution_summary(physical_scale),
        "physical_scale_zero_median_fallback_count": physical_scale_fallback_count,
        "physical_edge_distance": distribution_summary(physical_distance),
        "wireless_normalized_edge_length": distribution_summary(wireless_normalized),
        "physical_normalized_edge_length": distribution_summary(physical_normalized),
        "robust_mismatch": {
            "log_ratio_center": mismatch_center,
            "robust_scale_1p4826_mad": mismatch_robust_scale,
            "score_distribution": distribution_summary(mismatch_score),
        },
        "boundary_tie_ambiguity": {
            "all_edge_count": int(np.count_nonzero(boundary_ambiguous)),
            "all_edge_fraction": float(np.mean(boundary_ambiguous)),
            "top1pct_edge_count": int(np.count_nonzero(boundary_ambiguous & top1_mask)),
            "top1pct_edge_fraction": float(np.mean(boundary_ambiguous[top1_mask])),
            "top1pct_validation_severe_coverage": top1_validation_coverage,
            "tie_excluded_top1pct_validation_severe_coverage": nonambiguous_coverage,
            "retained_coverage_fraction": retained_coverage_fraction,
        },
        "split": {
            "calibration_pair_count": CALIBRATION_PAIR_COUNT,
            "discovery_source_count": len(discovery_sources),
            "discovery_pair_count": len(discovery_left),
            "validation_source_count": len(validation_sources),
            "validation_pair_count": len(validation_left),
            "pair_sets_disjoint": True,
            "source_sets_disjoint": bool(
                len(np.intersect1d(discovery_sources, validation_sources)) == 0
            ),
            "discovery_sources_per_density_quartile": np.bincount(
                density_quartile[discovery_sources], minlength=4
            ).astype(int).tolist(),
            "validation_sources_per_density_quartile": np.bincount(
                density_quartile[validation_sources], minlength=4
            ).astype(int).tolist(),
        },
        "calibration": {
            "model": "isotonic log(D_geo) on log(d_phys)",
            "threshold_quantile": SEVERE_CALIBRATION_QUANTILE,
            "severe_residual_threshold": severe_threshold,
            "residual_distribution": distribution_summary(calibration_residual),
            "isotonic_threshold_count": int(len(calibration.X_thresholds_)),
        },
        "pair_residuals": {
            "discovery": {
                "distribution": distribution_summary(discovery_residual),
                "severe_count": int(np.count_nonzero(discovery_severe)),
                "severe_fraction": float(np.mean(discovery_severe)),
            },
            "validation": {
                "distribution": distribution_summary(validation_residual),
                "severe_count": int(np.count_nonzero(validation_severe)),
                "severe_fraction": float(np.mean(validation_severe)),
            },
        },
        "path_impact": {
            "discovery_path_hops": distribution_summary(discovery_hops),
            "validation_path_hops": distribution_summary(validation_hops),
            "edges_used_by_any_discovery_path": int(np.count_nonzero(discovery_usage)),
            "edges_used_by_any_severe_discovery_path": int(np.count_nonzero(discovery_severe_usage)),
            "top20_edges": edge_frame.loc[
                top_edge_ids,
                [
                    "edge_id",
                    "u",
                    "v",
                    "robust_mismatch_score",
                    "discovery_path_usage",
                    "discovery_severe_path_usage",
                    "discovery_bad_path_enrichment",
                    "branch",
                    "boundary_tie_ambiguity",
                ],
            ].to_dict(orient="records"),
        },
        "concentration": concentration_rows,
        "wireless_branch_attribution": branch_stats,
        "decision_top1pct": {
            "edge_count": top1_count,
            "validation_severe_path_coverage": top1_validation_coverage,
            "matched_random_mean": top1_matched_mean,
            "matched_random_p95": top1_matched_p95,
            "lift_vs_matched_mean": (
                top1_validation_coverage / top1_matched_mean
                if top1_matched_mean > 0
                else None
            ),
            "median_robust_mismatch_score": top1_mismatch_median,
            "tie_excluded_coverage_retention": retained_coverage_fraction,
            "checks": decision_checks,
        },
        "proceed_to_H1_counterfactual": proceed_to_counterfactual,
        "timing_seconds": timing,
        "package_versions": package_versions(),
    }

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    edge_frame.to_parquet(RESULT_DIR / "h1_edge_table.parquet", index=False)
    pair_frame.to_parquet(RESULT_DIR / "h1_pair_table.parquet", index=False)
    np.savez(
        RESULT_DIR / "h1_calibration_curve.npz",
        X_thresholds=calibration.X_thresholds_,
        y_thresholds=calibration.y_thresholds_,
    )
    pd.DataFrame(concentration_rows).to_csv(
        RESULT_DIR / "h1_concentration.csv",
        index=False,
    )
    with (RESULT_DIR / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, sort_keys=True)
        handle.write("\n")
    with (RESULT_DIR / "h1_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")

    save_figures(
        edge_frame,
        positions,
        ranked_edges,
        cutoffs,
        validation_curve,
        uniform_controls,
        matched_controls,
        branch_stats,
    )

    report = f"""# Phase 2C H1 First-Stage Diagnostic

## Scope

This run covers edge anomaly, calibrated geodesic error, sampled path impact,
concentration, and wireless-branch attribution only. It does not perform edge
removal, H2, or method development. GT positions are used for offline diagnosis only.

## Frozen configuration

- Seed: `{SEED}`
- Stage-1 graph: verified directed CSR, shortest paths with `directed=False`
- Undirected edges: `{edge_count}`
- Physical local scale: q=`{PHYSICAL_Q}` median physical-neighbor distance
- Calibration pairs: `{CALIBRATION_PAIR_COUNT}`
- Discovery: `{DISCOVERY_SOURCE_COUNT}` sources x `{TARGETS_PER_SOURCE}` targets
- Validation: `{VALIDATION_SOURCE_COUNT}` sources x `{TARGETS_PER_SOURCE}` targets
- Severe threshold: calibration residual 5th percentile = `{severe_threshold:.6g}`
- Random controls: `{CONTROL_REPLICATES}` uniform + `{CONTROL_REPLICATES}` matched

## Main observations

- Discovery severe pairs: `{int(np.count_nonzero(discovery_severe))}` / `{len(discovery_severe)}`
  (`{float(np.mean(discovery_severe)):.4%}`).
- Validation severe pairs: `{int(np.count_nonzero(validation_severe))}` / `{len(validation_severe)}`
  (`{float(np.mean(validation_severe)):.4%}`).
- Top-1% discovery-ranked edges explain `{top1_validation_coverage:.4%}` of severe
  validation paths.
- Matched-random top-1% mean / P95: `{top1_matched_mean:.4%}` / `{top1_matched_p95:.4%}`.
- Lift over matched-random mean: `{top1_validation_coverage / top1_matched_mean if top1_matched_mean else float('nan'):.3f}x`.
- Top-1% median robust mismatch score: `{top1_mismatch_median:.3f}`.
- Removing boundary-tie-ambiguous candidates from the top-1% set retains
  `{retained_coverage_fraction:.4%}` of its severe validation-path coverage.

## Decision

`proceed_to_H1_counterfactual = {str(proceed_to_counterfactual).lower()}`

Checks: `{json.dumps(decision_checks, sort_keys=True)}`

This is not an `H1_supported` decision. Full H1 support would still require the
pre-specified targeted-removal counterfactual and sensitivity checks.
"""
    (RESULT_DIR / "REPORT.md").write_text(report, encoding="utf-8")

    summary["timing_seconds"]["total"] = time.perf_counter() - started
    with (RESULT_DIR / "h1_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(json.dumps({
        "proceed_to_H1_counterfactual": proceed_to_counterfactual,
        "decision_top1pct": summary["decision_top1pct"],
        "discovery_severe_count": int(np.count_nonzero(discovery_severe)),
        "validation_severe_count": int(np.count_nonzero(validation_severe)),
        "edge_count": edge_count,
        "output": str(RESULT_DIR / "h1_summary.json"),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
