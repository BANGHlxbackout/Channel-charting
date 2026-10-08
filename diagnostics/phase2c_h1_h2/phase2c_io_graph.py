from __future__ import annotations

import json
import platform
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components, shortest_path
from sklearn.neighbors import NearestNeighbors, kneighbors_graph


REPO_ROOT = Path(__file__).resolve().parents[2]
FUSED_PATH = REPO_ROOT / "results/adp_old_fused/stride1_n20997_seed10/fused.npy"
SAVED_GEODESIC_PATH = REPO_ROOT / "results/adp_old_fused/stride1_n20997_seed10/geodesic.npy"
OUTPUT_DIR = REPO_ROOT / "results/phase2c_h1_h2"
AUDIT_PATH = OUTPUT_DIR / "audit.json"
GRAPH_PATH = OUTPUT_DIR / "stage1_graph.npz"

K = 20
VALIDATION_SEED = 10
VALIDATION_SOURCE_COUNT = 32
RTOL = 1e-6
ATOL = 1e-5


def reconstruct_stage1_graph(fused: np.ndarray, k: int = K) -> sp.csr_matrix:
    """Reproduce the directed weighted CSR emitted by the Stage-1 baseline."""
    neighbors = NearestNeighbors(
        n_neighbors=k,
        metric="precomputed",
        n_jobs=-1,
    ).fit(fused)
    graph = kneighbors_graph(
        neighbors,
        k,
        metric="precomputed",
        mode="distance",
        n_jobs=-1,
    )
    if not sp.isspmatrix_csr(graph):
        raise TypeError(f"expected CSR graph, got {graph.getformat()}")
    return graph


def graph_sanity_statistics(graph: sp.csr_matrix, k: int = K) -> dict[str, object]:
    if graph.shape[0] != graph.shape[1]:
        raise ValueError(f"graph must be square, got {graph.shape}")

    n = graph.shape[0]
    outgoing_degree = np.diff(graph.indptr).astype(np.int64, copy=False)
    row_ids = np.repeat(np.arange(n, dtype=graph.indices.dtype), outgoing_degree)
    self_per_row = np.bincount(
        row_ids[graph.indices == row_ids],
        minlength=n,
    )
    nonself_degree = outgoing_degree - self_per_row
    component_count, _ = connected_components(graph, directed=False)

    return {
        "shape": [int(graph.shape[0]), int(graph.shape[1])],
        "dtype": str(graph.dtype),
        "format": graph.getformat(),
        "directed_nnz": int(graph.nnz),
        "directed_asymmetry_nnz": int((graph != graph.T).nnz),
        "outgoing_degree": {
            "min": int(outgoing_degree.min()),
            "median": float(np.median(outgoing_degree)),
            "max": int(outgoing_degree.max()),
        },
        "nonself_outgoing_degree": {
            "min": int(nonself_degree.min()),
            "median": float(np.median(nonself_degree)),
            "max": int(nonself_degree.max()),
        },
        "rows_with_exactly_k_nonself_neighbors": int(np.count_nonzero(nonself_degree == k)),
        "all_rows_have_exactly_k_nonself_neighbors": bool(np.all(nonself_degree == k)),
        "self_edge_count": int(self_per_row.sum()),
        "connected_components_directed_false": int(component_count),
    }


def _distance_summary(values: np.ndarray) -> dict[str, float]:
    values64 = values.astype(np.float64, copy=False)
    return {
        "min": float(np.min(values64)),
        "p25": float(np.percentile(values64, 25)),
        "median": float(np.median(values64)),
        "p75": float(np.percentile(values64, 75)),
        "max": float(np.max(values64)),
    }


def boundary_tie_statistics(
    fused: np.ndarray,
    k: int = K,
    block_rows: int = 64,
) -> dict[str, object]:
    """Compare the kth and (k+1)th non-self distances using exact equality."""
    n = fused.shape[0]
    if fused.shape != (n, n):
        raise ValueError(f"fused matrix must be square, got {fused.shape}")
    if not 0 < k < n - 1:
        raise ValueError(f"k must satisfy 0 < k < N-1, got k={k}, N={n}")

    kth = np.empty(n, dtype=fused.dtype)
    next_kth = np.empty(n, dtype=fused.dtype)
    tie_multiplicity = np.zeros(n, dtype=np.int32)
    top_k_plus_one_has_duplicate = np.zeros(n, dtype=bool)

    for start in range(0, n, block_rows):
        stop = min(start + block_rows, n)
        block = np.array(fused[start:stop], copy=True)
        block[np.arange(stop - start), np.arange(start, stop)] = np.inf
        block.partition((k - 1, k), axis=1)

        block_kth = block[:, k - 1].copy()
        block_next = block[:, k].copy()
        kth[start:stop] = block_kth
        next_kth[start:stop] = block_next
        tie_multiplicity[start:stop] = np.count_nonzero(
            block == block_kth[:, None],
            axis=1,
        )

        nearest = np.sort(block[:, : k + 1], axis=1)
        top_k_plus_one_has_duplicate[start:stop] = np.any(
            nearest[:, 1:] == nearest[:, :-1],
            axis=1,
        )

    boundary_tie = kth == next_kth
    boundary_tie_count = int(np.count_nonzero(boundary_tie))
    duplicate_row_count = int(np.count_nonzero(top_k_plus_one_has_duplicate))
    max_tie_multiplicity = (
        int(np.max(tie_multiplicity[boundary_tie]))
        if boundary_tie_count
        else 0
    )

    return {
        "boundary_tie_definition": "20th_distance == 21st_distance",
        "boundary_tie_row_count": boundary_tie_count,
        "boundary_tie_row_fraction": float(boundary_tie_count / n),
        "boundary_tie_row_ids": np.flatnonzero(boundary_tie).astype(int).tolist(),
        "max_boundary_tie_multiplicity": max_tie_multiplicity,
        "kth_distance": _distance_summary(kth),
        "next_kth_distance": _distance_summary(next_kth),
        "top_k_plus_one_duplicate_row_count": duplicate_row_count,
        "top_k_plus_one_duplicate_row_fraction": float(duplicate_row_count / n),
    }


def sampled_geodesic_reproduction(
    graph: sp.csr_matrix,
    saved_geodesic: np.ndarray,
    seed: int = VALIDATION_SEED,
    source_count: int = VALIDATION_SOURCE_COUNT,
    rtol: float = RTOL,
    atol: float = ATOL,
) -> dict[str, object]:
    n = graph.shape[0]
    if saved_geodesic.shape != (n, n):
        raise ValueError(
            f"saved geodesic shape {saved_geodesic.shape} does not match graph {graph.shape}"
        )

    rng = np.random.default_rng(seed)
    sources = rng.choice(n, size=source_count, replace=False)
    reconstructed = shortest_path(
        graph,
        method="auto",
        directed=False,
        indices=sources,
    ).astype(np.float32)
    saved = np.asarray(saved_geodesic[sources], dtype=np.float32)

    exact = reconstructed == saved
    near_exact = np.isclose(reconstructed, saved, rtol=rtol, atol=atol)
    difference = np.abs(
        reconstructed.astype(np.float64) - saved.astype(np.float64)
    )
    relative_error = difference / np.maximum(
        np.abs(saved.astype(np.float64)),
        1e-12,
    )

    non_exact_by_source = np.count_nonzero(~exact, axis=1)
    mismatch_by_source = np.count_nonzero(~near_exact, axis=1)
    distance_count = int(exact.size)

    return {
        "validation_seed": int(seed),
        "validation_source_count": int(source_count),
        "validation_source_ids": sources.astype(int).tolist(),
        "validation_distance_count": distance_count,
        "comparison_dtype": "float32",
        "rtol": float(rtol),
        "atol": float(atol),
        "exact_match_count": int(np.count_nonzero(exact)),
        "exact_match_fraction": float(np.count_nonzero(exact) / distance_count),
        "non_exact_count": int(np.count_nonzero(~exact)),
        "non_exact_source_ids": sources[non_exact_by_source > 0].astype(int).tolist(),
        "near_exact_match_count": int(np.count_nonzero(near_exact)),
        "near_exact_match_fraction": float(np.count_nonzero(near_exact) / distance_count),
        "mismatch_count": int(np.count_nonzero(~near_exact)),
        "mismatch_source_ids": sources[mismatch_by_source > 0].astype(int).tolist(),
        "mismatch_count_by_source": {
            str(int(source)): int(count)
            for source, count in zip(sources, mismatch_by_source)
            if count
        },
        "max_absolute_difference": float(np.max(difference)),
        "median_absolute_difference": float(np.median(difference)),
        "p99_absolute_difference": float(np.percentile(difference, 99)),
        "max_relative_error": float(np.max(relative_error)),
    }


def package_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "numpy": version("numpy"),
        "scipy": version("scipy"),
        "scikit_learn": version("scikit-learn"),
    }


def save_directed_csr(path: Path, graph: sp.csr_matrix) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sp.save_npz(path, graph)
    loaded = sp.load_npz(path)
    if not sp.isspmatrix_csr(loaded):
        raise RuntimeError("saved Stage-1 graph did not reload as CSR")
    if not (
        loaded.shape == graph.shape
        and loaded.dtype == graph.dtype
        and np.array_equal(loaded.indptr, graph.indptr)
        and np.array_equal(loaded.indices, graph.indices)
        and np.array_equal(loaded.data, graph.data)
    ):
        raise RuntimeError("saved Stage-1 graph differs from direct kNN CSR output")


def main() -> None:
    started = time.perf_counter()
    fused = np.load(FUSED_PATH, mmap_mode="r")
    saved_geodesic = np.load(SAVED_GEODESIC_PATH, mmap_mode="r")
    if fused.dtype != np.float32 or fused.shape != saved_geodesic.shape:
        raise ValueError(
            f"unexpected inputs: fused={fused.shape}/{fused.dtype}, "
            f"D_geo={saved_geodesic.shape}/{saved_geodesic.dtype}"
        )
    if saved_geodesic.dtype != np.float32:
        raise ValueError(f"expected float32 saved D_geo, got {saved_geodesic.dtype}")

    stage_started = time.perf_counter()
    graph = reconstruct_stage1_graph(fused, k=K)
    graph_seconds = time.perf_counter() - stage_started
    graph_stats = graph_sanity_statistics(graph, k=K)

    stage_started = time.perf_counter()
    tie_stats = boundary_tie_statistics(fused, k=K)
    tie_seconds = time.perf_counter() - stage_started

    stage_started = time.perf_counter()
    reproduction = sampled_geodesic_reproduction(
        graph,
        saved_geodesic,
        seed=VALIDATION_SEED,
        source_count=VALIDATION_SOURCE_COUNT,
        rtol=RTOL,
        atol=ATOL,
    )
    reproduction_seconds = time.perf_counter() - stage_started

    consistent = bool(
        graph_stats["connected_components_directed_false"] == 1
        and graph_stats["all_rows_have_exactly_k_nonself_neighbors"]
        and graph_stats["self_edge_count"] == 0
        and reproduction["mismatch_count"] == 0
    )
    provenance_result = "consistent" if consistent else "inconsistent"

    audit = {
        "gate": "Phase 2C Gate 0.5 - Minimal Stage-1 Graph Provenance Sanity Check",
        "scope": "provenance only; no H1/H2 diagnostics",
        "environment_name": "diffusion_charting",
        "package_versions": package_versions(),
        "source_artifacts": {
            "fused": str(FUSED_PATH),
            "saved_D_geo": str(SAVED_GEODESIC_PATH),
        },
        "input_arrays": {
            "fused": {"shape": list(fused.shape), "dtype": str(fused.dtype)},
            "saved_D_geo": {
                "shape": list(saved_geodesic.shape),
                "dtype": str(saved_geodesic.dtype),
            },
        },
        "graph_construction": {
            "k": K,
            "estimator": "NearestNeighbors(metric='precomputed')",
            "builder": "kneighbors_graph(mode='distance')",
            "self_excluded": True,
            "edge_weight": "fused[i,j]",
            "explicit_symmetrization": False,
        },
        "graph": graph_stats,
        "boundary_ties": tie_stats,
        "reproduction": reproduction,
        "timing_seconds": {
            "graph_reconstruction": float(graph_seconds),
            "boundary_tie_check": float(tie_seconds),
            "sampled_shortest_path": float(reproduction_seconds),
            "total_before_artifact_write": float(time.perf_counter() - started),
        },
        "provenance_result": provenance_result,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if consistent:
        save_directed_csr(GRAPH_PATH, graph)
        audit["saved_stage1_graph"] = str(GRAPH_PATH)
    else:
        audit["saved_stage1_graph"] = None

    with AUDIT_PATH.open("w", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
