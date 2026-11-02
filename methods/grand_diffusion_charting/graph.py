from __future__ import annotations

from typing import Any

import numpy as np
import torch
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from sklearn.neighbors import NearestNeighbors


def _safe_scale(values: np.ndarray) -> float:
    positive = np.asarray(values)[np.asarray(values) > 1e-12]
    return float(np.median(positive)) if positive.size else 1.0


def _normalise_edge_attr(adp_d: np.ndarray, fused_d: np.ndarray,
                         time_d: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    scales = (_safe_scale(adp_d), _safe_scale(fused_d), _safe_scale(time_d))
    cols = [np.log1p(np.maximum(adp_d, 0.0) / scales[0]),
            np.log1p(np.maximum(fused_d, 0.0) / scales[1]),
            np.log1p(np.maximum(time_d, 0.0) / scales[2])]
    return np.stack(cols, axis=1).astype(np.float32), {
        "adp_scale": scales[0], "fused_scale": scales[1], "time_scale": scales[2]
    }


def build_graph(adp: np.ndarray, fused: np.ndarray, timestamps: np.ndarray,
                source: str, neighbors: int, edge_attr_mode: str,
                device: torch.device
                ) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Build a symmetric, locally scaled kNN graph and sparse operators."""
    n = int(adp.shape[0])
    if adp.shape != (n, n) or fused.shape != (n, n) or len(timestamps) != n:
        raise ValueError("ADP, fused, and timestamp dimensions do not match")
    if source not in {"adp", "fused"}:
        raise ValueError(f"unknown graph source: {source}")
    if edge_attr_mode not in {"adp", "multiview"}:
        raise ValueError(f"unknown edge attribute mode: {edge_attr_mode}")
    d_graph = adp if source == "adp" else fused
    k = min(max(2, int(neighbors)), n - 1)

    nn_index = NearestNeighbors(n_neighbors=min(k + 1, n),
                                metric="precomputed", n_jobs=-1)
    nn_index.fit(d_graph)
    distances, indices = nn_index.kneighbors(
        d_graph, n_neighbors=min(k + 1, n)
    )

    row = np.repeat(np.arange(n, dtype=np.int64), k)
    col = np.empty(n * k, dtype=np.int64)
    sigma = np.empty(n, dtype=np.float64)
    for i in range(n):
        keep = indices[i] != i
        local_idx = indices[i][keep][:k]
        local_dist = distances[i][keep][:k]
        if len(local_idx) != k:
            raise RuntimeError(f"could not obtain {k} non-self neighbours for node {i}")
        col[i * k:(i + 1) * k] = local_idx
        sigma[i] = max(float(local_dist[-1]), 1e-8)

    lo = np.minimum(row, col)
    hi = np.maximum(row, col)
    pair_codes = np.unique(lo * n + hi)
    left = (pair_codes // n).astype(np.int64)
    right = (pair_codes % n).astype(np.int64)

    graph_d = np.asarray(d_graph[left, right], dtype=np.float64)
    denom = np.maximum(sigma[left] * sigma[right], 1e-12)
    undirected_weight = np.exp(-np.square(graph_d) / denom).astype(np.float32)
    undirected_weight = np.maximum(undirected_weight, 1e-8)

    adp_edge = np.asarray(adp[left, right], dtype=np.float64)
    fused_edge = np.asarray(fused[left, right], dtype=np.float64)
    time_edge = np.abs(np.asarray(timestamps[left], dtype=np.float64) -
                       np.asarray(timestamps[right], dtype=np.float64))
    undirected_attr, attr_scales = _normalise_edge_attr(
        adp_edge, fused_edge, time_edge
    )
    if edge_attr_mode == "adp":
        # Controlled setting: the adaptive gate sees ADP edge distance only.
        undirected_attr[:, 1:] = 0.0

    src = np.concatenate((left, right))
    dst = np.concatenate((right, left))
    base_weight = np.concatenate(
        (undirected_weight, undirected_weight)
    ).astype(np.float32)
    edge_attr = np.concatenate(
        (undirected_attr, undirected_attr), axis=0
    ).astype(np.float32)

    degree = np.bincount(src, weights=base_weight, minlength=n).astype(np.float64)
    norm_weight = base_weight / np.sqrt(
        np.maximum(degree[src] * degree[dst], 1e-12)
    )
    indices_t = torch.from_numpy(np.stack((src, dst), axis=0)).long().to(device)
    values_t = torch.from_numpy(norm_weight.astype(np.float32)).to(device)
    adj = torch.sparse_coo_tensor(
        indices_t, values_t, (n, n), device=device
    ).coalesce()

    graph = {
        "adj": adj,
        "edge_index": indices_t,
        "base_weight": torch.from_numpy(base_weight).to(device),
        "edge_attr": torch.from_numpy(edge_attr).to(device),
        "num_nodes": n,
    }
    support = sparse.coo_matrix(
        (np.ones(len(src), dtype=np.float32), (src, dst)), shape=(n, n)
    ).tocsr()
    n_components, labels = connected_components(support, directed=False)
    component_sizes = np.bincount(labels, minlength=n_components)
    metadata = {
        "graph_source": source,
        "edge_attr_mode": edge_attr_mode,
        "neighbors_per_node": k,
        "num_nodes": n,
        "num_undirected_edges": int(len(left)),
        "num_directed_edges": int(len(src)),
        "num_connected_components": int(n_components),
        "largest_component_size": int(component_sizes.max()),
        "degree_min": int(np.diff(support.indptr).min()),
        "degree_median": float(np.median(np.diff(support.indptr))),
        "degree_max": int(np.diff(support.indptr).max()),
        "base_weight_min": float(undirected_weight.min()),
        "base_weight_median": float(np.median(undirected_weight)),
        "base_weight_max": float(undirected_weight.max()),
        "edge_attribute_scales": attr_scales,
    }
    return graph, metadata


def raw_csi_features(h: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Convert calibrated complex CSI to standardised real-valued node features."""
    x = np.stack((h.real, h.imag), axis=-1).reshape(len(h), -1).astype(np.float32)
    mean = x.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = x.std(axis=0, dtype=np.float64).astype(np.float32)
    std[std < 1e-6] = 1.0
    x = (x - mean) / std
    return x.astype(np.float32), {
        "feature_type": "raw_csi_real_imag",
        "input_dim": int(x.shape[1]),
        "standardisation": "per-feature across all selected samples",
    }


def estimate_geodesic_margin(geo: np.ndarray, n: int, seed: int,
                             count: int = 200000) -> float:
    """Estimate the low-distance scale from off-diagonal pairs only."""
    rng = np.random.default_rng(seed)
    left_parts: list[np.ndarray] = []
    right_parts: list[np.ndarray] = []
    remaining = count
    while remaining > 0:
        batch = max(4096, remaining * 2)
        left = rng.integers(0, n, size=batch, dtype=np.int64)
        right = rng.integers(0, n, size=batch, dtype=np.int64)
        keep = left != right
        left, right = left[keep][:remaining], right[keep][:remaining]
        left_parts.append(left)
        right_parts.append(right)
        remaining -= len(left)
    values = np.asarray(
        geo[np.concatenate(left_parts), np.concatenate(right_parts)],
        dtype=np.float32
    )
    margin = float(np.quantile(values, 0.01))
    if not np.isfinite(margin) or margin <= 0:
        positive = values[values > 0]
        margin = float(np.quantile(positive, 0.01)) if positive.size else 1e-6
    return max(margin, 1e-6)


def sample_validation_pairs(n: int, count: int, seed: int
                             ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Uniform random validation pairs; avoids sorted-unique index bias."""
    rng = np.random.default_rng(seed)
    left_parts: list[np.ndarray] = []
    right_parts: list[np.ndarray] = []
    remaining = count
    while remaining > 0:
        batch = max(4096, remaining * 2)
        left = rng.integers(0, n, size=batch, dtype=np.int64)
        right = rng.integers(0, n, size=batch, dtype=np.int64)
        keep = left != right
        left, right = left[keep][:remaining], right[keep][:remaining]
        left_parts.append(left)
        right_parts.append(right)
        remaining -= len(left)
    left = np.concatenate(left_parts)
    right = np.concatenate(right_parts)
    keys = np.minimum(left, right) * n + np.maximum(left, right)
    return left, right, np.unique(keys)
