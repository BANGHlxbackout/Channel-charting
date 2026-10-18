"""Input, graph, and pair utilities for the G1-ACF experiment."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from scipy import sparse
from scipy.sparse.csgraph import dijkstra
from sklearn.neighbors import NearestNeighbors, kneighbors_graph

from run_adp_siamese import AdpFeatureLayer


def file_identity(path: Path) -> dict:
    stat = path.stat()
    return {"path": str(path.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def index_digest(indices: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(indices, dtype=np.int64).tobytes()).hexdigest()


def check_distance_run(distance_run: Path, data_path: Path, indices: np.ndarray,
                       stride: int, seed: int) -> tuple[np.ndarray, np.ndarray, int, dict]:
    saved = np.load(distance_run / "distance_data.npz")
    if not np.array_equal(saved["indices"], indices):
        raise ValueError("distance-run indices differ from selected H rows")
    metadata = json.loads((distance_run / "result.json").read_text(encoding="utf-8"))
    for name, expected in (("num_samples", len(indices)), ("stride", stride), ("seed", seed)):
        if metadata[name] != expected:
            raise ValueError(f"distance-run {name} mismatch: {metadata[name]} != {expected}")
    stored_path = Path(metadata["dataset"])
    if stored_path.is_absolute() and stored_path.resolve() != data_path.resolve():
        raise ValueError("distance-run data path mismatch")
    fused = np.load(distance_run / "fused.npy", mmap_mode="r")
    geo = np.load(distance_run / "geodesic.npy", mmap_mode="r")
    n = len(indices)
    if fused.shape != (n, n) or geo.shape != (n, n):
        raise ValueError("fused/geodesic shape mismatch")
    return fused, geo, int(metadata["neighbors"]), metadata


def rebuild_directed_graph(fused: np.ndarray, neighbors: int) -> sparse.csr_matrix:
    nearest = NearestNeighbors(n_neighbors=neighbors, metric="precomputed", n_jobs=-1).fit(fused)
    graph = kneighbors_graph(nearest, neighbors, metric="precomputed", mode="distance", n_jobs=-1).tocsr()
    n = graph.shape[0]
    if np.any(np.diff(graph.indptr) != neighbors) or np.any(graph.diagonal() != 0):
        # Explicit zero-distance self edges would also fail the row/column test below.
        raise ValueError("directed kNN degree or diagonal differs from baseline")
    row = np.repeat(np.arange(n), np.diff(graph.indptr))
    if np.any(row == graph.indices) or not np.all(np.isfinite(graph.data)):
        raise ValueError("directed kNN has self edges or nonfinite costs")
    return graph


def audit_shortest_paths(graph: sparse.csr_matrix, geo: np.ndarray,
                         n_sources: int = 32, atol: float = 1e-4) -> dict:
    n = graph.shape[0]
    sources = np.unique(np.linspace(0, n - 1, min(n_sources, n), dtype=np.int64))
    observed = dijkstra(graph, directed=False, indices=sources).astype(np.float32)
    target = np.asarray(geo[sources])
    diff = np.abs(observed - target)
    mismatch = int(np.count_nonzero(~np.isclose(observed, target, atol=atol, rtol=1e-5)))
    result = {"source_ids": sources.tolist(), "max_abs_diff": float(np.max(diff)),
              "mismatch_count": mismatch, "atol": atol, "directed_entries": int(graph.nnz)}
    if mismatch:
        raise ValueError(f"reconstructed graph disagrees with saved geodesic: {result}")
    return result


def propagation_adjacency(graph: sparse.csr_matrix, fused: np.ndarray) -> tuple[torch.Tensor, dict]:
    """Symmetric GCN affinity on the baseline directed graph's undirected support.

    The baseline's directed costs remain untouched. A canonical upper-triangle
    fused cost is used only for the new symmetric propagation affinity.
    """
    n = graph.shape[0]
    row = np.repeat(np.arange(n, dtype=np.int64), np.diff(graph.indptr))
    col = graph.indices.astype(np.int64, copy=False)
    lo, hi = np.minimum(row, col), np.maximum(row, col)
    pairs = np.unique(lo * n + hi)
    left, right = pairs // n, pairs % n
    rho = np.maximum.reduceat(graph.data, graph.indptr[:-1]).astype(np.float64)
    cost = np.asarray(fused[left, right], dtype=np.float64)
    denom = np.maximum(rho[left], 1e-6) * np.maximum(rho[right], 1e-6)
    weight = np.exp(-(cost * cost) / denom).astype(np.float32)
    rows = np.concatenate((left, right, np.arange(n)))
    cols = np.concatenate((right, left, np.arange(n)))
    values = np.concatenate((weight, weight, np.ones(n, dtype=np.float32)))
    affinity = sparse.coo_matrix((values, (rows, cols)), shape=(n, n)).tocsr()
    degree = np.asarray(affinity.sum(axis=1)).ravel()
    inv = 1.0 / np.sqrt(degree)
    edge_norm = weight * inv[left] * inv[right]
    norm_values = np.concatenate((edge_norm, edge_norm, inv * inv))
    normalized = sparse.coo_matrix((norm_values, (rows, cols)), shape=(n, n))
    indices = torch.from_numpy(np.stack((normalized.row, normalized.col)).astype(np.int64))
    values_t = torch.from_numpy(normalized.data.astype(np.float32))
    adj = torch.sparse_coo_tensor(indices, values_t, (n, n)).coalesce()
    if not np.all(np.isfinite(normalized.data)) or np.any(normalized.data < 0):
        raise ValueError("nonfinite or negative GCN affinity")
    if (normalized.tocsr() - normalized.T.tocsr()).nnz:
        raise ValueError("GCN affinity is not symmetric")
    reverse = graph.multiply(graph.T != 0)
    reciprocal = sparse.triu(reverse, k=1).tocoo()
    asym = np.abs(np.asarray(fused[reciprocal.row, reciprocal.col]) -
                  np.asarray(fused[reciprocal.col, reciprocal.row]))
    info = {"undirected_edges": int(len(pairs)),
            "reciprocal_asymmetric_pairs": int(np.count_nonzero(asym)),
            "reciprocal_max_abs_diff": float(np.max(asym)) if len(asym) else 0.0,
            "affinity_min": float(weight.min()), "affinity_max": float(weight.max())}
    return adj, info


def identity_adjacency(n: int) -> torch.Tensor:
    axis = torch.arange(n, dtype=torch.int64)
    return torch.sparse_coo_tensor(torch.stack((axis, axis)), torch.ones(n), (n, n)).coalesce()


def feature_cache(data_path: Path, indices: np.ndarray, h: np.ndarray,
                  cache_root: Path, chunk_size: int, device: torch.device) -> tuple[np.ndarray, dict]:
    cache_root.mkdir(parents=True, exist_ok=True)
    name = f"acf_n{len(indices)}_{index_digest(indices)[:12]}"
    path = cache_root / f"{name}.npy"
    meta_path = cache_root / f"{name}.json"
    expected = {"h_file": file_identity(data_path / "H.npy"), "index_digest": index_digest(indices),
                "shape": [len(indices), 26624], "dtype": "float32"}
    if path.exists() and meta_path.exists():
        actual = json.loads(meta_path.read_text(encoding="utf-8"))
        if actual != expected:
            raise ValueError("existing feature cache identity mismatch")
        feat = np.load(path, mmap_mode="r")
        if feat.shape != tuple(expected["shape"]) or feat.dtype != np.float32:
            raise ValueError("existing feature cache shape/dtype mismatch")
    else:
        feat = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32,
                                         shape=(len(indices), 26624))
        layer = AdpFeatureLayer().eval().to(device)
        with torch.no_grad():
            for start in range(0, len(indices), chunk_size):
                stop = min(start + chunk_size, len(indices))
                batch = torch.from_numpy(np.ascontiguousarray(h[start:stop])).to(device)
                feat[start:stop] = layer(batch).float().cpu().numpy()
        feat.flush()
        meta_path.write_text(json.dumps(expected, indent=2), encoding="utf-8")
        feat = np.load(path, mmap_mode="r")
    # Spot check the actual cached rows, including the array/tap order.
    rng = np.random.default_rng(173)
    sample = rng.choice(len(indices), size=min(8, len(indices)), replace=False)
    with torch.no_grad():
        direct = AdpFeatureLayer()(torch.from_numpy(np.ascontiguousarray(h[sample]))).numpy()
    if not np.allclose(feat[sample], direct, atol=1e-5, rtol=1e-5):
        raise ValueError("cached ACF features differ from B0 feature layer")
    return feat, {**expected, "cache_path": str(path)}


def validation_pairs(n: int, count: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    keys = np.empty(0, dtype=np.int64)
    while len(keys) < count:
        batch = max(4096, (count - len(keys)) * 2)
        a = rng.integers(0, n, size=batch, dtype=np.int64)
        b = rng.integers(0, n, size=batch, dtype=np.int64)
        valid = a != b
        candidate = np.minimum(a[valid], b[valid]) * n + np.maximum(a[valid], b[valid])
        keys = np.unique(np.concatenate((keys, candidate)))
    keys = keys[:count]
    return keys // n, keys % n, keys


def train_pairs(n: int, count: int, rng: np.random.Generator,
                excluded_keys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    left_parts, right_parts = [], []
    remaining = count
    while remaining:
        batch = max(4096, remaining * 2)
        a = rng.integers(0, n, size=batch, dtype=np.int64)
        b = rng.integers(0, n, size=batch, dtype=np.int64)
        keys = np.minimum(a, b) * n + np.maximum(a, b)
        keep = (a != b) & ~np.isin(keys, excluded_keys, assume_unique=False)
        aa, bb = a[keep][:remaining], b[keep][:remaining]
        left_parts.append(aa)
        right_parts.append(bb)
        remaining -= len(aa)
    return np.concatenate(left_parts), np.concatenate(right_parts)
