from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.spatial import distance_matrix
from sklearn.manifold import trustworthiness


def continuity(reference: np.ndarray, embedding: np.ndarray, n_neighbors: int) -> float:
    return float(trustworthiness(embedding, reference, n_neighbors=n_neighbors))


def kruskal_stress(reference: np.ndarray, embedding: np.ndarray) -> float:
    dist_ref = distance_matrix(reference, reference)
    dist_emb = distance_matrix(embedding, embedding)
    beta = np.sum(dist_ref * dist_emb) / np.sum(dist_emb * dist_emb)
    return float(np.sqrt(np.sum((dist_ref - beta * dist_emb) ** 2) / np.sum(dist_ref * dist_ref)))


def affine_transform(reference: np.ndarray, embedding: np.ndarray) -> np.ndarray:
    pad = lambda x: np.hstack([x, np.ones((x.shape[0], 1), dtype=x.dtype)])
    unpad = lambda x: x[:, :-1]
    transform, *_ = np.linalg.lstsq(pad(embedding), pad(reference), rcond=None)
    return unpad(pad(embedding) @ transform)


def localization_error_summary(reference: np.ndarray, embedding: np.ndarray) -> dict[str, float]:
    aligned = affine_transform(reference, embedding)
    errors = np.linalg.norm(aligned - reference, axis=1)
    return {
        "mean_m": float(np.mean(errors)),
        "median_m": float(np.median(errors)),
        "p90_m": float(np.percentile(errors, 90)),
        "p95_m": float(np.percentile(errors, 95)),
        "max_m": float(np.max(errors)),
    }


def chart_metrics(reference: np.ndarray, embedding: np.ndarray, neighbor_fraction: float = 0.05) -> dict[str, float]:
    n_neighbors = max(1, int(len(reference) * neighbor_fraction))
    return {
        "ct": continuity(reference, embedding, n_neighbors=n_neighbors),
        "tw": float(trustworthiness(reference, embedding, n_neighbors=n_neighbors)),
        "ks": kruskal_stress(reference, embedding),
        "n_neighbors": n_neighbors,
    }


def write_json(path: str | Path, data: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
