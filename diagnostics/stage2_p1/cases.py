"""Fixed synthetic observations/graphs, independent of real UE positions."""
from __future__ import annotations

import numpy as np
from scipy import sparse


def synthetic_case(n):
    rng = np.random.default_rng(20261019 + n)
    # Irregular weighted path plus local two-hop edges; all nodes have positive degree.
    rows = np.concatenate((np.arange(n - 1), np.arange(n - 2)))
    cols = np.concatenate((np.arange(1, n), np.arange(2, n)))
    weights = np.concatenate((np.linspace(0.5, 1.5, n - 1), rng.uniform(0.15, 0.4, n - 2)))
    upper = sparse.coo_matrix((weights, (rows, cols)), shape=(n, n)).tocsr()
    adjacency = upper + upper.T
    t = np.linspace(0, 1, n)
    clean = np.stack((np.cos(1.5 * np.pi * t), np.sin(1.5 * np.pi * t)), axis=1)
    observation = clean + rng.normal(scale=0.10, size=clean.shape)
    center = observation.mean(axis=0)
    scale = np.sqrt(np.mean(np.sum((observation - center) ** 2, axis=1)))
    return adjacency.tocsr(), (observation - center) / scale, clean, observation
