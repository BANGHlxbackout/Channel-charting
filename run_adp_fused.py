"""ADP (Angle-Delay Profile) + timestamp fused channel charting.

Reference paper:
    "Angle-Delay Profile-Based and Timestamp-Aided Dissimilarity Metrics
     for Channel Charting"

This script implements a full channel-charting pipeline that:
  1. Loads distributed-MIMO CSI measurements (from the DICHASUS dataset).
  2. Computes an ADP-based dissimilarity matrix between every pair of samples.
  3. Estimates a temporal scaling factor γ so that timestamp distances are on
     the same scale as ADP distances.
  4. Fuses ADP and timestamp distances, builds a kNN graph, and computes
     geodesic (shortest-path) distances on that graph.
  5. Projects the geodesic distances into 2-D using classical (metric) MDS.
  6. Evaluates the chart with trustworthiness, continuity, Kruskal stress, and
     localization-error metrics.

Usage:
    python run_adp_fused.py
    python run_adp_fused.py --stride 5 --neighbors 30 --force
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

# Use the non-interactive "Agg" backend so the script can run headless
# (no display server required).  Must be set *before* importing pyplot.
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.sparse.csgraph import connected_components, shortest_path
from sklearn.manifold import MDS
from sklearn.neighbors import NearestNeighbors, kneighbors_graph

from common_dichasus import DEFAULT_DATA1, load_data1
from common_metrics import affine_transform, chart_metrics, localization_error_summary, write_json


# ---------------------------------------------------------------------------
# Data preparation helpers
# ---------------------------------------------------------------------------

def select_indices(n: int, stride: int, max_samples: int | None) -> np.ndarray:
    """Select a regular subsample of dataset indices.

    Args:
        n: Total number of samples in the dataset.
        stride: Take every *stride*-th sample (e.g. stride=10 → 1/10 of data).
        max_samples: Hard cap on the number of samples; ``None`` means no cap.

    Returns:
        Sorted integer array of selected indices.
    """
    idx = np.arange(0, n, stride, dtype=np.int64)
    if max_samples is not None:
        idx = idx[:max_samples]
    return idx


def normalize_csi_vectors(h: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Reshape & L2-normalize CSI vectors along the antenna dimension.

    The raw CSI tensor has shape ``(N, arrays, antennas_per_array, taps)``.
    We merge the *arrays × taps* dimensions so each sample becomes a set of
    column vectors (one per antenna), then L2-normalize each column.

    Args:
        h: CSI tensor of shape ``(N, A, M, T)`` where
            - N = number of samples
            - A = number of antenna arrays (4 in DICHASUS)
            - M = antennas per array (8)
            - T = delay taps (13)
        eps: Small constant to avoid division by zero.

    Returns:
        Normalized complex tensor of shape ``(N, A*T, M)``.
    """
    # Transpose to (N, A, T, M) then merge A and T → (N, A*T, M)
    z = np.transpose(h, (0, 1, 3, 2)).reshape(h.shape[0], h.shape[1] * h.shape[3], h.shape[2])
    # L2-normalize each row-vector (per antenna dimension)
    norm = np.linalg.norm(z, axis=-1, keepdims=True)
    return (z / np.maximum(norm, eps)).astype(np.complex64)


# ---------------------------------------------------------------------------
# ADP dissimilarity matrix
# ---------------------------------------------------------------------------

def compute_adp_matrix(h: np.ndarray, block_size: int, device: str) -> np.ndarray:
    """Compute the pairwise ADP dissimilarity matrix.

    For two CSI samples *i* and *j*, the ADP dissimilarity is defined as:

        D_ADP(i,j) = Σ_l ( 1 − |⟨z_i^l , z_j^l⟩|² )

    where *l* runs over the A×T component dimension and z_i^l is the
    unit-norm antenna vector for sample *i*, component *l*.  The sum equals
    ``num_components − Σ_l |⟨z_i^l, z_j^l⟩|²``.

    The computation is done in blocks on GPU (if available) to limit memory
    usage.

    Args:
        h: Raw CSI tensor of shape ``(N, A, M, T)``.
        block_size: Number of rows processed per GPU batch.
        device: PyTorch device string (e.g. ``"cuda:0"`` or ``"cpu"``).

    Returns:
        Symmetric non-negative float32 matrix of shape ``(N, N)``.
    """
    z = normalize_csi_vectors(h)                              # (N, A*T, M)
    n, n_components, _ = z.shape
    out = np.empty((n, n), dtype=np.float32)

    # Move to GPU (or CPU) once; slice from device tensor to avoid
    # repeated host→device transfers.
    z_t = torch.from_numpy(z).to(device)
    total = float(n_components)

    for start in range(0, n, block_size):
        stop = min(start + block_size, n)
        zi = z_t[start:stop]                                  # (block_i, A*T, M)
        for j_start in range(start, n, block_size):
            j_stop = min(j_start + block_size, n)
            zj = z_t[j_start:j_stop]                          # (block_j, A*T, M)

            # Dot products per component: conj(zi) ⊙ zj, summed over M
            # → (block_i, block_j, A*T)
            dots = torch.einsum("ilm,jlm->ijl", torch.conj(zi), zj)

            # ADP dissimilarity: total − Σ_l |dot_l|²
            block = total - torch.sum(torch.abs(dots) ** 2, dim=-1)

            block_np = block.detach().cpu().to(torch.float32).numpy()
            out[start:stop, j_start:j_stop] = block_np

            # Fill the symmetric counterpart (exploit D_ADP(i,j) = D_ADP(j,i))
            if j_start != start:
                out[j_start:j_stop, start:stop] = block_np.T

        print(f"ADP rows {stop}/{n}", flush=True)

    # Diagonal should be exactly zero (a sample is identical to itself)
    np.fill_diagonal(out, 0.0)
    return out


# ---------------------------------------------------------------------------
# Temporal scaling factor γ
# ---------------------------------------------------------------------------

def scaling_factor_from_time(
    adp: np.ndarray,
    timestamps: np.ndarray,
    time_threshold: float = 2.0,
    exact_notebook: bool = False,
) -> float:
    """Estimate the scaling factor γ that aligns ADP distances with timestamps.

    The idea: for pairs whose wall-clock separation is small (< threshold),
    the ADP distance should be roughly proportional to the time difference,
    i.e.  ADP(i,j) ≈ γ · |t_i − t_j|.  We find γ by histogramming the
    ratio ADP / Δt for all close-in-time pairs and locating the main mode,
    then setting γ to the *left shoulder* of that mode (where the count
    drops below the median of the left half).

    Args:
        adp: ADP dissimilarity matrix (N, N).
        timestamps: 1-D array of sample timestamps (seconds).
        time_threshold: Only consider pairs with Δt in (0, time_threshold).
        exact_notebook: If True, use the exact bin-picking logic from the
            reference notebook (slightly different edge-case handling).

    Returns:
        Estimated scalar γ.

    Raises:
        ValueError: If no pairs fall within the time threshold or the
            histogram is empty.
    """
    # Pairwise absolute time differences
    timestamp_d = np.abs(np.subtract.outer(timestamps, timestamps)).astype(np.float32)

    # Only consider close-in-time pairs (same mobility "regime")
    mask = (timestamp_d < time_threshold) & (timestamp_d > 0)
    if not np.any(mask):
        raise ValueError(
            f"no timestamp pairs in (0, {time_threshold}) seconds; "
            "use a denser sample stride or increase the time threshold"
        )

    # Ratio of ADP distance to time difference for eligible pairs
    ratios = adp[mask] / timestamp_d[mask]

    # Build a fine-grained histogram of the ratios
    occurrences, edges = np.histogram(ratios, range=(0, 50), bins=1500)
    centers = edges[:-1] + np.diff(edges) / 2.0

    # The main mode of the histogram
    max_bin = int(np.argmax(occurrences))

    if exact_notebook:
        # --- Exact logic matching the reference Jupyter notebook ---
        # Median count in the left half (bins 0..max_bin) as threshold
        min_threshold = np.quantile(occurrences[:max_bin], 0.5) if max_bin > 0 else 0.0
        threshold_bin = 0
        # Walk leftward from the mode; stop when count < median
        for candidate in range(max_bin - 1, -1, -1):
            threshold_bin = candidate
            if occurrences[candidate] < min_threshold:
                break
        return float(centers[threshold_bin])

    # --- Default (robust) path ---
    nonzero_bins = np.flatnonzero(occurrences)
    if len(nonzero_bins) == 0:
        raise ValueError("ADP/time histogram is empty in the configured range")
    first_nonzero = int(nonzero_bins[0])

    # Median count in the *left* portion of the mode (from first nonzero to mode)
    left_mode = occurrences[first_nonzero:max_bin]
    min_threshold = np.quantile(left_mode, 0.5) if len(left_mode) else 0

    # Walk leftward from mode; γ = bin centre where count first dips below median
    threshold_bin = first_nonzero
    for candidate in range(max_bin - 1, first_nonzero - 1, -1):
        threshold_bin = candidate
        if occurrences[candidate] < min_threshold:
            break

    return float(centers[threshold_bin])


# ---------------------------------------------------------------------------
# Fused geodesic distance
# ---------------------------------------------------------------------------

def fused_geodesic(
    adp: np.ndarray,
    timestamps: np.ndarray,
    gamma: float,
    n_neighbors: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Build a fused ADP+timestamp distance matrix and compute geodesic distances.

    The *fused* distance between samples *i* and *j* is:

        D_fused(i,j) = min( D_ADP(i,j),  γ · |t_i − t_j| )

    Taking the minimum ensures that both the channel-similarity and the
    temporal-adjacency signals contribute — whichever is *smaller* wins.

    From this fused matrix we build a k-NN graph (with precomputed
    distances) and then compute all-pairs shortest paths (geodesic
    distances) on that graph.  If the graph is disconnected, we increase
    *k* until it becomes connected.

    Args:
        adp: ADP dissimilarity matrix (N, N).
        timestamps: 1-D array of timestamps.
        gamma: Temporal scaling factor (from ``scaling_factor_from_time``).
        n_neighbors: Initial k for the kNN graph.

    Returns:
        Tuple of (fused matrix, geodesic matrix, final k used).

    Raises:
        RuntimeError: If the graph cannot be made connected or the
            geodesic matrix contains non-finite values.
    """
    timestamp_d = np.abs(np.subtract.outer(timestamps, timestamps)).astype(np.float32)

    # Element-wise minimum: take whichever of ADP or scaled-time is smaller
    fused = np.minimum(adp, timestamp_d * gamma).astype(np.float32)

    graph = None
    used_neighbors = n_neighbors

    # Iteratively increase k until the graph has exactly one connected component
    for used_neighbors in range(
        n_neighbors,
        min(fused.shape[0] - 1, max(n_neighbors, 200)) + 1,
        5,  # step size for neighbor count increase
    ):
        nbrs = NearestNeighbors(
            n_neighbors=used_neighbors, metric="precomputed", n_jobs=-1
        ).fit(fused)
        graph = kneighbors_graph(
            nbrs, used_neighbors, metric="precomputed", mode="distance", n_jobs=-1
        )
        n_components, _ = connected_components(graph, directed=False)
        print(f"kNN graph: neighbors={used_neighbors}, components={n_components}", flush=True)
        if n_components == 1:
            break

    if graph is None:
        raise RuntimeError("failed to build kNN graph")

    # All-pairs shortest path on the sparse graph → geodesic distance matrix
    geodesic = shortest_path(graph, method="auto", directed=False).astype(np.float32)

    if not np.all(np.isfinite(geodesic)):
        raise RuntimeError(
            f"geodesic matrix still contains non-finite values with {used_neighbors} neighbors"
        )

    return fused, geodesic, used_neighbors


# ---------------------------------------------------------------------------
# MDS embedding
# ---------------------------------------------------------------------------

def run_mds(dissimilarity: np.ndarray, seed: int, max_iter: int, n_init: int) -> np.ndarray:
    """Run classical (metric) MDS on a precomputed dissimilarity matrix.

    MDS finds a 2-D layout that best preserves the given pairwise distances.

    Args:
        dissimilarity: Symmetric non-negative (N, N) dissimilarity matrix.
        seed: Random seed for reproducibility.
        max_iter: Maximum number of SMACOF iterations.
        n_init: Number of random initializations (best result is kept).

    Returns:
        2-D embedding of shape ``(N, 2)``.
    """
    kwargs = {
        "n_components": 2,
        "metric": True,                  # metric MDS (preserves distances)
        "dissimilarity": "precomputed",  # input is already a distance matrix
        "max_iter": max_iter,
        "random_state": seed,
        "n_init": n_init,
        "eps": 1e-6,
    }
    try:
        # Newer scikit-learn versions support normalized_stress
        kwargs["normalized_stress"] = False
        return MDS(**kwargs).fit_transform(dissimilarity)
    except TypeError:
        # Older scikit-learn: remove the unsupported kwarg
        kwargs.pop("normalized_stress", None)
        return MDS(**kwargs).fit_transform(dissimilarity)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_chart(path: Path, reference: np.ndarray, embedding: np.ndarray, title: str) -> None:
    """Save a side-by-side scatter plot of ground truth vs. chart embedding.

    Points are coloured by their (x, y) ground-truth position so that
    spatially corresponding regions can be visually compared.

    Args:
        path: Output file path (e.g. ``results/.../chart.png``).
        reference: Ground-truth 2-D positions (N, 2).
        embedding: MDS embedding positions (N, 2), will be affinely aligned.
        title: Title for the right-hand subplot.
    """
    # Align embedding to ground truth via affine (similarity) transform
    aligned = affine_transform(reference, embedding)

    # Colour map: R = normalized x, G = normalized y, B = constant
    colors = np.zeros((len(reference), 3), dtype=np.float32)
    colors[:, 0] = (reference[:, 0] - reference[:, 0].min()) / np.ptp(reference[:, 0])
    colors[:, 1] = (reference[:, 1] - reference[:, 1].min()) / np.ptp(reference[:, 1])
    colors[:, 2] = 0.25

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))

    # Left: ground truth
    axes[0].scatter(reference[:, 0], reference[:, 1], c=colors, s=5, linewidths=0)
    axes[0].set_title("Ground Truth")
    axes[0].axis("equal")
    axes[0].grid(True, alpha=0.25)

    # Right: aligned embedding
    axes[1].scatter(aligned[:, 0], aligned[:, 1], c=colors, s=5, linewidths=0)
    axes[1].set_title(title)
    axes[1].axis("equal")
    axes[1].grid(True, alpha=0.25)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    # ---- Command-line arguments ----
    parser = argparse.ArgumentParser(
        description="ADP + timestamp fused channel charting pipeline"
    )
    parser.add_argument(
        "--data-path", type=Path, default=DEFAULT_DATA1,
        help="Path to the DICHASUS data1 directory",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=Path("results/adp_fused"),
        help="Root directory for experiment outputs",
    )
    parser.add_argument(
        "--stride", type=int, default=10,
        help="Subsample every N-th sample to reduce computation",
    )
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="Hard cap on number of samples (overrides stride if dataset is smaller)",
    )
    parser.add_argument(
        "--seed", type=int, default=10,
        help="Random seed for data splitting and MDS",
    )
    parser.add_argument(
        "--block-size", type=int, default=512,
        help="Number of rows per GPU block in ADP computation",
    )
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="PyTorch device for ADP computation",
    )
    parser.add_argument(
        "--neighbors", type=int, default=20,
        help="Initial k for the kNN graph in fused geodesic",
    )
    parser.add_argument(
        "--mds-iter", type=int, default=80,
        help="Maximum SMACOF iterations for MDS",
    )
    parser.add_argument(
        "--mds-n-init", type=int, default=4,
        help="Number of random MDS initializations",
    )
    parser.add_argument(
        "--mds-reduction", type=int, default=1,
        help="Further subsample ratio for MDS (1 = use all selected samples)",
    )
    parser.add_argument(
        "--exact-notebook-gamma", action="store_true",
        help="Use the exact γ estimation logic from the reference notebook",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Ignore cached .npy files and recompute everything",
    )
    args = parser.parse_args()

    t0 = time.time()

    # ---- 1. Load data ----
    # load_data1 returns a DichasusData namedtuple with:
    #   h_by_array: (N, 4, 8, 13)  — CSI tensor split by array
    #   ue_pos:     (N, 2)          — user-equipment ground-truth positions
    #   timestamps: (N,)            — wall-clock timestamps (seconds)
    data = load_data1(args.data_path, seed=args.seed)

    # Subsample to keep computation tractable
    idx = select_indices(len(data.h_by_array), args.stride, args.max_samples)
    h = data.h_by_array[idx]
    pos = data.ue_pos[idx]
    timestamps = data.timestamps[idx]

    # ---- 2. Prepare output directory & caching paths ----
    run_name = f"stride{args.stride}_n{len(idx)}_seed{args.seed}"
    out_dir = args.out_dir / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    adp_path = out_dir / "adp.npy"
    fused_path = out_dir / "fused.npy"
    geodesic_path = out_dir / "geodesic.npy"

    print(f"ADP/fused run: {run_name}")
    print(f"samples={len(idx)}, device={args.device}, out={out_dir}")

    # ---- 3. Compute (or load cached) ADP matrix ----
    if adp_path.exists() and not args.force:
        adp = np.load(adp_path)
        print("Loaded cached ADP matrix")
    else:
        adp = compute_adp_matrix(h, block_size=args.block_size, device=args.device)
        np.save(adp_path, adp)

    # ---- 4. Estimate temporal scaling factor γ ----
    gamma = scaling_factor_from_time(adp, timestamps, exact_notebook=args.exact_notebook_gamma)
    print(f"gamma={gamma:.6f}")

    # ---- 5. Compute (or load cached) fused + geodesic distances ----
    if geodesic_path.exists() and fused_path.exists() and not args.force:
        fused = np.load(fused_path)
        geodesic = np.load(geodesic_path)
        print("Loaded cached fused/geodesic matrices")
    else:
        fused, geodesic, used_neighbors = fused_geodesic(
            adp, timestamps, gamma=gamma, n_neighbors=args.neighbors
        )
        np.save(fused_path, fused)
        np.save(geodesic_path, geodesic)

    # Track which k was actually used (may have been increased for connectivity)
    if "used_neighbors" not in locals():
        used_neighbors = args.neighbors

    # ---- 6. MDS embedding ----
    # Optionally subsample further for MDS (expensive O(n²) in n_init iterations)
    mds_indices = np.arange(0, len(idx), args.mds_reduction, dtype=np.int64)
    embedding = run_mds(
        geodesic[np.ix_(mds_indices, mds_indices)],
        seed=args.seed,
        max_iter=args.mds_iter,
        n_init=args.mds_n_init,
    )

    # ---- 7. Evaluate the chart ----
    pos_mds = pos[mds_indices]
    metrics = chart_metrics(pos_mds, embedding)
    errors = localization_error_summary(pos_mds, embedding)

    # ---- 8. Assemble and save results ----
    result = {
        "paper": "Angle-Delay Profile-Based and Timestamp-Aided Dissimilarity Metrics for Channel Charting",
        "method": "ADP + timestamp fused + geodesic + metric MDS",
        "dataset": str(args.data_path),
        "run_name": run_name,
        "num_samples": int(len(idx)),
        "stride": int(args.stride),
        "seed": int(args.seed),
        "gamma": gamma,
        "neighbors": int(used_neighbors),
        "mds_iter": int(args.mds_iter),
        "mds_n_init": int(args.mds_n_init),
        "mds_reduction": int(args.mds_reduction),
        "mds_samples": int(len(mds_indices)),
        "exact_notebook_gamma": bool(args.exact_notebook_gamma),
        "metrics": metrics,
        "localization_errors_after_affine": errors,
        "elapsed_s": float(time.time() - t0),
    }

    # Save embedding, distance data, and JSON result
    np.savez(out_dir / "embedding.npz", indices=idx[mds_indices], positions=pos_mds, embedding=embedding)
    np.savez(out_dir / "distance_data.npz", indices=idx, positions=pos)
    write_json(out_dir / "result.json", result)

    # ---- 9. Visualization ----
    plot_chart(
        out_dir / "chart.png",
        pos_mds,
        embedding,
        f"ADP Fused MDS\nCT={metrics['ct']:.3f}, TW={metrics['tw']:.3f}, KS={metrics['ks']:.3f}",
    )

    print("RESULT", result)


if __name__ == "__main__":
    main()
