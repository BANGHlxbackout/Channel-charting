from __future__ import annotations

import argparse
import time
from pathlib import Path

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


def select_indices(n: int, stride: int, max_samples: int | None) -> np.ndarray:
    idx = np.arange(0, n, stride, dtype=np.int64)
    if max_samples is not None:
        idx = idx[:max_samples]
    return idx


def normalize_csi_vectors(h: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    # Input: (N, arrays, antennas, taps). Output: (N, arrays*taps, antennas).
    z = np.transpose(h, (0, 1, 3, 2)).reshape(h.shape[0], h.shape[1] * h.shape[3], h.shape[2])
    norm = np.linalg.norm(z, axis=-1, keepdims=True)
    return (z / np.maximum(norm, eps)).astype(np.complex64)


def compute_adp_matrix(h: np.ndarray, block_size: int, device: str) -> np.ndarray:
    z = normalize_csi_vectors(h)
    n, n_components, _ = z.shape
    out = np.empty((n, n), dtype=np.float32)
    z_t = torch.from_numpy(z).to(device)
    total = float(n_components)

    for start in range(0, n, block_size):
        stop = min(start + block_size, n)
        zi = z_t[start:stop]
        for j_start in range(start, n, block_size):
            j_stop = min(j_start + block_size, n)
            zj = z_t[j_start:j_stop]
            dots = torch.einsum("ilm,jlm->ijl", torch.conj(zi), zj)
            block = total - torch.sum(torch.abs(dots) ** 2, dim=-1)
            block_np = block.detach().cpu().to(torch.float32).numpy()
            out[start:stop, j_start:j_stop] = block_np
            if j_start != start:
                out[j_start:j_stop, start:stop] = block_np.T
        print(f"ADP rows {stop}/{n}", flush=True)

    np.fill_diagonal(out, 0.0)
    return out


def scaling_factor_from_time(adp: np.ndarray, timestamps: np.ndarray, time_threshold: float = 2.0, exact_notebook: bool = False) -> float:
    timestamp_d = np.abs(np.subtract.outer(timestamps, timestamps)).astype(np.float32)
    mask = (timestamp_d < time_threshold) & (timestamp_d > 0)
    if not np.any(mask):
        raise ValueError(
            f"no timestamp pairs in (0, {time_threshold}) seconds; "
            "use a denser sample stride or increase the time threshold"
        )
    ratios = adp[mask] / timestamp_d[mask]
    occurrences, edges = np.histogram(ratios, range=(0, 50), bins=1500)
    centers = edges[:-1] + np.diff(edges) / 2.0
    max_bin = int(np.argmax(occurrences))
    if exact_notebook:
        min_threshold = np.quantile(occurrences[:max_bin], 0.5) if max_bin > 0 else 0.0
        threshold_bin = 0
        for candidate in range(max_bin - 1, -1, -1):
            threshold_bin = candidate
            if occurrences[candidate] < min_threshold:
                break
        return float(centers[threshold_bin])
    nonzero_bins = np.flatnonzero(occurrences)
    if len(nonzero_bins) == 0:
        raise ValueError("ADP/time histogram is empty in the configured range")
    first_nonzero = int(nonzero_bins[0])
    left_mode = occurrences[first_nonzero:max_bin]
    min_threshold = np.quantile(left_mode, 0.5) if len(left_mode) else 0
    threshold_bin = first_nonzero
    for candidate in range(max_bin - 1, first_nonzero - 1, -1):
        threshold_bin = candidate
        if occurrences[candidate] < min_threshold:
            break
    return float(centers[threshold_bin])


def fused_geodesic(adp: np.ndarray, timestamps: np.ndarray, gamma: float, n_neighbors: int) -> tuple[np.ndarray, np.ndarray, int]:
    timestamp_d = np.abs(np.subtract.outer(timestamps, timestamps)).astype(np.float32)
    fused = np.minimum(adp, timestamp_d * gamma).astype(np.float32)
    graph = None
    used_neighbors = n_neighbors
    for used_neighbors in range(n_neighbors, min(fused.shape[0] - 1, max(n_neighbors, 200)) + 1, 5):
        nbrs = NearestNeighbors(n_neighbors=used_neighbors, metric="precomputed", n_jobs=-1).fit(fused)
        graph = kneighbors_graph(nbrs, used_neighbors, metric="precomputed", mode="distance", n_jobs=-1)
        n_components, _ = connected_components(graph, directed=False)
        print(f"kNN graph: neighbors={used_neighbors}, components={n_components}", flush=True)
        if n_components == 1:
            break
    if graph is None:
        raise RuntimeError("failed to build kNN graph")
    geodesic = shortest_path(graph, method="auto", directed=False).astype(np.float32)
    if not np.all(np.isfinite(geodesic)):
        raise RuntimeError(f"geodesic matrix still contains non-finite values with {used_neighbors} neighbors")
    return fused, geodesic, used_neighbors


def run_mds(dissimilarity: np.ndarray, seed: int, max_iter: int, n_init: int) -> np.ndarray:
    kwargs = {
        "n_components": 2,
        "metric": True,
        "dissimilarity": "precomputed",
        "max_iter": max_iter,
        "random_state": seed,
        "n_init": n_init,
        "eps": 1e-6,
    }
    try:
        kwargs["normalized_stress"] = False
        return MDS(**kwargs).fit_transform(dissimilarity)
    except TypeError:
        kwargs.pop("normalized_stress", None)
        return MDS(**kwargs).fit_transform(dissimilarity)


def plot_chart(path: Path, reference: np.ndarray, embedding: np.ndarray, title: str) -> None:
    aligned = affine_transform(reference, embedding)
    colors = np.zeros((len(reference), 3), dtype=np.float32)
    colors[:, 0] = (reference[:, 0] - reference[:, 0].min()) / np.ptp(reference[:, 0])
    colors[:, 1] = (reference[:, 1] - reference[:, 1].min()) / np.ptp(reference[:, 1])
    colors[:, 2] = 0.25

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].scatter(reference[:, 0], reference[:, 1], c=colors, s=5, linewidths=0)
    axes[0].set_title("Ground Truth")
    axes[0].axis("equal")
    axes[0].grid(True, alpha=0.25)
    axes[1].scatter(aligned[:, 0], aligned[:, 1], c=colors, s=5, linewidths=0)
    axes[1].set_title(title)
    axes[1].axis("equal")
    axes[1].grid(True, alpha=0.25)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA1)
    parser.add_argument("--out-dir", type=Path, default=Path("results/adp_fused"))
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--block-size", type=int, default=512)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--neighbors", type=int, default=20)
    parser.add_argument("--mds-iter", type=int, default=80)
    parser.add_argument("--mds-n-init", type=int, default=4)
    parser.add_argument("--mds-reduction", type=int, default=1)
    parser.add_argument("--exact-notebook-gamma", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    t0 = time.time()
    data = load_data1(args.data_path, seed=args.seed)
    idx = select_indices(len(data.h_by_array), args.stride, args.max_samples)
    h = data.h_by_array[idx]
    pos = data.ue_pos[idx]
    timestamps = data.timestamps[idx]

    run_name = f"stride{args.stride}_n{len(idx)}_seed{args.seed}"
    out_dir = args.out_dir / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    adp_path = out_dir / "adp.npy"
    fused_path = out_dir / "fused.npy"
    geodesic_path = out_dir / "geodesic.npy"

    print(f"ADP/fused run: {run_name}")
    print(f"samples={len(idx)}, device={args.device}, out={out_dir}")

    if adp_path.exists() and not args.force:
        adp = np.load(adp_path)
        print("Loaded cached ADP matrix")
    else:
        adp = compute_adp_matrix(h, block_size=args.block_size, device=args.device)
        np.save(adp_path, adp)

    gamma = scaling_factor_from_time(adp, timestamps, exact_notebook=args.exact_notebook_gamma)
    print(f"gamma={gamma:.6f}")

    if geodesic_path.exists() and fused_path.exists() and not args.force:
        fused = np.load(fused_path)
        geodesic = np.load(geodesic_path)
        print("Loaded cached fused/geodesic matrices")
    else:
        fused, geodesic, used_neighbors = fused_geodesic(adp, timestamps, gamma=gamma, n_neighbors=args.neighbors)
        np.save(fused_path, fused)
        np.save(geodesic_path, geodesic)
    if "used_neighbors" not in locals():
        used_neighbors = args.neighbors

    mds_indices = np.arange(0, len(idx), args.mds_reduction, dtype=np.int64)
    embedding = run_mds(
        geodesic[np.ix_(mds_indices, mds_indices)],
        seed=args.seed,
        max_iter=args.mds_iter,
        n_init=args.mds_n_init,
    )
    pos_mds = pos[mds_indices]
    metrics = chart_metrics(pos_mds, embedding)
    errors = localization_error_summary(pos_mds, embedding)
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
    np.savez(out_dir / "embedding.npz", indices=idx[mds_indices], positions=pos_mds, embedding=embedding)
    np.savez(out_dir / "distance_data.npz", indices=idx, positions=pos)
    write_json(out_dir / "result.json", result)
    plot_chart(out_dir / "chart.png", pos_mds, embedding, f"ADP Fused MDS\nCT={metrics['ct']:.3f}, TW={metrics['tw']:.3f}, KS={metrics['ks']:.3f}")

    print("RESULT", result)


if __name__ == "__main__":
    main()
