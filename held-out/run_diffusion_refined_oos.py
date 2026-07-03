from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HELDOUT_DIR = ROOT / "held-out"
for path in (ROOT, HELDOUT_DIR):
    path_text = str(path)
    if path_text not in sys.path:
        sys.path.insert(0, path_text)

from common_heldout import (  # noqa: E402
    apply_affine_transform,
    fit_affine_transform,
    heldout_stage_dir,
    localization_error_summary_aligned,
)
from common_metrics import chart_metrics, write_json  # noqa: E402
from run_adp_fused_heldout import validate_full_adp_cache  # noqa: E402
from run_adp_siamese_heldout import plot_aligned_chart  # noqa: E402


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_gamma(distance_run: Path) -> float:
    result = load_json(distance_run / "result.json")
    if "gamma_train" not in result:
        raise ValueError(f"{distance_run / 'result.json'} does not contain gamma_train")
    return float(result["gamma_train"])


def load_split(distance_run: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(distance_run / "distance_data.npz")
    return data["indices"].astype(np.int64), data["test_indices"].astype(np.int64)


def validate_embedding_indices(path: Path, expected_indices: np.ndarray) -> np.lib.npyio.NpzFile:
    data = np.load(path)
    if not np.array_equal(data["indices"], expected_indices):
        raise ValueError(f"{path} indices do not match held-out split")
    return data


def select_sorted_topk(distances: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    if not 1 <= k <= distances.shape[1]:
        raise ValueError("k must be in [1, number of candidates]")
    unsorted = np.argpartition(distances, kth=k - 1, axis=1)[:, :k]
    unsorted_values = np.take_along_axis(distances, unsorted, axis=1)
    order = np.argsort(unsorted_values, axis=1)
    indices = np.take_along_axis(unsorted, order, axis=1).astype(np.int64)
    values = np.take_along_axis(unsorted_values, order, axis=1).astype(np.float32)
    return indices, values


def compute_oos_geodesic_block(
    geodesic_train: np.ndarray,
    entry_local_indices: np.ndarray,
    entry_distances: np.ndarray,
) -> np.ndarray:
    if entry_local_indices.shape != entry_distances.shape:
        raise ValueError("entry indices and distances must have the same shape")

    block_size = entry_local_indices.shape[0]
    extended = np.empty((block_size, geodesic_train.shape[1]), dtype=np.float32)
    for row in range(block_size):
        candidates = (
            geodesic_train[entry_local_indices[row]].astype(np.float32, copy=False)
            + entry_distances[row, :, None]
        )
        extended[row] = np.min(candidates, axis=0)
    return extended


def compute_train_sigma(
    geodesic_train: np.ndarray,
    neighbors: np.ndarray,
    graph_k: int,
) -> np.ndarray:
    if geodesic_train.shape[0] != geodesic_train.shape[1]:
        raise ValueError("geodesic_train must be square")
    if neighbors.shape[0] != geodesic_train.shape[0]:
        raise ValueError("neighbors row count must match geodesic_train")
    if not 1 <= graph_k <= neighbors.shape[1]:
        raise ValueError("graph_k must be in [1, number of cached neighbors]")

    kth_neighbor = neighbors[:, graph_k - 1].astype(np.int64)
    sigma = np.asarray(
        geodesic_train[np.arange(geodesic_train.shape[0]), kth_neighbor],
        dtype=np.float32,
    )
    if not np.all(np.isfinite(sigma)):
        raise ValueError("train sigma contains non-finite values")
    return sigma


def compute_direct_fused_block(
    adp_full: np.ndarray,
    timestamps: np.ndarray,
    test_indices: np.ndarray,
    train_indices: np.ndarray,
    gamma_train: float,
) -> np.ndarray:
    adp_block = np.asarray(
        adp_full[np.ix_(test_indices, train_indices)],
        dtype=np.float32,
    )
    time_block = np.abs(
        timestamps[test_indices, None] - timestamps[train_indices][None, :]
    ).astype(np.float32)
    return np.minimum(adp_block, gamma_train * time_block).astype(np.float32)


def compute_oos_weights(
    neighbor_distances: np.ndarray,
    neighbor_local_indices: np.ndarray,
    sigma_test: np.ndarray,
    sigma_train: np.ndarray,
    eps: float,
) -> np.ndarray:
    train_scale = sigma_train[neighbor_local_indices]
    denom = np.maximum(sigma_test[:, None], eps) * np.maximum(train_scale, eps)
    return np.exp(-np.square(neighbor_distances) / denom).astype(np.float32)


def local_tikhonov_oos(
    y_test: np.ndarray,
    z_train: np.ndarray,
    neighbor_local_indices: np.ndarray,
    neighbor_weights: np.ndarray,
    lambda_oos: float,
) -> np.ndarray:
    z_neighbors = z_train[neighbor_local_indices]
    weighted_sum = np.sum(neighbor_weights[:, :, None] * z_neighbors, axis=1)
    weight_sum = np.sum(neighbor_weights, axis=1, keepdims=True)
    return (
        (y_test + lambda_oos * weighted_sum) / (1.0 + lambda_oos * weight_sum)
    ).astype(np.float32)


def compute_oos_extension(
    adp_full: np.ndarray,
    timestamps: np.ndarray,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    y_test: np.ndarray,
    z_train: np.ndarray,
    geodesic_train: np.ndarray,
    sigma_train: np.ndarray,
    gamma_train: float,
    entry_m: int,
    k: int,
    lambda_oos: float,
    block_size: int,
    eps: float,
) -> dict[str, np.ndarray]:
    num_test = len(test_indices)
    z_test = np.empty_like(y_test, dtype=np.float32)
    neighbor_local = np.empty((num_test, k), dtype=np.int64)
    neighbor_distances = np.empty((num_test, k), dtype=np.float32)
    neighbor_weights = np.empty((num_test, k), dtype=np.float32)
    entry_local = np.empty((num_test, entry_m), dtype=np.int64)
    entry_distances = np.empty((num_test, entry_m), dtype=np.float32)

    for start in range(0, num_test, block_size):
        stop = min(start + block_size, num_test)
        block_indices = test_indices[start:stop]
        fused = compute_direct_fused_block(
            adp_full,
            timestamps,
            block_indices,
            train_indices,
            gamma_train,
        )
        block_entry_local, block_entry_dist = select_sorted_topk(fused, entry_m)
        oos_geo = compute_oos_geodesic_block(
            geodesic_train,
            block_entry_local,
            block_entry_dist,
        )
        block_neighbor_local, block_neighbor_dist = select_sorted_topk(oos_geo, k)
        sigma_test = block_neighbor_dist[:, -1]
        block_weights = compute_oos_weights(
            block_neighbor_dist,
            block_neighbor_local,
            sigma_test,
            sigma_train,
            eps,
        )
        block_z = local_tikhonov_oos(
            y_test[start:stop],
            z_train,
            block_neighbor_local,
            block_weights,
            lambda_oos,
        )

        z_test[start:stop] = block_z
        neighbor_local[start:stop] = block_neighbor_local
        neighbor_distances[start:stop] = block_neighbor_dist
        neighbor_weights[start:stop] = block_weights
        entry_local[start:stop] = block_entry_local
        entry_distances[start:stop] = block_entry_dist
        print(f"OOS block {start}:{stop} / {num_test}", flush=True)

    return {
        "embedding": z_test,
        "neighbor_local_indices": neighbor_local,
        "neighbor_distances": neighbor_distances,
        "neighbor_weights": neighbor_weights,
        "entry_local_indices": entry_local,
        "entry_distances": entry_distances,
    }


def evaluate_oos(
    pos_train: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    z_train: np.ndarray,
    z_test: np.ndarray,
    pos_test: np.ndarray,
) -> dict[str, object]:
    fcf_transform = fit_affine_transform(pos_train, y_train)
    fcf_aligned = apply_affine_transform(y_test, fcf_transform)
    diff_transform = fit_affine_transform(pos_train, z_train)
    diff_aligned = apply_affine_transform(z_test, diff_transform)
    return {
        "fcf_test_localization_errors": localization_error_summary_aligned(
            pos_test,
            fcf_aligned,
        ),
        "diffusion_oos_test_localization_errors": localization_error_summary_aligned(
            pos_test,
            diff_aligned,
        ),
        "fcf_test_metrics": chart_metrics(pos_test, y_test),
        "diffusion_oos_test_metrics": chart_metrics(pos_test, z_test),
        "fcf_aligned_test": fcf_aligned,
        "diffusion_oos_aligned_test": diff_aligned,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run diffusion-refined held-out out-of-sample baseline."
    )
    parser.add_argument("--data-path", type=Path, default=Path("datasets/adp_old_shard4"))
    parser.add_argument(
        "--distance-run",
        type=Path,
        default=Path("results/heldout/seed10/adp_fused_train"),
    )
    parser.add_argument(
        "--full-distance-run",
        type=Path,
        default=Path("results/adp_old_fused/stride1_n20997_seed10"),
    )
    parser.add_argument(
        "--fcf-train",
        type=Path,
        default=Path("results/heldout/seed10/adp_siamese/embedding_train.npz"),
    )
    parser.add_argument(
        "--fcf-test",
        type=Path,
        default=Path("results/heldout/seed10/adp_siamese/embedding_test.npz"),
    )
    parser.add_argument(
        "--train-refinement",
        type=Path,
        default=Path("results/heldout/seed10/diffusion_train/embedding_train.npz"),
    )
    parser.add_argument("--out-dir", type=Path, default=Path("results/heldout"))
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--entry-m", type=int, default=32)
    parser.add_argument("--k", type=int, default=128)
    parser.add_argument("--graph-k", type=int, default=128)
    parser.add_argument("--lambda-oos", type=float, default=0.15)
    parser.add_argument("--top-k", type=int, default=512)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--eps", type=float, default=1e-8)
    args = parser.parse_args()

    started = time.time()
    if args.entry_m < 1:
        raise ValueError("--entry-m must be positive")
    if args.k < 1:
        raise ValueError("--k must be positive")
    if args.block_size < 1:
        raise ValueError("--block-size must be positive")

    timestamps = np.load(args.data_path / "timestamps.npy").astype(np.float32)
    train_indices, test_indices = load_split(args.distance_run)
    gamma_train = load_gamma(args.distance_run)

    fcf_train = validate_embedding_indices(args.fcf_train, train_indices)
    fcf_test = validate_embedding_indices(args.fcf_test, test_indices)
    train_refinement = validate_embedding_indices(args.train_refinement, train_indices)

    y_train = fcf_train["embedding"].astype(np.float32)
    y_test = fcf_test["embedding"].astype(np.float32)
    z_train = train_refinement["embedding"].astype(np.float32)
    pos_train = fcf_train["positions"].astype(np.float32)

    geodesic_train = np.load(args.distance_run / "geodesic_train.npy", mmap_mode="r")
    if geodesic_train.shape != (len(train_indices), len(train_indices)):
        raise ValueError("train geodesic shape does not match held-out train split")
    neighbors = np.load(args.distance_run / f"geodesic_top{args.top_k}.npy", mmap_mode="r")
    sigma_train = compute_train_sigma(
        geodesic_train,
        neighbors,
        graph_k=args.graph_k,
    )

    adp_full, full_indices = validate_full_adp_cache(
        args.full_distance_run,
        num_samples=len(timestamps),
    )
    if not np.array_equal(full_indices, np.arange(len(timestamps), dtype=full_indices.dtype)):
        raise ValueError("full ADP cache is not in complete dataset order")

    extension = compute_oos_extension(
        adp_full=adp_full,
        timestamps=timestamps,
        train_indices=train_indices,
        test_indices=test_indices,
        y_test=y_test,
        z_train=z_train,
        geodesic_train=geodesic_train,
        sigma_train=sigma_train,
        gamma_train=gamma_train,
        entry_m=args.entry_m,
        k=args.k,
        lambda_oos=args.lambda_oos,
        block_size=args.block_size,
        eps=args.eps,
    )

    pos_test = fcf_test["positions"].astype(np.float32)
    evaluation = evaluate_oos(
        pos_train=pos_train,
        y_train=y_train,
        y_test=y_test,
        z_train=z_train,
        z_test=extension["embedding"],
        pos_test=pos_test,
    )

    out_dir = heldout_stage_dir(args.out_dir, args.seed, "diffusion_oos")
    out_dir.mkdir(parents=True, exist_ok=True)
    output_npz = out_dir / "embedding_test.npz"
    np.savez(
        output_npz,
        indices=test_indices,
        positions=pos_test,
        init_embedding=y_test,
        embedding=extension["embedding"],
        neighbor_indices=train_indices[extension["neighbor_local_indices"]],
        neighbor_local_indices=extension["neighbor_local_indices"],
        neighbor_weights=extension["neighbor_weights"],
        neighbor_distances=extension["neighbor_distances"],
        entry_indices=train_indices[extension["entry_local_indices"]],
        entry_local_indices=extension["entry_local_indices"],
        entry_distances=extension["entry_distances"],
    )
    chart_path = out_dir / "chart.png"
    plot_aligned_chart(
        chart_path,
        pos_test,
        evaluation["diffusion_oos_aligned_test"],
        (
            "Diffusion Refined Held-Out Test\n"
            f"CT={evaluation['diffusion_oos_test_metrics']['ct']:.3f}, "
            f"TW={evaluation['diffusion_oos_test_metrics']['tw']:.3f}"
        ),
    )

    result = {
        "protocol": "diffusion_refined_oos_heldout",
        "dataset": str(args.data_path),
        "distance_run": str(args.distance_run),
        "full_distance_run": str(args.full_distance_run),
        "fcf_train": str(args.fcf_train),
        "fcf_test": str(args.fcf_test),
        "train_refinement": str(args.train_refinement),
        "output_npz": str(output_npz),
        "chart": str(chart_path),
        "seed": int(args.seed),
        "num_train": int(len(train_indices)),
        "num_test": int(len(test_indices)),
        "entry_m": int(args.entry_m),
        "k": int(args.k),
        "graph_k": int(args.graph_k),
        "top_k": int(args.top_k),
        "lambda_oos": float(args.lambda_oos),
        "gamma_train": float(gamma_train),
        "uses_test_test_graph": False,
        "uses_test_positions_for_extension": False,
        "uses_test_positions_for_evaluation_only": True,
        "chart_alignment": "train_fitted_affine_applied_to_test",
        "oos_distance": "min-plus extension from direct test-to-train fused distance into train geodesic graph",
        "weight_formula": "exp(-(d_oos_geo^2)/(sigma_test*sigma_train))",
        "fcf_test_localization_errors": evaluation["fcf_test_localization_errors"],
        "diffusion_oos_test_localization_errors": evaluation[
            "diffusion_oos_test_localization_errors"
        ],
        "fcf_test_metrics": evaluation["fcf_test_metrics"],
        "diffusion_oos_test_metrics": evaluation["diffusion_oos_test_metrics"],
        "elapsed_s": float(time.time() - started),
    }
    write_json(out_dir / "result.json", result)
    print("RESULT", json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
