from __future__ import annotations

import argparse
import hashlib
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

from common_dichasus import p2_split_indices
from common_heldout import heldout_stage_dir
from common_metrics import write_json
from run_adp_fused_heldout import validate_full_adp_cache


def load_timestamps_and_split(
    data_path: Path,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    timestamps = np.load(data_path / "timestamps.npy").astype(np.float32)
    train_idx, test_idx = p2_split_indices(len(timestamps), seed=seed)
    return timestamps, train_idx, test_idx


def array_sha256(values: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(values)
    return hashlib.sha256(contiguous.view(np.uint8)).hexdigest()


def load_split(distance_run: Path) -> tuple[np.ndarray, np.ndarray]:
    split_data = np.load(distance_run / "distance_data.npz")
    return split_data["indices"].astype(np.int64), split_data["test_indices"].astype(np.int64)


def load_gamma(distance_run: Path) -> float:
    result = json.loads((distance_run / "result.json").read_text(encoding="utf-8"))
    if "gamma_train" not in result:
        raise ValueError(f"{distance_run / 'result.json'} does not contain gamma_train")
    return float(result["gamma_train"])


def choose_farthest_landmarks(
    geodesic_train: np.ndarray,
    num_landmarks: int,
) -> np.ndarray:
    n_train = geodesic_train.shape[0]
    if geodesic_train.shape != (n_train, n_train):
        raise ValueError("geodesic_train must be square")
    if not 1 <= num_landmarks <= n_train:
        raise ValueError("--num-landmarks must be in [1, num_train]")

    row_mean = np.asarray(np.mean(geodesic_train, axis=1), dtype=np.float64)
    if not np.all(np.isfinite(row_mean)):
        raise ValueError("geodesic_train row means contain non-finite values")
    first = int(np.argmin(row_mean))

    landmarks = np.empty(num_landmarks, dtype=np.int64)
    landmarks[0] = first
    min_distance = np.asarray(geodesic_train[:, first], dtype=np.float32).copy()
    min_distance[first] = -np.inf

    for pos in range(1, num_landmarks):
        next_idx = int(np.argmax(min_distance))
        if not np.isfinite(min_distance[next_idx]):
            raise ValueError("farthest-point sampling ran out of finite candidates")
        landmarks[pos] = next_idx
        candidate_distance = np.asarray(geodesic_train[:, next_idx], dtype=np.float32)
        min_distance = np.minimum(min_distance, candidate_distance)
        min_distance[landmarks[: pos + 1]] = -np.inf

    return landmarks


def compute_fused_to_landmarks(
    adp_full: np.ndarray,
    timestamps: np.ndarray,
    sample_indices: np.ndarray,
    landmark_indices: np.ndarray,
    gamma_train: float,
) -> np.ndarray:
    adp_block = np.asarray(
        adp_full[np.ix_(sample_indices, landmark_indices)],
        dtype=np.float32,
    )
    time_block = np.abs(
        timestamps[sample_indices, None] - timestamps[landmark_indices][None, :]
    ).astype(np.float32)
    return np.minimum(adp_block, gamma_train * time_block).astype(np.float32)


def train_rbf_context(
    fused_train: np.ndarray,
    eps: float,
) -> tuple[np.ndarray, np.ndarray]:
    sigma = np.median(fused_train, axis=0).astype(np.float32)
    positive = sigma[sigma > eps]
    fallback = float(np.median(positive)) if len(positive) else 1.0
    sigma = np.where(sigma > eps, sigma, fallback).astype(np.float32)
    q_train = np.exp(-np.square(fused_train / np.maximum(sigma[None, :], eps))).astype(np.float32)
    return q_train, sigma


def apply_rbf_context(
    fused: np.ndarray,
    sigma: np.ndarray,
    eps: float,
) -> np.ndarray:
    return np.exp(-np.square(fused / np.maximum(sigma[None, :], eps))).astype(np.float32)


def validate_context(
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    landmark_indices: np.ndarray,
    q_train: np.ndarray,
    q_test: np.ndarray,
    sigma: np.ndarray,
) -> dict[str, object]:
    train_set = set(int(x) for x in train_indices.tolist())
    landmark_set = set(int(x) for x in landmark_indices.tolist())
    if not landmark_set.issubset(train_set):
        raise ValueError("landmark_indices must be a subset of train_indices")
    if len(landmark_set) != len(landmark_indices):
        raise ValueError("landmark_indices contain duplicates")
    expected_train_shape = (len(train_indices), len(landmark_indices))
    expected_test_shape = (len(test_indices), len(landmark_indices))
    if q_train.shape != expected_train_shape:
        raise ValueError(f"expected q_train shape {expected_train_shape}, got {q_train.shape}")
    if q_test.shape != expected_test_shape:
        raise ValueError(f"expected q_test shape {expected_test_shape}, got {q_test.shape}")
    for name, values in {"q_train": q_train, "q_test": q_test, "sigma": sigma}.items():
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{name} contains non-finite values")
    if np.any(sigma <= 0):
        raise ValueError("sigma must be positive")

    return {
        "q_train_shape": [int(x) for x in q_train.shape],
        "q_test_shape": [int(x) for x in q_test.shape],
        "q_train_min": float(np.min(q_train)),
        "q_train_max": float(np.max(q_train)),
        "q_train_mean": float(np.mean(q_train)),
        "q_test_min": float(np.min(q_test)),
        "q_test_max": float(np.max(q_test)),
        "q_test_mean": float(np.mean(q_test)),
        "sigma_min": float(np.min(sigma)),
        "sigma_median": float(np.median(sigma)),
        "sigma_max": float(np.max(sigma)),
        "train_indices_sha256": array_sha256(train_indices),
        "test_indices_sha256": array_sha256(test_indices),
        "landmark_indices_sha256": array_sha256(landmark_indices),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate CF-HRD physical landmark context for held-out train/test samples."
    )
    parser.add_argument("--data-path", type=Path, default=Path("datasets/adp_old_shard4"))
    parser.add_argument(
        "--distance-run",
        type=Path,
        default=Path("results/heldout/seed10/adp_fused_train"),
        help="Held-out train ADP/time graph stage.",
    )
    parser.add_argument(
        "--full-distance-run",
        type=Path,
        default=Path("results/adp_old_fused/stride1_n20997_seed10"),
        help="Full ADP cache used only for sample-to-train-landmark distances.",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("results/CF_HRD"))
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--num-landmarks", type=int, default=128)
    parser.add_argument("--eps", type=float, default=1e-6)
    args = parser.parse_args()

    started = time.time()
    timestamps, expected_train_idx, expected_test_idx = load_timestamps_and_split(
        args.data_path,
        seed=args.seed,
    )
    train_indices, test_indices = load_split(args.distance_run)
    if not np.array_equal(train_indices, expected_train_idx):
        raise ValueError("distance-run train indices do not match data split")
    if not np.array_equal(test_indices, expected_test_idx):
        raise ValueError("distance-run test indices do not match data split")

    gamma_train = load_gamma(args.distance_run)
    adp_full, full_indices = validate_full_adp_cache(
        args.full_distance_run,
        num_samples=len(timestamps),
    )
    if not np.array_equal(full_indices, np.arange(len(timestamps), dtype=full_indices.dtype)):
        raise ValueError("full ADP cache is not in complete dataset order")

    geodesic_train = np.load(args.distance_run / "geodesic_train.npy", mmap_mode="r")
    landmark_local_indices = choose_farthest_landmarks(
        geodesic_train,
        num_landmarks=args.num_landmarks,
    )
    landmark_indices = train_indices[landmark_local_indices]

    fused_train = compute_fused_to_landmarks(
        adp_full,
        timestamps,
        train_indices,
        landmark_indices,
        gamma_train=gamma_train,
    )
    fused_test = compute_fused_to_landmarks(
        adp_full,
        timestamps,
        test_indices,
        landmark_indices,
        gamma_train=gamma_train,
    )
    q_train, sigma = train_rbf_context(fused_train, eps=args.eps)
    q_test = apply_rbf_context(fused_test, sigma=sigma, eps=args.eps)

    summary = validate_context(
        train_indices,
        test_indices,
        landmark_indices,
        q_train,
        q_test,
        sigma,
    )

    out_dir = heldout_stage_dir(args.out_dir, args.seed, "landmark_context")
    out_dir.mkdir(parents=True, exist_ok=True)
    output_npz = out_dir / "context.npz"
    np.savez(
        output_npz,
        train_indices=train_indices,
        test_indices=test_indices,
        landmark_indices=landmark_indices,
        landmark_local_indices=landmark_local_indices,
        q_train=q_train,
        q_test=q_test,
        sigma=sigma,
        gamma_train=np.asarray(gamma_train, dtype=np.float32),
        fused_train=fused_train,
        fused_test=fused_test,
    )

    result = {
        "protocol": "cf_hrd_stage3_landmark_context",
        "description": (
            "Train-only landmark selection plus train/test ADP-time fused distances to "
            "train landmarks. Test features use test CSI/timestamps only against train "
            "landmarks; no test positions, test-test graph, or train+test diffusion solve."
        ),
        "dataset": str(args.data_path),
        "distance_run": str(args.distance_run),
        "full_distance_run": str(args.full_distance_run),
        "output_npz": str(output_npz),
        "seed": int(args.seed),
        "num_train": int(len(train_indices)),
        "num_test": int(len(test_indices)),
        "num_landmarks": int(args.num_landmarks),
        "landmark_selection": "farthest_point_on_train_geodesic_start_min_mean_geodesic",
        "feature": "rbf_of_direct_adp_time_fused_distance_to_train_landmarks",
        "feature_formula": "q_i,b = exp(-(min(adp(i,a_b), gamma_train*abs(dt)) / sigma_b)^2)",
        "normalization": "sigma_b is median train fused distance to landmark b; fitted on train only",
        "gamma_train": float(gamma_train),
        "uses_test_positions_for_features": False,
        "uses_test_test_graph": False,
        "uses_train_test_diffusion_solve": False,
        "source_arrays": {
            "adp_full": str(args.full_distance_run / "adp.npy"),
            "geodesic_train": str(args.distance_run / "geodesic_train.npy"),
            "timestamps": str(args.data_path / "timestamps.npy"),
        },
        "validation": summary,
        "elapsed_s": float(time.time() - started),
    }
    write_json(out_dir / "result.json", result)
    print("RESULT", json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
