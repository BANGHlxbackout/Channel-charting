from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from common_dichasus import DEFAULT_DATA1, load_data1
from common_heldout import heldout_stage_dir, validate_split
from common_metrics import write_json
from run_adp_fused import fused_geodesic, scaling_factor_from_time


def validate_full_adp_cache(
    full_distance_run: Path,
    num_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    saved = np.load(full_distance_run / "distance_data.npz")
    indices = saved["indices"]
    expected = np.arange(num_samples, dtype=indices.dtype)
    if not np.array_equal(indices, expected):
        raise ValueError("full ADP cache indices do not match complete dataset order")

    adp = np.load(full_distance_run / "adp.npy", mmap_mode="r")
    expected_shape = (num_samples, num_samples)
    if adp.shape != expected_shape:
        raise ValueError(f"expected ADP shape {expected_shape}, got {adp.shape}")
    return adp, indices


def compute_train_geodesic(
    adp_full: np.ndarray,
    timestamps: np.ndarray,
    train_idx: np.ndarray,
    n_neighbors: int,
) -> tuple[float, np.ndarray, int]:
    adp_train = np.asarray(
        adp_full[np.ix_(train_idx, train_idx)],
        dtype=np.float32,
    )
    train_timestamps = timestamps[train_idx]
    gamma = scaling_factor_from_time(
        adp_train,
        train_timestamps,
        exact_notebook=False,
    )
    fused_train, geodesic_train, used_neighbors = fused_geodesic(
        adp_train,
        train_timestamps,
        gamma=gamma,
        n_neighbors=n_neighbors,
    )
    del fused_train
    return gamma, geodesic_train, used_neighbors


def cached_outputs_valid(out_dir: Path, train_idx: np.ndarray, test_idx: np.ndarray) -> bool:
    geodesic_path = out_dir / "geodesic_train.npy"
    data_path = out_dir / "distance_data.npz"
    result_path = out_dir / "result.json"
    if not (geodesic_path.exists() and data_path.exists() and result_path.exists()):
        return False

    saved = np.load(data_path)
    return bool(
        np.array_equal(saved["indices"], train_idx)
        and np.array_equal(saved["test_indices"], test_idx)
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build held-out train-only ADP/time geodesic distances."
    )
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA1)
    parser.add_argument("--full-distance-run", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("results/heldout"))
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--neighbors", type=int, default=20)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    started = time.time()
    data = load_data1(args.data_path, seed=args.seed)
    validate_split(len(data.h_flat), data.train_idx, data.test_idx)

    out_dir = heldout_stage_dir(args.out_dir, args.seed, "adp_fused_train")
    out_dir.mkdir(parents=True, exist_ok=True)

    adp_full, _ = validate_full_adp_cache(
        args.full_distance_run,
        num_samples=len(data.h_flat),
    )

    if cached_outputs_valid(out_dir, data.train_idx, data.test_idx) and not args.force:
        geodesic_train = np.load(out_dir / "geodesic_train.npy", mmap_mode="r")
        result = json.loads((out_dir / "result.json").read_text(encoding="utf-8"))
        print(
            "Loaded cached held-out train geodesic",
            geodesic_train.shape,
            flush=True,
        )
        print("RESULT", json.dumps(result, indent=2, sort_keys=True))
        return

    gamma, geodesic_train, used_neighbors = compute_train_geodesic(
        adp_full,
        data.timestamps,
        data.train_idx,
        n_neighbors=args.neighbors,
    )

    if geodesic_train.shape != (len(data.train_idx), len(data.train_idx)):
        raise ValueError(
            f"expected train geodesic shape {(len(data.train_idx), len(data.train_idx))}, "
            f"got {geodesic_train.shape}"
        )
    if not np.all(np.isfinite(geodesic_train)):
        raise ValueError("train geodesic contains non-finite values")

    np.save(out_dir / "geodesic_train.npy", geodesic_train.astype(np.float32, copy=False))
    np.savez(
        out_dir / "distance_data.npz",
        indices=data.train_idx,
        positions=data.ue_pos[data.train_idx],
        test_indices=data.test_idx,
    )

    result = {
        "protocol": "heldout_train_graph",
        "dataset": str(args.data_path),
        "source_full_distance_run": str(args.full_distance_run),
        "seed": int(args.seed),
        "num_samples": int(len(data.h_flat)),
        "num_train": int(len(data.train_idx)),
        "num_test": int(len(data.test_idx)),
        "gamma_train": float(gamma),
        "neighbors": int(used_neighbors),
        "geodesic_shape": [int(x) for x in geodesic_train.shape],
        "seed10_historical_visibility": bool(args.seed == 10),
        "elapsed_s": float(time.time() - started),
    }
    write_json(out_dir / "result.json", result)
    print("RESULT", json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
