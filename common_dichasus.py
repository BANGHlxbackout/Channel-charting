from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


DEFAULT_DATA1 = (
    Path(__file__).resolve().parents[1]
    / "channel-charting-distributed-mimo-p2-fix"
    / "data1"
)


@dataclass(frozen=True)
class DichasusData:
    h_flat: np.ndarray
    h_by_array: np.ndarray
    ue_pos: np.ndarray
    timestamps: np.ndarray
    ap_pos: np.ndarray
    train_idx: np.ndarray
    test_idx: np.ndarray


def p2_split_indices(n_samples: int, seed: int = 10, test_ratio: float = 0.2) -> tuple[np.ndarray, np.ndarray]:
    """Match the measured-indoor P2 split protocol used by the local reproduction."""
    rng = np.random.RandomState(seed)
    n_test = round(n_samples * test_ratio)
    test_idx = rng.choice(n_samples, n_test, replace=False)
    all_idx = np.arange(n_samples)
    train_idx = all_idx[np.isin(all_idx, test_idx, invert=True)]
    return train_idx.astype(np.int64), test_idx.astype(np.int64)


def load_data1(data_path: str | Path = DEFAULT_DATA1, seed: int = 10, test_ratio: float = 0.2) -> DichasusData:
    data_path = Path(data_path)
    h_flat = np.load(data_path / "H.npy").astype(np.complex64)
    ue_pos = np.load(data_path / "UE_pos.npy")[:, :2].astype(np.float32)
    timestamps = np.load(data_path / "timestamps.npy").astype(np.float32)
    ap_pos = np.load(data_path / "AP_pos.npy")[:, :2].astype(np.float32)

    if h_flat.ndim != 3 or h_flat.shape[1:] != (32, 13):
        raise ValueError(f"expected H shape (N, 32, 13), got {h_flat.shape}")

    h_by_array = h_flat.reshape(h_flat.shape[0], 4, 8, 13)
    train_idx, test_idx = p2_split_indices(len(h_flat), seed=seed, test_ratio=test_ratio)

    return DichasusData(
        h_flat=h_flat,
        h_by_array=h_by_array,
        ue_pos=ue_pos,
        timestamps=timestamps,
        ap_pos=ap_pos,
        train_idx=train_idx,
        test_idx=test_idx,
    )


def main() -> None:
    data = load_data1()
    print("H flat:", data.h_flat.shape, data.h_flat.dtype)
    print("H by array:", data.h_by_array.shape, data.h_by_array.dtype)
    print("UE positions:", data.ue_pos.shape, data.ue_pos.dtype)
    print("timestamps:", data.timestamps.shape, data.timestamps.dtype)
    print("AP positions:", data.ap_pos.shape, data.ap_pos.dtype)
    print("train/test:", len(data.train_idx), len(data.test_idx))


if __name__ == "__main__":
    main()
