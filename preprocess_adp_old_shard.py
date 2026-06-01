from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parent
P2_ROOT = REPO_ROOT  # preprocess_dichasus_numpy.py 在同一目录
sys.path.insert(0, str(P2_ROOT))

from preprocess_dichasus_numpy import decode_tensor  # type: ignore  # noqa: E402
from tfrecord import example_pb2  # noqa: E402
from tfrecord.reader import tfrecord_iterator  # noqa: E402


PATHS = [
    ("dichasus-cf02.tfrecords", "reftx-offsets-dichasus-cf02.json"),
    ("dichasus-cf03.tfrecords", "reftx-offsets-dichasus-cf03.json"),
    ("dichasus-cf04.tfrecords", "reftx-offsets-dichasus-cf04.json"),
]

ASSIGNMENTS = [
    [0, 13, 31, 29, 3, 7, 1, 12],
    [30, 26, 21, 25, 24, 8, 22, 15],
    [28, 5, 10, 14, 6, 2, 16, 18],
    [19, 4, 23, 17, 20, 11, 9, 27],
]


def parse_example(record: memoryview) -> tuple[np.ndarray, np.ndarray, float]:
    example = example_pb2.Example()
    example.ParseFromString(bytes(record))
    features = example.features.feature
    csi = decode_tensor(features["csi"].bytes_list.value[0])
    pos = decode_tensor(features["pos-tachy"].bytes_list.value[0])
    timestamp = float(features["time"].float_list.value[0])
    return csi, pos, timestamp


def process_csi(csi_float: np.ndarray, offsets: dict) -> np.ndarray:
    csi = csi_float[..., 0].astype(np.float32) + 1j * csi_float[..., 1].astype(np.float32)
    csi = np.fft.fftshift(csi, axes=1)

    n_subcarriers = csi.shape[1]
    sto = np.asarray(offsets["sto"], dtype=np.float32)[:, None]
    cpo = np.asarray(offsets["cpo"], dtype=np.float32)[:, None]
    phase = sto * (2 * np.pi * np.arange(n_subcarriers, dtype=np.float32) / np.float32(n_subcarriers)) + cpo
    csi = csi * np.exp(1j * phase)

    time_domain = np.fft.fftshift(np.fft.ifft(np.fft.fftshift(csi, axes=1), axis=1), axes=1)
    taps = time_domain[:, 507:520]
    ordered = np.stack([taps[np.asarray(assignment, dtype=np.int64)] for assignment in ASSIGNMENTS])
    return ordered.astype(np.complex64)


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess ADP tutorial old-version shard(4,0) sparse TFRecords.")
    parser.add_argument("--raw-dir", type=Path, default=REPO_ROOT / "datasets")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "datasets" / "adp_old_shard4")
    parser.add_argument("--expect-samples", type=int, default=20997)
    parser.add_argument(
        "--sort-by-time",
        action="store_true",
        help="Sort samples by timestamp after preprocessing. The official notebook keeps TFRecord/shard order.",
    )
    args = parser.parse_args()

    h_rows: list[np.ndarray] = []
    positions: list[np.ndarray] = []
    timestamps: list[float] = []
    start = time.time()

    global_idx = 0
    for tfrecord_name, offsets_name in PATHS:
        path = args.raw_dir / tfrecord_name
        if not path.exists():
            raise FileNotFoundError(path)
        with (args.raw_dir / offsets_name).open() as f:
            offsets = json.load(f)
        print(f"processing {path}", flush=True)
        for record in tfrecord_iterator(str(path), None):
            # shard(4, 0): 每 4 个样本取 1 个（取 idx % 4 == 0 的样本）
            if global_idx % 4 == 0:
                csi_float, pos, timestamp = parse_example(record)
                h_rows.append(process_csi(csi_float, offsets).reshape(32, 13))
                positions.append(pos.astype(np.float64))
                timestamps.append(timestamp)
                if len(h_rows) % 1000 == 0:
                    print(f"  processed {len(h_rows)} samples in {(time.time() - start) / 60:.1f} min", flush=True)
            global_idx += 1

    H = np.stack(h_rows).astype(np.complex64)
    UE_pos = np.stack(positions).astype(np.float64)
    timestamps_arr = np.asarray(timestamps, dtype=np.float32)

    if args.sort_by_time:
        order = np.argsort(timestamps_arr)
        H = H[order]
        UE_pos = UE_pos[order]
        timestamps_arr = timestamps_arr[order]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.out_dir / "H.npy", H)
    np.save(args.out_dir / "UE_pos.npy", UE_pos)
    np.save(args.out_dir / "timestamps.npy", timestamps_arr)
    np.save(args.out_dir / "AP_pos.npy", np.zeros((4, 3), dtype=np.float32))

    print("H", H.shape, H.dtype)
    print("UE_pos", UE_pos.shape, UE_pos.dtype)
    print("timestamps", timestamps_arr.shape, timestamps_arr.dtype)
    if args.expect_samples and H.shape[0] != args.expect_samples:
        raise RuntimeError(f"expected {args.expect_samples} samples, got {H.shape[0]}")


if __name__ == "__main__":
    main()
