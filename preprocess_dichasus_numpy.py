#!/usr/bin/env python3
"""TensorFlow-free preprocessing for the DICHASUS cf02/cf03/cf04 data.

This mirrors the default path in preprocess_dichasus.py:
TFRecord -> offset calibration -> antenna assignment ordering -> time-domain CSI
-> keep every fifth sample -> taps 507:520 -> data1/*.npy.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from tensorboard.compat.proto import tensor_pb2, types_pb2
from tfrecord import example_pb2
from tfrecord.reader import tfrecord_iterator


PATHS = [
    ("dichasus-cf02.tfrecords", "reftx-offsets-dichasus-cf02.json"),
    ("dichasus-cf03.tfrecords", "reftx-offsets-dichasus-cf03.json"),
    ("dichasus-cf04.tfrecords", "reftx-offsets-dichasus-cf04.json"),
]


def decode_tensor(serialized: bytes) -> np.ndarray:
    tensor = tensor_pb2.TensorProto()
    tensor.ParseFromString(serialized)
    shape = tuple(dim.size for dim in tensor.tensor_shape.dim)

    if tensor.dtype == types_pb2.DT_FLOAT:
        dtype = np.dtype("<f4")
        repeated = tensor.float_val
    elif tensor.dtype == types_pb2.DT_DOUBLE:
        dtype = np.dtype("<f8")
        repeated = tensor.double_val
    elif tensor.dtype == types_pb2.DT_INT64:
        dtype = np.dtype("<i8")
        repeated = tensor.int64_val
    elif tensor.dtype == types_pb2.DT_INT32:
        dtype = np.dtype("<i4")
        repeated = tensor.int_val
    else:
        raise TypeError(f"unsupported TensorProto dtype {tensor.dtype}")

    if tensor.tensor_content:
        arr = np.frombuffer(tensor.tensor_content, dtype=dtype).copy()
    else:
        arr = np.asarray(repeated, dtype=dtype)
        if arr.size == 1 and np.prod(shape, dtype=np.int64) > 1:
            arr = np.repeat(arr, int(np.prod(shape, dtype=np.int64)))
    return arr.reshape(shape)


def parse_example(record: memoryview) -> tuple[np.ndarray, np.ndarray, float]:
    example = example_pb2.Example()
    example.ParseFromString(bytes(record))
    features = example.features.feature
    csi = decode_tensor(features["csi"].bytes_list.value[0])
    pos = decode_tensor(features["pos-tachy"].bytes_list.value[0])
    timestamp = float(features["time"].float_list.value[0])
    return csi, pos, timestamp


def load_spec(raw_dir: Path) -> tuple[list[list[list[int]]], np.ndarray, float]:
    with (raw_dir / "spec.json").open() as f:
        spec = json.load(f)
    assignments = [antenna["assignments"] for antenna in spec["antennas"]]
    ap_pos = np.asarray([antenna["location"] for antenna in spec["antennas"]], dtype=np.float64)
    antenna_count = sum(len(row) for antenna in assignments for row in antenna)
    if antenna_count != 32:
        raise ValueError(f"expected 32 antennas, got {antenna_count}")
    return assignments, ap_pos, float(spec["bandwidth"])


def process_csi(csi_float: np.ndarray, offsets: dict, assignments: list[list[list[int]]]) -> tuple[np.ndarray, np.ndarray]:
    csi = csi_float[..., 0].astype(np.float32) + 1j * csi_float[..., 1].astype(np.float32)
    csi = np.fft.fftshift(csi, axes=1)

    n_subcarriers = csi.shape[1]
    sto = np.asarray(offsets["sto"], dtype=np.float32)[:, None]
    cpo = np.asarray(offsets["cpo"], dtype=np.float32)[:, None]
    phase = sto * (2 * np.pi * np.arange(n_subcarriers, dtype=np.float32) / np.float32(n_subcarriers)) + cpo
    csi = csi * np.exp(1j * phase)

    ordered = np.asarray(
        [[csi[np.asarray(row, dtype=np.int64)] for row in antenna_assignment] for antenna_assignment in assignments],
        dtype=np.complex64,
    )
    time_domain = np.fft.fftshift(
        np.fft.ifft(np.fft.fftshift(ordered, axes=-1), axis=-1),
        axes=-1,
    ).astype(np.complex64)

    pow_per_ap_full = 20 * np.log10(np.linalg.norm(time_domain.reshape(time_domain.shape[0], -1), ord=2, axis=-1))
    taps = time_domain[:, :, :, 507:520].reshape(time_domain.shape[0], 8, 13)
    return taps, pow_per_ap_full.astype(np.float32)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default="data_raw")
    parser.add_argument("--out-dir", default="data1")
    parser.add_argument("--expect-samples", type=int, default=16681)
    parser.add_argument(
        "--already-decimated",
        action="store_true",
        help="Process every record in the input TFRecords; use when files contain only idx %% 5 == 0 records.",
    )
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    assignments, ap_pos, _ = load_spec(raw_dir)

    H_rows = []
    positions = []
    timestamps = []
    powers = []
    global_index = 0
    kept = 0
    start_time = time.monotonic()

    for tfrecord_name, offsets_name in PATHS:
        with (raw_dir / offsets_name).open() as f:
            offsets = json.load(f)
        path = raw_dir / tfrecord_name
        print(f"processing {path}", flush=True)
        for record in tfrecord_iterator(str(path), None):
            if args.already_decimated or global_index % 5 == 0:
                csi_float, pos, timestamp = parse_example(record)
                taps, pow_per_ap_full = process_csi(csi_float, offsets, assignments)
                H_rows.append(taps.reshape(32, 13))
                positions.append(pos.astype(np.float64))
                timestamps.append(timestamp)
                powers.append(pow_per_ap_full)
                kept += 1
                if kept % 1000 == 0:
                    elapsed = time.monotonic() - start_time
                    print(f"kept {kept} samples in {elapsed / 60:.1f} min", flush=True)
            global_index += 1

    H = np.stack(H_rows).astype(np.complex64)
    UE_pos = np.stack(positions).astype(np.float64)
    timestamps = np.asarray(timestamps, dtype=np.float32)
    pow_per_ap_full = np.stack(powers).astype(np.float32)

    order = np.argsort(timestamps)
    H = H[order]
    UE_pos = UE_pos[order]
    timestamps = timestamps[order]
    pow_per_ap_full = pow_per_ap_full[order]

    angle = np.pi / 15
    rotation_matrix = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]],
        dtype=np.float64,
    )
    UE_pos[:, :2] = UE_pos[:, :2] @ rotation_matrix
    ap_pos[:, :2] = ap_pos[:, :2] @ rotation_matrix

    bounding_boxes_ap = np.array(
        [
            [[-5.5, -13.5], [-0.5, -1]],
            [[-13, -7.5], [-0.5, -1]],
            [[-5.5, -13.5], [-0.5, -1]],
            [[-13, -7.5], [-0.5, -1]],
        ],
        dtype=np.float64,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "UE_pos.npy", UE_pos)
    np.save(out_dir / "AP_pos.npy", ap_pos)
    np.save(out_dir / "H.npy", H)
    np.save(out_dir / "pow_per_ap_full.npy", pow_per_ap_full)
    np.save(out_dir / "timestamps.npy", timestamps)
    np.save(out_dir / "bounding_boxes_APs.npy", bounding_boxes_ap)

    print("H", H.shape, H.dtype)
    print("UE_pos", UE_pos.shape, UE_pos.dtype)
    print("timestamps", timestamps.shape, timestamps.dtype)
    print("AP_pos", ap_pos.shape, ap_pos.dtype)
    print("pow_per_ap_full", pow_per_ap_full.shape, pow_per_ap_full.dtype)
    print("negative time diffs", int(np.sum(timestamps[1:] - timestamps[:-1] < 0)))
    if args.expect_samples and H.shape[0] != args.expect_samples:
        raise RuntimeError(f"expected {args.expect_samples} samples, got {H.shape[0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
