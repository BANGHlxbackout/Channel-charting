"""
Check whether the measured-indoor data matches the paper P2 protocol.

This script is intentionally pre-training only. P2 should not be trained or
reported against Table IV until these AP-pair statistics match Fig. 6/Table IV.
"""
import argparse
from pathlib import Path

import numpy as np

from utils.loss_helpers import count_wrong_triplets


PAPER_NUM_SAMPLES = 16681
PAPER_FALSE_RATIO = {0: 0.275, 3: 0.050, 6: 0.018}
PAPER_AVG_PAIRS = {0: 1.70, 3: 0.44, 6: 0.10}


def load_data(data_path, W):
    required = ["AP_pos.npy", "H.npy", "UE_pos.npy", "timestamps.npy", "bounding_boxes_APs.npy"]
    missing = [name for name in required if not (data_path / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing files in {data_path}: {', '.join(missing)}")

    ap_pos = np.load(data_path / "AP_pos.npy")[:, :2].astype(np.float32)
    H = np.load(data_path / "H.npy").astype(np.complex64)[:, :, :W]
    UE_pos = np.load(data_path / "UE_pos.npy")[:, :2].astype(np.float32)
    timestamps = np.load(data_path / "timestamps.npy").astype(np.float32)
    pow_path = data_path / "pow_per_ap_full.npy"
    pow_per_ap = np.load(pow_path).astype(np.float32) if pow_path.exists() else None
    return ap_pos, H, UE_pos, timestamps, pow_per_ap


def compute_pow_per_ap(H, A):
    Mr = int(H.shape[1] / A)
    temp = np.reshape(H, (H.shape[0], A, Mr * H.shape[-1]))
    return 20 * np.log10(np.linalg.norm(temp, ord=2, axis=-1))


def pair_stats(pow_per_ap, UE_pos, ap_pos, M_p, P_thresh):
    err_per_u, num_per_u = count_wrong_triplets(pow_per_ap, UE_pos, ap_pos, M_p, P_thresh)
    num_pairs = int(np.sum(num_per_u))
    num_wrong = int(np.sum(err_per_u))
    false_ratio = np.nan if num_pairs == 0 else num_wrong / num_pairs
    return {
        "users_with_pairs": int(np.count_nonzero(num_per_u)),
        "num_pairs": num_pairs,
        "num_wrong": num_wrong,
        "false_ratio": float(false_ratio),
        "avg_pairs_per_u": float(np.mean(num_per_u)),
    }


def print_stats(title, pow_per_ap, UE_pos, ap_pos, P_thresh):
    print(title)
    rows = {}
    for mp in (0, 3, 6):
        stats = pair_stats(pow_per_ap, UE_pos, ap_pos, mp, P_thresh)
        rows[mp] = stats
        print(
            f"  M_p={mp}: false_ratio={stats['false_ratio']:.4f}, "
            f"avg_pairs_per_u={stats['avg_pairs_per_u']:.4f}, "
            f"users={stats['users_with_pairs']}, pairs={stats['num_pairs']}, "
            f"wrong={stats['num_wrong']}"
        )
    return rows


def train_split(num_samples, test_to_all_ratio, seed):
    np.random.seed(seed)
    n_te = round(num_samples * test_to_all_ratio)
    te_UEs = np.random.choice(num_samples, n_te, False)
    all_UEs = np.arange(num_samples)
    tr_UEs = all_UEs[np.isin(all_UEs, te_UEs, invert=True)]
    return tr_UEs, te_UEs


def strict_check(num_samples, rows, false_tol, avg_pairs_tol):
    failed = num_samples != PAPER_NUM_SAMPLES
    if failed:
        print(f"FAIL: expected {PAPER_NUM_SAMPLES} samples, got {num_samples}")

    for mp, stats in rows.items():
        false_delta = abs(stats["false_ratio"] - PAPER_FALSE_RATIO[mp])
        avg_delta = abs(stats["avg_pairs_per_u"] - PAPER_AVG_PAIRS[mp])
        if false_delta > false_tol:
            print(
                f"FAIL: M_p={mp} false_ratio expected "
                f"{PAPER_FALSE_RATIO[mp]:.4f}, got {stats['false_ratio']:.4f}"
            )
            failed = True
        if avg_delta > avg_pairs_tol:
            print(
                f"FAIL: M_p={mp} avg_pairs_per_u expected "
                f"{PAPER_AVG_PAIRS[mp]:.4f}, got {stats['avg_pairs_per_u']:.4f}"
            )
            failed = True
    return failed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default="data1")
    parser.add_argument("--W", type=int, default=13)
    parser.add_argument("--p-thresh", type=float, default=16.2)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--test-to-all-ratio", type=float, default=0.2)
    parser.add_argument("--false-ratio-tol", type=float, default=0.03)
    parser.add_argument("--avg-pairs-tol", type=float, default=0.15)
    parser.add_argument("--no-strict", action="store_true")
    args = parser.parse_args()

    data_path = Path(args.data_path)
    ap_pos, H, UE_pos, timestamps, pow_per_ap = load_data(data_path, args.W)
    if pow_per_ap is None:
        pow_per_ap = compute_pow_per_ap(H, ap_pos.shape[0])
        print("Power source: 13-tap H.npy fallback")
    else:
        print("Power source:", data_path / "pow_per_ap_full.npy")

    print("Data path:", data_path)
    print("Shapes:", "H", H.shape, "UE_pos", UE_pos.shape, "AP_pos", ap_pos.shape, "timestamps", timestamps.shape)
    print("Paper targets:")
    print("  false_ratio M_p=0/3/6:", PAPER_FALSE_RATIO)
    print("  avg_pairs_per_u M_p=0/3/6:", PAPER_AVG_PAIRS)

    full_rows = print_stats("Full-dataset AP-pair stats:", pow_per_ap, UE_pos, ap_pos, args.p_thresh)

    tr_UEs, te_UEs = train_split(H.shape[0], args.test_to_all_ratio, args.seed)
    print("Train/test sizes:", len(tr_UEs), len(te_UEs), "seed:", args.seed)
    print_stats("Train-split AP-pair stats:", pow_per_ap[tr_UEs], UE_pos[tr_UEs], ap_pos, args.p_thresh)

    if not args.no_strict:
        failed = strict_check(H.shape[0], full_rows, args.false_ratio_tol, args.avg_pairs_tol)
        if failed:
            raise SystemExit(2)
        print("PASS: data matches the paper P2 AP-pair statistics within tolerance.")


if __name__ == "__main__":
    main()
