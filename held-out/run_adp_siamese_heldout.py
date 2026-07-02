from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from common_dichasus import DEFAULT_DATA1, load_data1
from common_heldout import (
    apply_affine_transform,
    deterministic_subset_indices,
    fit_affine_transform,
    heldout_stage_dir,
    localization_error_summary_aligned,
)
from common_metrics import chart_metrics, write_json
from run_adp_siamese import (
    ForwardChartingFunction,
    PairDataset,
    predict,
    siamese_loss,
    train_schedule,
)


def prepare_heldout_arrays(
    h: np.ndarray,
    positions: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return h[train_idx], h[test_idx], positions[train_idx], positions[test_idx]


def validate_distance_run_split(
    distance_run: Path,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> np.lib.npyio.NpzFile:
    saved = np.load(distance_run / "distance_data.npz")
    if not np.array_equal(saved["indices"], train_idx) or not np.array_equal(
        saved["test_indices"],
        test_idx,
    ):
        raise ValueError("distance-run split does not match requested split")
    return saved


def evaluate_heldout_embeddings(
    pos_train: np.ndarray,
    y_train: np.ndarray,
    pos_test: np.ndarray,
    y_test: np.ndarray,
    seed: int,
    metric_subset_fraction: float,
) -> dict:
    train_subset = deterministic_subset_indices(
        len(pos_train),
        metric_subset_fraction,
        seed,
    )
    transform = fit_affine_transform(pos_train, y_train)
    aligned_train = apply_affine_transform(y_train, transform)
    aligned_test = apply_affine_transform(y_test, transform)
    return {
        "train_metric_subset": train_subset,
        "train_metrics": chart_metrics(
            pos_train[train_subset],
            y_train[train_subset],
        ),
        "train_localization_errors": localization_error_summary_aligned(
            pos_train,
            aligned_train,
        ),
        "test_metrics": chart_metrics(pos_test, y_test),
        "test_localization_errors": localization_error_summary_aligned(
            pos_test,
            aligned_test,
        ),
        "aligned_train": aligned_train,
        "aligned_test": aligned_test,
        "affine_transform": transform,
    }


def train_fcf(
    h_train: np.ndarray,
    geodesic_train: np.ndarray,
    seed: int,
    device: torch.device,
    session_limit: int | None,
    samples_per_session: int | None,
) -> tuple[ForwardChartingFunction, list[dict]]:
    model = ForwardChartingFunction(input_shape=h_train.shape[1:]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), eps=1e-7)
    margin = float(np.quantile(geodesic_train, 0.01))
    schedule = train_schedule()
    if session_limit is not None:
        schedule = schedule[:session_limit]

    history = []
    for session_idx, (lr, batch_size, requested_samples) in enumerate(
        schedule,
        start=1,
    ):
        if samples_per_session is not None:
            requested_samples = samples_per_session
        effective_samples = (requested_samples // batch_size) * batch_size
        if effective_samples == 0:
            raise ValueError(
                f"samples_per_session must be at least batch size {batch_size}"
            )
        for group in optimizer.param_groups:
            group["lr"] = lr
        dataset = PairDataset(
            h_train,
            geodesic_train,
            samples=effective_samples,
            seed=seed + session_idx,
        )
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=False,
        )
        model.train()
        running_loss = 0.0
        for step, (left_h, right_h, target) in enumerate(loader, start=1):
            left_h = left_h.to(device)
            right_h = right_h.to(device)
            target = target.to(device)
            optimizer.zero_grad(set_to_none=True)
            left_y = model(left_h)
            right_y = model(right_h)
            loss = siamese_loss(left_y, right_y, target, margin)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item())
            if step % 20 == 0 or step == len(loader):
                print(
                    f"session {session_idx}/{len(schedule)} step {step}/{len(loader)} "
                    f"lr={lr:g} bs={batch_size} loss={running_loss / step:.6f}",
                    flush=True,
                )
        history.append(
            {
                "session": session_idx,
                "lr": lr,
                "batch_size": batch_size,
                "samples": effective_samples,
                "requested_samples": requested_samples,
                "loss": running_loss / max(1, len(loader)),
            }
        )
    return model, history


def plot_aligned_chart(
    path: Path,
    reference: np.ndarray,
    aligned_embedding: np.ndarray,
    title: str,
) -> None:
    colors = np.zeros((len(reference), 3), dtype=np.float32)
    colors[:, 0] = (reference[:, 0] - reference[:, 0].min()) / np.ptp(reference[:, 0])
    colors[:, 1] = (reference[:, 1] - reference[:, 1].min()) / np.ptp(reference[:, 1])
    colors[:, 2] = 0.25

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].scatter(reference[:, 0], reference[:, 1], c=colors, s=5, linewidths=0)
    axes[0].set_title("Ground Truth")
    axes[0].axis("equal")
    axes[0].grid(True, alpha=0.25)
    axes[1].scatter(
        aligned_embedding[:, 0],
        aligned_embedding[:, 1],
        c=colors,
        s=5,
        linewidths=0,
    )
    axes[1].set_title(title)
    axes[1].axis("equal")
    axes[1].grid(True, alpha=0.25)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train held-out Siamese FCF.")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA1)
    parser.add_argument("--distance-run", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("results/heldout"))
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--session-limit", type=int, default=None)
    parser.add_argument("--samples-per-session", type=int, default=None)
    parser.add_argument("--metric-subset-fraction", type=float, default=0.2)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    started = time.time()
    device = torch.device(args.device)
    data = load_data1(args.data_path, seed=args.seed)
    validate_distance_run_split(args.distance_run, data.train_idx, data.test_idx)
    h_train, h_test, pos_train, pos_test = prepare_heldout_arrays(
        data.h_by_array,
        data.ue_pos,
        data.train_idx,
        data.test_idx,
    )

    geodesic_train = np.load(args.distance_run / "geodesic_train.npy").astype(np.float32)
    expected_shape = (len(data.train_idx), len(data.train_idx))
    if geodesic_train.shape != expected_shape:
        raise ValueError(f"expected train geodesic shape {expected_shape}, got {geodesic_train.shape}")
    if not np.all(np.isfinite(geodesic_train)):
        raise ValueError("train geodesic contains non-finite values")

    out_dir = heldout_stage_dir(args.out_dir, args.seed, "adp_siamese")
    out_dir.mkdir(parents=True, exist_ok=True)

    model, history = train_fcf(
        h_train,
        geodesic_train,
        seed=args.seed,
        device=device,
        session_limit=args.session_limit,
        samples_per_session=args.samples_per_session,
    )
    torch.save(model.state_dict(), out_dir / "model.pt")

    y_train = predict(model, h_train, device)
    y_test = predict(model, h_test, device)
    evaluation = evaluate_heldout_embeddings(
        pos_train,
        y_train,
        pos_test,
        y_test,
        seed=args.seed,
        metric_subset_fraction=args.metric_subset_fraction,
    )

    np.savez(
        out_dir / "embedding_train.npz",
        indices=data.train_idx,
        positions=pos_train,
        embedding=y_train,
    )
    np.savez(
        out_dir / "embedding_test.npz",
        indices=data.test_idx,
        positions=pos_test,
        embedding=y_test,
    )

    result = {
        "protocol": "heldout_fcf",
        "dataset": str(args.data_path),
        "distance_run": str(args.distance_run),
        "seed": int(args.seed),
        "num_train": int(len(data.train_idx)),
        "num_test": int(len(data.test_idx)),
        "history": history,
        "metric_subset_fraction": float(args.metric_subset_fraction),
        "train_metric_subset_size": int(len(evaluation["train_metric_subset"])),
        "train_metrics": evaluation["train_metrics"],
        "train_localization_errors": evaluation["train_localization_errors"],
        "test_metrics": evaluation["test_metrics"],
        "test_localization_errors": evaluation["test_localization_errors"],
        "seed10_historical_visibility": bool(args.seed == 10),
        "elapsed_s": float(time.time() - started),
    }
    write_json(out_dir / "result.json", result)

    plot_aligned_chart(
        out_dir / "chart_train.png",
        pos_train,
        evaluation["aligned_train"],
        f"Held-out FCF Train\nCT={result['train_metrics']['ct']:.3f}, TW={result['train_metrics']['tw']:.3f}",
    )
    plot_aligned_chart(
        out_dir / "chart_test.png",
        pos_test,
        evaluation["aligned_test"],
        f"Held-out FCF Test\nCT={result['test_metrics']['ct']:.3f}, TW={result['test_metrics']['tw']:.3f}",
    )
    print("RESULT", json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
