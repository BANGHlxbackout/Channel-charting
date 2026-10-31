from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HELDOUT_DIR = ROOT / "held-out"
for path in (ROOT, HELDOUT_DIR):
    path_text = str(path)
    if path_text not in sys.path:
        sys.path.insert(0, path_text)

from common_heldout import apply_affine_transform, fit_affine_transform


VARIANTS = ("y0_only", "q_only", "y0_q")


def load_npz(path: Path) -> np.lib.npyio.NpzFile:
    if not path.exists():
        raise FileNotFoundError(path)
    return np.load(path)


def colors_from_positions(reference: np.ndarray) -> np.ndarray:
    colors = np.zeros((len(reference), 3), dtype=np.float32)
    x_range = max(float(np.ptp(reference[:, 0])), 1e-12)
    y_range = max(float(np.ptp(reference[:, 1])), 1e-12)
    colors[:, 0] = (reference[:, 0] - reference[:, 0].min()) / x_range
    colors[:, 1] = (reference[:, 1] - reference[:, 1].min()) / y_range
    colors[:, 2] = 0.25
    return colors


def plot_aligned_chart(
    path: Path,
    reference: np.ndarray,
    aligned_embedding: np.ndarray,
    title: str,
) -> None:
    colors = colors_from_positions(reference)
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
    parser = argparse.ArgumentParser(
        description="Plot CF-HRD residual-student train/test charts with train-fitted affine alignment."
    )
    parser.add_argument(
        "--residual-student-run",
        type=Path,
        default=Path("results/CF_HRD/seed10/residual_student"),
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
    args = parser.parse_args()

    fcf_train = load_npz(args.fcf_train)
    fcf_test = load_npz(args.fcf_test)
    train_positions = fcf_train["positions"].astype(np.float32)
    test_positions = fcf_test["positions"].astype(np.float32)
    train_indices = fcf_train["indices"].astype(np.int64)
    test_indices = fcf_test["indices"].astype(np.int64)

    for variant in VARIANTS:
        train_data = load_npz(args.residual_student_run / f"{variant}_embedding_train.npz")
        test_data = load_npz(args.residual_student_run / f"{variant}_embedding_test.npz")
        if not np.array_equal(train_data["indices"].astype(np.int64), train_indices):
            raise ValueError(f"{variant} train indices do not match FCF train")
        if not np.array_equal(test_data["indices"].astype(np.int64), test_indices):
            raise ValueError(f"{variant} test indices do not match FCF test")

        train_embedding = train_data["embedding"].astype(np.float32)
        test_embedding = test_data["embedding"].astype(np.float32)
        transform = fit_affine_transform(train_positions, train_embedding)
        aligned_train = apply_affine_transform(train_embedding, transform)
        aligned_test = apply_affine_transform(test_embedding, transform)

        plot_aligned_chart(
            args.residual_student_run / f"{variant}_chart_train.png",
            train_positions,
            aligned_train,
            f"CF-HRD {variant} Train",
        )
        plot_aligned_chart(
            args.residual_student_run / f"{variant}_chart_test.png",
            test_positions,
            aligned_test,
            f"CF-HRD {variant} Held-out Test",
        )


if __name__ == "__main__":
    main()
