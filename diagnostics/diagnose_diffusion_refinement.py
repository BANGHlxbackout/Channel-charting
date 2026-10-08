from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

from common_dichasus import DEFAULT_DATA1, load_data1
from common_metrics import affine_transform, write_json
from run_adp_fused import select_indices


def summarize_values(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        raise ValueError("cannot summarize empty values")
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "p5": float(np.percentile(values, 5)),
        "p25": float(np.percentile(values, 25)),
        "median": float(np.median(values)),
        "p75": float(np.percentile(values, 75)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def finite_corrcoef(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    if int(np.sum(mask)) < 2:
        return float("nan")
    if np.std(x[mask]) == 0.0 or np.std(y[mask]) == 0.0:
        return float("nan")
    return float(np.corrcoef(x[mask], y[mask])[0, 1])


def knn_indices(points: np.ndarray, k: int) -> np.ndarray:
    if k <= 0:
        raise ValueError("k must be positive")
    tree = cKDTree(np.asarray(points, dtype=np.float32))
    _, idx = tree.query(points, k=k + 1)
    return idx[:, 1:].astype(np.int32)


def row_overlap(left: np.ndarray, right: np.ndarray, k: int | None = None) -> np.ndarray:
    if left.shape[0] != right.shape[0]:
        raise ValueError("left and right must have the same number of rows")
    if k is None:
        k = min(left.shape[1], right.shape[1])
    if k <= 0:
        raise ValueError("k must be positive")
    left = left[:, :k]
    right = right[:, :k]
    overlap = np.empty(left.shape[0], dtype=np.float32)
    for i in range(left.shape[0]):
        overlap[i] = len(set(left[i]).intersection(right[i])) / float(k)
    return overlap


def median_neighbor_lengths(points: np.ndarray, neighbors: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    rows = np.arange(points.shape[0])[:, None]
    lengths = np.linalg.norm(points[rows] - points[neighbors], axis=2)
    return np.maximum(np.median(lengths, axis=1), eps).astype(np.float32)


def write_node_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("no rows to write")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_histogram(path: Path, values: np.ndarray, title: str, xlabel: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5), layout="constrained")
    ax.hist(values, bins=80, color="#4062bb", alpha=0.85)
    ax.axvline(0.0, color="#222222", linewidth=1.0)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_scatter(path: Path, x: np.ndarray, y: np.ndarray, title: str, xlabel: str, ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5.0), layout="constrained")
    ax.scatter(x, y, s=5, alpha=0.25, color="#0b6e4f", edgecolors="none")
    ax.axhline(0.0, color="#222222", linewidth=1.0)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_chart_delta(path: Path, positions: np.ndarray, error_delta: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 5.2), layout="constrained")
    vmax = float(np.percentile(np.abs(error_delta), 98))
    vmax = max(vmax, 1e-6)
    sc = ax.scatter(
        positions[:, 0],
        positions[:, 1],
        c=error_delta,
        s=4,
        cmap="coolwarm",
        vmin=-vmax,
        vmax=vmax,
        linewidths=0,
    )
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Refinement error delta by UE position")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    fig.colorbar(sc, ax=ax, label="refined error - initial error [m]")
    fig.savefig(path, dpi=220)
    plt.close(fig)


def top_examples(
    order_values: np.ndarray,
    indices: np.ndarray,
    init_errors: np.ndarray,
    refined_errors: np.ndarray,
    displacement: np.ndarray,
    count: int,
    reverse: bool,
) -> list[dict[str, float | int]]:
    order = np.argsort(order_values)
    if reverse:
        order = order[::-1]
    examples = []
    for local_idx in order[:count]:
        examples.append(
            {
                "local_id": int(local_idx),
                "dataset_index": int(indices[local_idx]),
                "error_delta_m": float(order_values[local_idx]),
                "init_error_m": float(init_errors[local_idx]),
                "refined_error_m": float(refined_errors[local_idx]),
                "aligned_displacement_m": float(displacement[local_idx]),
            }
        )
    return examples


def summarize_slice(
    mask: np.ndarray,
    init_errors: np.ndarray,
    refined_errors: np.ndarray,
    error_delta: np.ndarray,
    aligned_displacement: np.ndarray,
    compression_ratio: np.ndarray,
    true_overlap_delta: np.ndarray,
) -> dict[str, object]:
    count = int(np.sum(mask))
    if count == 0:
        return {"count": 0}
    return {
        "count": count,
        "fraction": float(np.mean(mask)),
        "init_error_m": summarize_values(init_errors[mask]),
        "refined_error_m": summarize_values(refined_errors[mask]),
        "error_delta_m": summarize_values(error_delta[mask]),
        "worsened_fraction": float(np.mean(error_delta[mask] > 0.0)),
        "strongly_worsened_fraction_delta_gt_0p05m": float(np.mean(error_delta[mask] > 0.05)),
        "aligned_displacement_m": summarize_values(aligned_displacement[mask]),
        "compression_ratio_refined_over_init": summarize_values(compression_ratio[mask]),
        "true_neighbor_overlap_delta": summarize_values(true_overlap_delta[mask]),
    }


def write_markdown_report(path: Path, summary: dict[str, object]) -> None:
    error = summary["error"]
    movement = summary["movement"]
    neighborhood = summary["neighborhood"]
    local_scale = summary["local_scale"]
    slices = summary["slices"]
    outputs = summary["outputs"]

    lines = [
        "# Diffusion Refinement Diagnostics",
        "",
        "This is a post-hoc diagnostic report. UE positions are used only for diagnosis, not for refinement.",
        "",
        "## Key Metrics",
        "",
        "| item | value |",
        "| --- | ---: |",
        f"| improved fraction | {error['improved_fraction']:.4f} |",
        f"| worsened fraction | {error['worsened_fraction']:.4f} |",
        f"| strongly worsened fraction (>0.05 m) | {error['strongly_worsened_fraction_delta_gt_0p05m']:.4f} |",
        f"| mean error delta (refined - init) | {error['delta_refined_minus_init']['mean']:.4f} m |",
        f"| median error delta | {error['delta_refined_minus_init']['median']:.4f} m |",
        f"| aligned displacement median | {movement['aligned_displacement_m']['median']:.4f} m |",
        f"| corr(displacement, error delta) | {movement['corr_aligned_displacement_error_delta']:.4f} |",
        f"| true-neighbor overlap init -> refined | {neighborhood['true_neighbor_overlap_init']['mean']:.4f} -> {neighborhood['true_neighbor_overlap_refined']['mean']:.4f} |",
        f"| physical-neighbor overlap init -> refined | {neighborhood['physical_neighbor_overlap_init']['mean']:.4f} -> {neighborhood['physical_neighbor_overlap_refined']['mean']:.4f} |",
        f"| local compression median refined/init | {local_scale['compression_ratio_refined_over_init']['median']:.4f} |",
        f"| compressed fraction (<0.90) | {local_scale['compressed_fraction_ratio_lt_0p90']:.4f} |",
        f"| heavily compressed fraction (<0.75) | {local_scale['heavily_compressed_fraction_ratio_lt_0p75']:.4f} |",
        "",
        "## Initial Error Slices",
        "",
        "| slice | count | fraction | worsened frac | strong worsened frac | mean delta m | median compression |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key, item in slices["by_initial_error"].items():
        lines.append(
            f"| {key} | {item['count']} | {item['fraction']:.4f} | "
            f"{item['worsened_fraction']:.4f} | {item['strongly_worsened_fraction_delta_gt_0p05m']:.4f} | "
            f"{item['error_delta_m']['mean']:.4f} | {item['compression_ratio_refined_over_init']['median']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Diagnostic Interpretation",
            "",
            "- Diffusion refinement is strongly beneficial overall: most samples improve and the worst initial outliers are repaired.",
            "- It is not a free improvement: a nontrivial fraction of samples worsens, especially points that were already accurate before refinement.",
            "- Local physical-neighbor lengths are compressed after refinement, which is direct evidence of a smoothing/compression effect.",
            "- Neighbor overlap with true local neighborhoods improves substantially, so the method is not simply harmful over-smoothing; it repairs topology for many points while damaging some already-good local geometry.",
            "- A useful next module should therefore be selective: preserve already-good local regions and target outlier/low-consistency regions, rather than adding more global attraction.",
            "",
            "## Output Files",
            "",
        ]
    )
    for name, output_path in outputs.items():
        lines.append(f"- `{name}`: `{output_path}`")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Post-hoc diagnostics for graph-harmonic diffusion refined channel charting."
    )
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA1)
    parser.add_argument("--distance-run", type=Path, default=Path("results/adp_old_fused/stride1_n20997_seed10"))
    parser.add_argument("--init-embedding", type=Path, default=Path("results/adp_old_siamese/stride1_n20997_seed10/embedding.npz"))
    parser.add_argument(
        "--refined-embedding",
        type=Path,
        default=Path("results/diffusion_refined_charting/stride1_n20997_k128_lambda0.15_seed10/embedding.npz"),
    )
    parser.add_argument("--out-dir", type=Path, default=Path("results/diffusion_refinement_diagnostics"))
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--neighbor-k", type=int, default=32)
    parser.add_argument("--physical-k", type=int, default=128)
    parser.add_argument("--top-examples", type=int, default=25)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    data = load_data1(args.data_path, seed=args.seed)
    indices = select_indices(len(data.h_by_array), args.stride, args.max_samples)
    positions = data.ue_pos[indices]

    distance_data = np.load(args.distance_run / "distance_data.npz")
    distance_indices = distance_data["indices"]
    if len(distance_indices) != len(indices) or not np.array_equal(distance_indices, indices):
        raise ValueError("distance-run indices do not match requested data selection")

    init_data = np.load(args.init_embedding)
    refined_data = np.load(args.refined_embedding)
    for name, embedding_data in [("init", init_data), ("refined", refined_data)]:
        embedding_indices = embedding_data["indices"]
        if len(embedding_indices) != len(indices) or not np.array_equal(embedding_indices, indices):
            raise ValueError(f"{name} embedding indices do not match requested data selection")

    y0 = init_data["embedding"].astype(np.float32)
    z = refined_data["embedding"].astype(np.float32)

    aligned_y0 = affine_transform(positions, y0).astype(np.float32)
    aligned_z = affine_transform(positions, z).astype(np.float32)
    init_errors = np.linalg.norm(aligned_y0 - positions, axis=1)
    refined_errors = np.linalg.norm(aligned_z - positions, axis=1)
    error_delta = refined_errors - init_errors
    aligned_displacement = np.linalg.norm(aligned_z - aligned_y0, axis=1)
    raw_displacement = np.linalg.norm(z - y0, axis=1)

    physical_neighbors_path = args.distance_run / f"geodesic_top512.npy"
    if not physical_neighbors_path.exists():
        raise FileNotFoundError(f"missing cached physical neighbors: {physical_neighbors_path}")
    physical_neighbors = np.load(physical_neighbors_path)[:, : args.physical_k].astype(np.int32)

    true_neighbors = knn_indices(positions, args.neighbor_k)
    init_neighbors = knn_indices(y0, args.neighbor_k)
    refined_neighbors = knn_indices(z, args.neighbor_k)
    physical_eval_neighbors = physical_neighbors[:, : args.neighbor_k]

    true_overlap_init = row_overlap(true_neighbors, init_neighbors, args.neighbor_k)
    true_overlap_refined = row_overlap(true_neighbors, refined_neighbors, args.neighbor_k)
    physical_overlap_init = row_overlap(physical_eval_neighbors, init_neighbors, args.neighbor_k)
    physical_overlap_refined = row_overlap(physical_eval_neighbors, refined_neighbors, args.neighbor_k)

    true_local_scale = median_neighbor_lengths(positions, physical_eval_neighbors)
    init_local_scale = median_neighbor_lengths(aligned_y0, physical_eval_neighbors)
    refined_local_scale = median_neighbor_lengths(aligned_z, physical_eval_neighbors)
    scale_ratio_init = init_local_scale / true_local_scale
    scale_ratio_refined = refined_local_scale / true_local_scale
    scale_ratio_delta = scale_ratio_refined - scale_ratio_init
    compression_ratio = refined_local_scale / np.maximum(init_local_scale, 1e-9)
    true_overlap_delta = true_overlap_refined - true_overlap_init

    improved = error_delta < 0.0
    worsened = error_delta > 0.0
    strongly_worsened = error_delta > 0.05
    strongly_improved = error_delta < -0.05
    compressed = compression_ratio < 0.9
    heavily_compressed = compression_ratio < 0.75

    node_rows: list[dict[str, object]] = []
    for i in range(len(indices)):
        node_rows.append(
            {
                "local_id": int(i),
                "dataset_index": int(indices[i]),
                "x_m": float(positions[i, 0]),
                "y_m": float(positions[i, 1]),
                "init_error_m": float(init_errors[i]),
                "refined_error_m": float(refined_errors[i]),
                "error_delta_m": float(error_delta[i]),
                "aligned_displacement_m": float(aligned_displacement[i]),
                "raw_displacement": float(raw_displacement[i]),
                "true_neighbor_overlap_init": float(true_overlap_init[i]),
                "true_neighbor_overlap_refined": float(true_overlap_refined[i]),
                "true_neighbor_overlap_delta": float(true_overlap_refined[i] - true_overlap_init[i]),
                "physical_neighbor_overlap_init": float(physical_overlap_init[i]),
                "physical_neighbor_overlap_refined": float(physical_overlap_refined[i]),
                "physical_neighbor_overlap_delta": float(physical_overlap_refined[i] - physical_overlap_init[i]),
                "true_local_scale_m": float(true_local_scale[i]),
                "init_local_scale_m": float(init_local_scale[i]),
                "refined_local_scale_m": float(refined_local_scale[i]),
                "scale_ratio_init": float(scale_ratio_init[i]),
                "scale_ratio_refined": float(scale_ratio_refined[i]),
                "scale_ratio_delta": float(scale_ratio_delta[i]),
                "compression_ratio_refined_over_init": float(compression_ratio[i]),
            }
        )

    out_dir = args.out_dir / f"stride{args.stride}_n{len(indices)}_seed{args.seed}_k{args.neighbor_k}"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_node_csv(out_dir / "node_diagnostics.csv", node_rows)

    save_histogram(out_dir / "error_delta_hist.png", error_delta, "Error delta after diffusion refinement", "refined error - initial error [m]")
    save_scatter(
        out_dir / "displacement_vs_error_delta.png",
        aligned_displacement,
        error_delta,
        "Movement vs error delta",
        "aligned displacement ||Z - Y0|| [m]",
        "refined error - initial error [m]",
    )
    save_scatter(
        out_dir / "compression_vs_error_delta.png",
        compression_ratio,
        error_delta,
        "Local physical-neighbor compression vs error delta",
        "median physical-neighbor length ratio refined / initial",
        "refined error - initial error [m]",
    )
    save_chart_delta(out_dir / "ue_position_error_delta.png", positions, error_delta)

    summary = {
        "diagnostic_only_uses_positions": True,
        "positions_used_for_method": False,
        "dataset": str(args.data_path),
        "distance_run": str(args.distance_run),
        "init_embedding": str(args.init_embedding),
        "refined_embedding": str(args.refined_embedding),
        "num_samples": int(len(indices)),
        "seed": int(args.seed),
        "neighbor_k": int(args.neighbor_k),
        "physical_k": int(args.physical_k),
        "error": {
            "init": summarize_values(init_errors),
            "refined": summarize_values(refined_errors),
            "delta_refined_minus_init": summarize_values(error_delta),
            "improved_fraction": float(np.mean(improved)),
            "worsened_fraction": float(np.mean(worsened)),
            "strongly_improved_fraction_delta_lt_minus_0p05m": float(np.mean(strongly_improved)),
            "strongly_worsened_fraction_delta_gt_0p05m": float(np.mean(strongly_worsened)),
        },
        "movement": {
            "aligned_displacement_m": summarize_values(aligned_displacement),
            "raw_displacement": summarize_values(raw_displacement),
            "corr_aligned_displacement_error_delta": finite_corrcoef(aligned_displacement, error_delta),
            "corr_raw_displacement_error_delta": finite_corrcoef(raw_displacement, error_delta),
        },
        "neighborhood": {
            "true_neighbor_overlap_init": summarize_values(true_overlap_init),
            "true_neighbor_overlap_refined": summarize_values(true_overlap_refined),
            "true_neighbor_overlap_delta": summarize_values(true_overlap_refined - true_overlap_init),
            "physical_neighbor_overlap_init": summarize_values(physical_overlap_init),
            "physical_neighbor_overlap_refined": summarize_values(physical_overlap_refined),
            "physical_neighbor_overlap_delta": summarize_values(physical_overlap_refined - physical_overlap_init),
        },
        "local_scale": {
            "scale_ratio_init_to_true": summarize_values(scale_ratio_init),
            "scale_ratio_refined_to_true": summarize_values(scale_ratio_refined),
            "scale_ratio_delta": summarize_values(scale_ratio_delta),
            "compression_ratio_refined_over_init": summarize_values(compression_ratio),
            "compressed_fraction_ratio_lt_0p90": float(np.mean(compressed)),
            "heavily_compressed_fraction_ratio_lt_0p75": float(np.mean(heavily_compressed)),
            "corr_compression_ratio_error_delta": finite_corrcoef(compression_ratio, error_delta),
        },
        "slices": {
            "worsened": {
                "count": int(np.sum(worsened)),
                "aligned_displacement_m": summarize_values(aligned_displacement[worsened]) if np.any(worsened) else None,
                "compression_ratio_refined_over_init": summarize_values(compression_ratio[worsened]) if np.any(worsened) else None,
                "true_neighbor_overlap_delta": summarize_values((true_overlap_refined - true_overlap_init)[worsened]) if np.any(worsened) else None,
            },
            "improved": {
                "count": int(np.sum(improved)),
                "aligned_displacement_m": summarize_values(aligned_displacement[improved]) if np.any(improved) else None,
                "compression_ratio_refined_over_init": summarize_values(compression_ratio[improved]) if np.any(improved) else None,
                "true_neighbor_overlap_delta": summarize_values((true_overlap_refined - true_overlap_init)[improved]) if np.any(improved) else None,
            },
            "by_initial_error": {
                "init_error_lt_0p2m": summarize_slice(
                    init_errors < 0.2,
                    init_errors,
                    refined_errors,
                    error_delta,
                    aligned_displacement,
                    compression_ratio,
                    true_overlap_delta,
                ),
                "init_error_0p2m_to_0p5m": summarize_slice(
                    (init_errors >= 0.2) & (init_errors < 0.5),
                    init_errors,
                    refined_errors,
                    error_delta,
                    aligned_displacement,
                    compression_ratio,
                    true_overlap_delta,
                ),
                "init_error_0p5m_to_1p0m": summarize_slice(
                    (init_errors >= 0.5) & (init_errors < 1.0),
                    init_errors,
                    refined_errors,
                    error_delta,
                    aligned_displacement,
                    compression_ratio,
                    true_overlap_delta,
                ),
                "init_error_ge_1p0m": summarize_slice(
                    init_errors >= 1.0,
                    init_errors,
                    refined_errors,
                    error_delta,
                    aligned_displacement,
                    compression_ratio,
                    true_overlap_delta,
                ),
            },
        },
        "top_worsened_examples": top_examples(
            error_delta,
            indices,
            init_errors,
            refined_errors,
            aligned_displacement,
            args.top_examples,
            reverse=True,
        ),
        "top_improved_examples": top_examples(
            error_delta,
            indices,
            init_errors,
            refined_errors,
            aligned_displacement,
            args.top_examples,
            reverse=False,
        ),
        "outputs": {
            "node_csv": str(out_dir / "node_diagnostics.csv"),
            "error_delta_hist": str(out_dir / "error_delta_hist.png"),
            "displacement_vs_error_delta": str(out_dir / "displacement_vs_error_delta.png"),
            "compression_vs_error_delta": str(out_dir / "compression_vs_error_delta.png"),
            "ue_position_error_delta": str(out_dir / "ue_position_error_delta.png"),
            "markdown_report": str(out_dir / "diagnostic_summary.md"),
        },
    }
    write_json(out_dir / "diagnostics.json", summary)
    write_markdown_report(out_dir / "diagnostic_summary.md", summary)
    print("DIAGNOSTICS", json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
