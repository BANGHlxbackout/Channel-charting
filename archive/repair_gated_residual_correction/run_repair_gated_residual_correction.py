from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from common_dichasus import DEFAULT_DATA1, load_data1
from common_metrics import affine_transform, chart_metrics, write_json
from diagnostics.diagnose_diffusion_refinement import summarize_values
from diagnostics.diagnose_local_consistency_proxies import (
    graph_smoothness_residual,
    median_physical_scale_stress,
    robust_zscore,
)
from run_adp_fused import select_indices
from run_diffusion_refined_charting import build_local_scaled_graph


def hard_top_gate(scores: np.ndarray, top_fraction: float) -> np.ndarray:
    if not 0.0 < top_fraction <= 1.0:
        raise ValueError("top_fraction must be in (0, 1]")
    scores = np.asarray(scores, dtype=np.float32)
    count = max(1, int(round(scores.shape[0] * top_fraction)))
    order = np.argsort(scores)[::-1]
    gate = np.zeros(scores.shape[0], dtype=np.float32)
    gate[order[:count]] = 1.0
    return gate


def floor_top_gate(scores: np.ndarray, top_fraction: float, floor: float) -> np.ndarray:
    if not 0.0 <= floor <= 1.0:
        raise ValueError("floor must be in [0, 1]")
    gate = hard_top_gate(scores, top_fraction)
    return np.maximum(gate, np.float32(floor)).astype(np.float32)


def soft_sigmoid_gate(scores: np.ndarray, top_fraction: float, alpha: float) -> np.ndarray:
    if alpha <= 0.0:
        raise ValueError("alpha must be positive")
    scores = np.asarray(scores, dtype=np.float32)
    threshold = float(np.quantile(scores, 1.0 - top_fraction))
    logits = np.clip(alpha * (scores - threshold), -60.0, 60.0)
    return (1.0 / (1.0 + np.exp(-logits))).astype(np.float32)


def make_global_gate(n: int, gamma: float) -> np.ndarray:
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be in [0, 1]")
    return np.full(n, gamma, dtype=np.float32)


def make_random_gate(gate: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    shuffled = np.asarray(gate, dtype=np.float32).copy()
    rng.shuffle(shuffled)
    return shuffled


def apply_residual_gate(y0: np.ndarray, z_base: np.ndarray, gate: np.ndarray) -> np.ndarray:
    y0 = np.asarray(y0, dtype=np.float32)
    z_base = np.asarray(z_base, dtype=np.float32)
    gate = np.asarray(gate, dtype=np.float32)
    if y0.shape != z_base.shape:
        raise ValueError("y0 and z_base must have the same shape")
    if gate.shape != (y0.shape[0],):
        raise ValueError("gate must be one-dimensional with one value per row")
    return (y0 + gate[:, None] * (z_base - y0)).astype(np.float32)


def retained_repair_ratio(
    init_error: np.ndarray,
    baseline_error: np.ndarray,
    candidate_error: np.ndarray,
    outlier_mask: np.ndarray,
) -> float:
    outlier_mask = np.asarray(outlier_mask, dtype=bool)
    if not np.any(outlier_mask):
        return float("nan")
    baseline_improvement = float(np.mean(init_error[outlier_mask] - baseline_error[outlier_mask]))
    candidate_improvement = float(np.mean(init_error[outlier_mask] - candidate_error[outlier_mask]))
    if abs(baseline_improvement) < 1e-12:
        return float("nan")
    return float(candidate_improvement / baseline_improvement)


def write_rows_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("no rows to write")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def localization_errors(reference: np.ndarray, embedding: np.ndarray) -> np.ndarray:
    aligned = affine_transform(reference, embedding).astype(np.float32)
    return np.linalg.norm(aligned - reference, axis=1)


def slice_summary(
    init_error: np.ndarray,
    candidate_error: np.ndarray,
    mask: np.ndarray,
    strong_threshold: float,
) -> dict[str, object]:
    count = int(np.sum(mask))
    if count == 0:
        return {"count": 0}
    delta = candidate_error[mask] - init_error[mask]
    return {
        "count": count,
        "fraction": float(np.mean(mask)),
        "mean_delta_m": float(np.mean(delta)),
        "median_delta_m": float(np.median(delta)),
        "worsened_fraction": float(np.mean(delta > 0.0)),
        "strong_worsened_fraction": float(np.mean(delta > strong_threshold)),
        "error_delta_m": summarize_values(delta),
    }


def evaluate_embedding(
    name: str,
    embedding: np.ndarray,
    positions: np.ndarray,
    init_error: np.ndarray,
    baseline_error: np.ndarray,
    metric_subset: np.ndarray,
    good_initial: np.ndarray,
    outlier_initial: np.ndarray,
    strong_threshold: float,
    gate: np.ndarray | None = None,
) -> dict[str, object]:
    errors = localization_errors(positions, embedding)
    delta = errors - init_error
    result = {
        "name": name,
        "localization_errors_after_affine": {
            "mean_m": float(np.mean(errors)),
            "median_m": float(np.median(errors)),
            "p95_m": float(np.percentile(errors, 95)),
            "max_m": float(np.max(errors)),
        },
        "metrics": chart_metrics(positions[metric_subset], embedding[metric_subset]),
        "overall": {
            "improved_fraction": float(np.mean(delta < 0.0)),
            "worsened_fraction": float(np.mean(delta > 0.0)),
            "strong_worsened_fraction": float(np.mean(delta > strong_threshold)),
            "mean_delta_m": float(np.mean(delta)),
        },
        "good_initial_slice": slice_summary(init_error, errors, good_initial, strong_threshold),
        "outlier_initial_slice": {
            **slice_summary(init_error, errors, outlier_initial, strong_threshold),
            "retained_repair_ratio_vs_baseline": retained_repair_ratio(
                init_error,
                baseline_error,
                errors,
                outlier_initial,
            ),
        },
    }
    if gate is not None:
        result["gate"] = {
            "mean": float(np.mean(gate)),
            "min": float(np.min(gate)),
            "p25": float(np.percentile(gate, 25)),
            "median": float(np.median(gate)),
            "p75": float(np.percentile(gate, 75)),
            "max": float(np.max(gate)),
            "active_fraction_gt_0p5": float(np.mean(gate > 0.5)),
        }
    return result


def result_row(result: dict[str, object]) -> dict[str, object]:
    errors = result["localization_errors_after_affine"]
    metrics = result["metrics"]
    good = result["good_initial_slice"]
    outlier = result["outlier_initial_slice"]
    overall = result["overall"]
    gate = result.get("gate", {})
    return {
        "name": result["name"],
        "mean_m": errors["mean_m"],
        "median_m": errors["median_m"],
        "p95_m": errors["p95_m"],
        "max_m": errors["max_m"],
        "ct": metrics["ct"],
        "tw": metrics["tw"],
        "ks": metrics["ks"],
        "overall_improved_fraction": overall["improved_fraction"],
        "overall_strong_worsened_fraction": overall["strong_worsened_fraction"],
        "good_worsened_fraction": good.get("worsened_fraction"),
        "good_strong_worsened_fraction": good.get("strong_worsened_fraction"),
        "good_mean_delta_m": good.get("mean_delta_m"),
        "outlier_mean_delta_m": outlier.get("mean_delta_m"),
        "outlier_retained_repair_ratio": outlier.get("retained_repair_ratio_vs_baseline"),
        "gate_mean": gate.get("mean"),
        "gate_active_fraction_gt_0p5": gate.get("active_fraction_gt_0p5"),
    }


def write_markdown(path: Path, summary: dict[str, object]) -> None:
    rows = summary["result_rows"]
    best = summary["best_methods"]
    lines = [
        "# Repair-Gated Residual Correction",
        "",
        "Residual correction uses `Z_gate = Y0 + g_i (Z_base - Y0)` and does not solve a new graph system.",
        "",
        "## Best Methods",
        "",
        f"- Best overall mean: `{best['best_mean']['name']}` mean={best['best_mean']['mean_m']:.4f} m",
        f"- Best repair-gated mean: `{best['best_repair_gated_mean']['name']}` mean={best['best_repair_gated_mean']['mean_m']:.4f} m",
        f"- Best global interpolation mean: `{best['best_global_mean']['name']}` mean={best['best_global_mean']['mean_m']:.4f} m",
        f"- Best random gate mean: `{best['best_random_mean']['name']}` mean={best['best_random_mean']['mean_m']:.4f} m",
        "",
        "## Success Check",
        "",
    ]
    success = summary["success_check"]
    for key, value in success.items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(
        [
            "",
            "## Result Table",
            "",
            "| name | mean | median | P95 | max | CT | TW | KS | good worsened | good strong worsened | outlier delta | retained repair | improved frac | gate mean |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['name']} | {row['mean_m']:.4f} | {row['median_m']:.4f} | {row['p95_m']:.4f} | "
            f"{row['max_m']:.4f} | {row['ct']:.4f} | {row['tw']:.4f} | {row['ks']:.4f} | "
            f"{row['good_worsened_fraction']:.4f} | {row['good_strong_worsened_fraction']:.4f} | "
            f"{row['outlier_mean_delta_m']:.4f} | {row['outlier_retained_repair_ratio']:.4f} | "
            f"{row['overall_improved_fraction']:.4f} | {row['gate_mean'] if row['gate_mean'] is not None else ''} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run repair-gated residual correction on a baseline diffusion embedding.")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA1)
    parser.add_argument("--distance-run", type=Path, default=Path("results/adp_old_fused/stride1_n20997_seed10"))
    parser.add_argument("--init-embedding", type=Path, default=Path("results/adp_old_siamese/stride1_n20997_seed10/embedding.npz"))
    parser.add_argument(
        "--baseline-refined-embedding",
        type=Path,
        default=Path("results/diffusion_refined_charting/stride1_n20997_k128_lambda0.15_seed10/embedding.npz"),
    )
    parser.add_argument("--out-dir", type=Path, default=Path("results/repair_gated_residual_correction"))
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--physical-k", type=int, default=128)
    parser.add_argument("--top-fractions", nargs="+", type=float, default=[0.20, 0.40, 0.60, 0.80])
    parser.add_argument("--floor-values", nargs="+", type=float, default=[0.2, 0.3, 0.5])
    parser.add_argument("--global-gammas", nargs="+", type=float, default=[0.2, 0.4, 0.6, 0.8, 1.0])
    parser.add_argument("--rho", type=float, default=0.25)
    parser.add_argument("--soft-alpha", type=float, default=4.0)
    parser.add_argument("--metric-subset-fraction", type=float, default=0.2)
    parser.add_argument("--good-init-threshold", type=float, default=0.2)
    parser.add_argument("--outlier-init-threshold", type=float, default=1.0)
    parser.add_argument("--strong-worsen-threshold", type=float, default=0.05)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    data = load_data1(args.data_path, seed=args.seed)
    indices = select_indices(len(data.h_by_array), args.stride, args.max_samples)
    positions = data.ue_pos[indices]

    distance_data = np.load(args.distance_run / "distance_data.npz")
    if not np.array_equal(distance_data["indices"], indices):
        raise ValueError("distance-run indices do not match requested data selection")

    init_data = np.load(args.init_embedding)
    refined_data = np.load(args.baseline_refined_embedding)
    if not np.array_equal(init_data["indices"], indices):
        raise ValueError("init embedding indices do not match requested data selection")
    if not np.array_equal(refined_data["indices"], indices):
        raise ValueError("baseline refined embedding indices do not match requested data selection")

    y0 = init_data["embedding"].astype(np.float32)
    z_base = refined_data["embedding"].astype(np.float32)
    geodesic = np.load(args.distance_run / "geodesic.npy", mmap_mode="r")
    neighbors = np.load(args.distance_run / "geodesic_top512.npy")[:, : args.physical_k].astype(np.int32)

    scale_stress, global_eta = median_physical_scale_stress(y0, geodesic, neighbors[:, :32])
    graph = build_local_scaled_graph(geodesic, neighbors, args.physical_k)
    residual = graph_smoothness_residual(y0, graph)
    score_components = {
        "r": robust_zscore(residual),
        "Bpos": robust_zscore(np.maximum(scale_stress, 0.0)),
    }
    repair_scores = {
        "z_r": score_components["r"],
        "z_Bpos": score_components["Bpos"],
        f"z_Bpos_plus_{args.rho:g}z_r": score_components["Bpos"] + args.rho * score_components["r"],
    }

    init_error = localization_errors(positions, y0)
    baseline_error = localization_errors(positions, z_base)
    good_initial = init_error < args.good_init_threshold
    outlier_initial = init_error >= args.outlier_init_threshold
    subset_size = max(1, int(len(positions) * args.metric_subset_fraction))
    metric_subset = np.array(random.sample(range(len(positions)), subset_size), dtype=np.int32)

    method_results: list[dict[str, object]] = []

    def add_method(name: str, gate: np.ndarray) -> None:
        embedding = apply_residual_gate(y0, z_base, gate)
        method_results.append(
            evaluate_embedding(
                name,
                embedding,
                positions,
                init_error,
                baseline_error,
                metric_subset,
                good_initial,
                outlier_initial,
                args.strong_worsen_threshold,
                gate=gate,
            )
        )

    method_results.append(
        evaluate_embedding(
            "initial_Y0",
            y0,
            positions,
            init_error,
            baseline_error,
            metric_subset,
            good_initial,
            outlier_initial,
            args.strong_worsen_threshold,
            gate=np.zeros(len(y0), dtype=np.float32),
        )
    )
    method_results.append(
        evaluate_embedding(
            "baseline_Z_base",
            z_base,
            positions,
            init_error,
            baseline_error,
            metric_subset,
            good_initial,
            outlier_initial,
            args.strong_worsen_threshold,
            gate=np.ones(len(y0), dtype=np.float32),
        )
    )

    for gamma in args.global_gammas:
        add_method(f"global_gamma{gamma:g}", make_global_gate(len(y0), gamma))

    for score_name, score in repair_scores.items():
        for fraction in args.top_fractions:
            hard_gate = hard_top_gate(score, fraction)
            add_method(f"repair_{score_name}_hard_top{fraction:g}", hard_gate)
            add_method(
                f"random_{score_name}_hard_top{fraction:g}",
                make_random_gate(hard_gate, args.seed + int(fraction * 1000)),
            )
            soft_gate = soft_sigmoid_gate(score, fraction, args.soft_alpha)
            add_method(f"repair_{score_name}_soft_top{fraction:g}", soft_gate)
            add_method(
                f"random_{score_name}_soft_top{fraction:g}",
                make_random_gate(soft_gate, args.seed + 17 + int(fraction * 1000)),
            )
            for floor in args.floor_values:
                floor_gate = floor_top_gate(score, fraction, floor)
                add_method(f"repair_{score_name}_floor{floor:g}_top{fraction:g}", floor_gate)
                add_method(
                    f"random_{score_name}_floor{floor:g}_top{fraction:g}",
                    make_random_gate(floor_gate, args.seed + int(floor * 100) + int(fraction * 1000)),
                )

    rows = [result_row(result) for result in method_results]
    repair_rows = [row for row in rows if row["name"].startswith("repair_")]
    global_rows = [row for row in rows if row["name"].startswith("global_")]
    random_rows = [row for row in rows if row["name"].startswith("random_")]
    baseline_row = next(row for row in rows if row["name"] == "baseline_Z_base")
    best_repair = min(repair_rows, key=lambda row: row["mean_m"])
    best_global = min(global_rows, key=lambda row: row["mean_m"])
    best_random = min(random_rows, key=lambda row: row["mean_m"])
    best_mean = min(rows, key=lambda row: row["mean_m"])
    baseline_outlier_repair = -float(baseline_row["outlier_mean_delta_m"])
    success_check = {
        "repair_beats_best_global_mean": bool(best_repair["mean_m"] < best_global["mean_m"]),
        "repair_beats_best_random_mean": bool(best_repair["mean_m"] < best_random["mean_m"]),
        "repair_keeps_90pct_baseline_outlier_repair": bool(best_repair["outlier_retained_repair_ratio"] >= 0.90),
        "repair_good_worsened_le_baseline": bool(best_repair["good_worsened_fraction"] <= baseline_row["good_worsened_fraction"]),
        "repair_good_strong_worsened_le_baseline": bool(
            best_repair["good_strong_worsened_fraction"] <= baseline_row["good_strong_worsened_fraction"]
        ),
        "baseline_outlier_mean_repair_m": baseline_outlier_repair,
    }

    run_name = f"stride{args.stride}_n{len(indices)}_seed{args.seed}_phys{args.physical_k}"
    out_dir = args.out_dir / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    write_rows_csv(out_dir / "method_summary.csv", rows)
    summary = {
        "method": "repair-gated residual correction",
        "formula": "Z_gate_i = Y0_i + g_i * (Z_base_i - Y0_i)",
        "dataset": str(args.data_path),
        "distance_run": str(args.distance_run),
        "init_embedding": str(args.init_embedding),
        "baseline_refined_embedding": str(args.baseline_refined_embedding),
        "num_samples": int(len(indices)),
        "seed": int(args.seed),
        "physical_k": int(args.physical_k),
        "global_eta_y0_over_geodesic": float(global_eta),
        "repair_score_names": list(repair_scores),
        "top_fractions": [float(v) for v in args.top_fractions],
        "floor_values": [float(v) for v in args.floor_values],
        "global_gammas": [float(v) for v in args.global_gammas],
        "best_methods": {
            "best_mean": best_mean,
            "best_repair_gated_mean": best_repair,
            "best_global_mean": best_global,
            "best_random_mean": best_random,
            "baseline": baseline_row,
        },
        "success_check": success_check,
        "results": method_results,
        "result_rows": rows,
        "outputs": {
            "json": str(out_dir / "result.json"),
            "markdown": str(out_dir / "result_summary.md"),
            "method_summary_csv": str(out_dir / "method_summary.csv"),
        },
    }
    write_json(out_dir / "result.json", summary)
    write_markdown(out_dir / "result_summary.md", summary)
    print("REPAIR_GATED_RESIDUAL_CORRECTION", json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
