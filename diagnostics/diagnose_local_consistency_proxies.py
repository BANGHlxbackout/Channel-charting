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
import scipy.sparse as sp
from scipy.stats import rankdata, spearmanr

from common_dichasus import DEFAULT_DATA1, load_data1
from common_metrics import affine_transform, write_json
from diagnostics.diagnose_diffusion_refinement import knn_indices, row_overlap, summarize_values
from run_adp_fused import select_indices
from run_diffusion_refined_charting import build_local_scaled_graph


def binary_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=bool)
    if scores.ndim != 1 or labels.ndim != 1 or scores.shape[0] != labels.shape[0]:
        raise ValueError("scores and labels must be one-dimensional arrays with the same length")
    valid = np.isfinite(scores)
    scores = scores[valid]
    labels = labels[valid]
    positives = int(np.sum(labels))
    negatives = int(labels.size - positives)
    if positives == 0 or negatives == 0:
        return float("nan")
    ranks = rankdata(scores, method="average")
    positive_rank_sum = float(np.sum(ranks[labels]))
    return float((positive_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives))


def safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    if int(np.sum(valid)) < 3:
        return float("nan")
    if np.std(x[valid]) == 0.0 or np.std(y[valid]) == 0.0:
        return float("nan")
    return float(spearmanr(x[valid], y[valid]).statistic)


def robust_zscore(values: np.ndarray, clip: float = 6.0) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    median = float(np.nanmedian(values))
    q25, q75 = np.nanpercentile(values, [25, 75])
    scale = float(q75 - q25)
    if not np.isfinite(scale) or scale < 1e-9:
        scale = float(np.nanstd(values))
    if not np.isfinite(scale) or scale < 1e-9:
        return np.zeros_like(values, dtype=np.float32)
    return np.clip((values - median) / scale, -clip, clip).astype(np.float32)


def mutual_support(physical_neighbors: np.ndarray, y0_neighbors: np.ndarray, k: int) -> np.ndarray:
    if physical_neighbors.shape[0] != y0_neighbors.shape[0]:
        raise ValueError("neighbor matrices must have the same number of rows")
    y0_top_sets = [set(row) for row in y0_neighbors[:, :k]]
    support = np.empty(physical_neighbors.shape[0], dtype=np.float32)
    for i, row in enumerate(physical_neighbors[:, :k]):
        count = 0
        for j in row:
            j_int = int(j)
            if j_int in y0_top_sets[i] and i in y0_top_sets[j_int]:
                count += 1
        support[i] = count / float(k)
    return support


def rank_weighted_support(
    physical_neighbors: np.ndarray,
    y0_rank_neighbors: np.ndarray,
    k: int,
    tau: float,
) -> np.ndarray:
    if tau <= 0.0:
        raise ValueError("tau must be positive")
    rank_lookup: list[dict[int, int]] = []
    for row in y0_rank_neighbors:
        rank_lookup.append({int(node): rank + 1 for rank, node in enumerate(row)})
    missing_rank = y0_rank_neighbors.shape[1] + 1
    support = np.empty(physical_neighbors.shape[0], dtype=np.float32)
    for i, row in enumerate(physical_neighbors[:, :k]):
        weights = [
            np.exp(-rank_lookup[i].get(int(j), missing_rank) / tau)
            for j in row
        ]
        support[i] = float(np.mean(weights))
    return support


def multi_k_stability(
    physical_neighbors: np.ndarray,
    y0_neighbors: np.ndarray,
    ks: tuple[int, ...],
) -> np.ndarray:
    if not ks:
        raise ValueError("ks must not be empty")
    overlaps = []
    for k in ks:
        if k > physical_neighbors.shape[1] or k > y0_neighbors.shape[1]:
            raise ValueError("all stability ks must fit the neighbor matrices")
        overlaps.append(row_overlap(physical_neighbors, y0_neighbors, k))
    return np.min(np.vstack(overlaps), axis=0).astype(np.float32)


def median_physical_scale_stress(
    y0: np.ndarray,
    geodesic: np.ndarray,
    physical_neighbors: np.ndarray,
    eps: float = 1e-6,
) -> tuple[np.ndarray, float]:
    rows = np.arange(y0.shape[0])[:, None]
    y0_dist = np.linalg.norm(y0[rows] - y0[physical_neighbors], axis=2)
    geo_dist = np.asarray(geodesic[rows, physical_neighbors], dtype=np.float32)
    ratio = (y0_dist + eps) / (geo_dist + eps)
    finite_ratio = ratio[np.isfinite(ratio)]
    if finite_ratio.size == 0:
        raise ValueError("cannot estimate global eta from empty finite ratios")
    eta = float(np.median(finite_ratio))
    stress = np.median(np.log((y0_dist + eps) / (eta * (geo_dist + eps))), axis=1)
    return stress.astype(np.float32), eta


def graph_smoothness_residual(y0: np.ndarray, graph: sp.coo_matrix) -> np.ndarray:
    graph = graph.tocsr()
    degree = np.asarray(graph.sum(axis=1)).ravel().astype(np.float32)
    weighted_avg = graph @ y0
    nonzero = degree > 1e-12
    weighted_avg[nonzero] = weighted_avg[nonzero] / degree[nonzero, None]
    weighted_avg[~nonzero] = y0[~nonzero]
    return np.linalg.norm(y0 - weighted_avg, axis=1).astype(np.float32)


def anti_repair_gate(scale_stress: np.ndarray, residual: np.ndarray, rho: float) -> np.ndarray:
    z_b_pos = np.maximum(robust_zscore(np.maximum(scale_stress, 0.0)), 0.0)
    z_r_pos = np.maximum(robust_zscore(residual), 0.0)
    return np.exp(-z_b_pos - rho * z_r_pos).astype(np.float32)


def score_report(
    score: np.ndarray,
    init_error: np.ndarray,
    error_delta: np.ndarray,
    good_initial: np.ndarray,
    strong_worsened: np.ndarray,
    outlier_initial: np.ndarray,
) -> dict[str, float]:
    auc_good = binary_auc(score, good_initial)
    auc_strong = binary_auc(score, strong_worsened)
    auc_outlier = binary_auc(score, outlier_initial)
    return {
        "spearman_init_error": safe_spearman(score, init_error),
        "spearman_error_delta": safe_spearman(score, error_delta),
        "auc_good_initial_high_score_positive": auc_good,
        "auc_good_initial_best_direction": max(auc_good, 1.0 - auc_good) if np.isfinite(auc_good) else float("nan"),
        "auc_strong_worsened_high_score_positive": auc_strong,
        "auc_strong_worsened_best_direction": max(auc_strong, 1.0 - auc_strong) if np.isfinite(auc_strong) else float("nan"),
        "auc_outlier_initial_high_score_positive": auc_outlier,
        "auc_outlier_initial_best_direction": max(auc_outlier, 1.0 - auc_outlier) if np.isfinite(auc_outlier) else float("nan"),
    }


def top_fraction_report(
    score: np.ndarray,
    good_initial: np.ndarray,
    strong_worsened: np.ndarray,
    outlier_initial: np.ndarray,
    fraction: float,
) -> dict[str, float]:
    if not 0.0 < fraction < 1.0:
        raise ValueError("fraction must be in (0, 1)")
    count = max(1, int(round(score.shape[0] * fraction)))
    order = np.argsort(score)[::-1][:count]
    global_good = float(np.mean(good_initial))
    global_strong = float(np.mean(strong_worsened))
    global_outlier = float(np.mean(outlier_initial))
    top_good = float(np.mean(good_initial[order]))
    top_strong = float(np.mean(strong_worsened[order]))
    top_outlier = float(np.mean(outlier_initial[order]))
    return {
        "top_fraction": float(fraction),
        "top_count": int(count),
        "global_good_initial_fraction": global_good,
        "top_good_initial_fraction": top_good,
        "good_initial_enrichment": top_good / max(global_good, 1e-12),
        "global_strong_worsened_fraction": global_strong,
        "top_strong_worsened_fraction": top_strong,
        "strong_worsened_enrichment": top_strong / max(global_strong, 1e-12),
        "global_outlier_initial_fraction": global_outlier,
        "top_outlier_initial_fraction": top_outlier,
        "outlier_initial_enrichment": top_outlier / max(global_outlier, 1e-12),
    }


def strict_gate_report(
    raw_report: dict[str, float],
    candidate_report: dict[str, float],
    top_report: dict[str, float],
) -> dict[str, object]:
    good_auc_gain = (
        candidate_report["auc_good_initial_high_score_positive"]
        - raw_report["auc_good_initial_high_score_positive"]
    )
    strong_auc_gain = (
        candidate_report["auc_strong_worsened_high_score_positive"]
        - raw_report["auc_strong_worsened_high_score_positive"]
    )
    absolute_auc_pass = (
        candidate_report["auc_good_initial_high_score_positive"] >= 0.65
        or candidate_report["auc_strong_worsened_high_score_positive"] >= 0.62
    )
    spearman_pass = abs(candidate_report["spearman_error_delta"]) >= 0.15
    enrichment_pass = (
        top_report["good_initial_enrichment"] >= 1.25
        or top_report["strong_worsened_enrichment"] >= 1.20
    )
    outlier_not_selected = top_report["outlier_initial_enrichment"] <= 1.10
    relative_gain_pass = good_auc_gain >= 0.07 and strong_auc_gain >= 0.05
    passes = bool(relative_gain_pass and absolute_auc_pass and spearman_pass and enrichment_pass and outlier_not_selected)
    return {
        "passes": passes,
        "relative_gain_pass": bool(relative_gain_pass),
        "absolute_auc_pass": bool(absolute_auc_pass),
        "spearman_pass": bool(spearman_pass),
        "enrichment_pass": bool(enrichment_pass),
        "outlier_not_selected_pass": bool(outlier_not_selected),
        "auc_good_initial_gain_over_raw_A_nbr": float(good_auc_gain),
        "auc_strong_worsened_gain_over_raw_A_nbr": float(strong_auc_gain),
        "requirements": {
            "good_auc_gain_min": 0.07,
            "strong_auc_gain_min": 0.05,
            "auc_good_initial_min_or": 0.65,
            "auc_strong_worsened_min_or": 0.62,
            "abs_spearman_error_delta_min": 0.15,
            "top20_good_enrichment_min_or": 1.25,
            "top20_strong_worsened_enrichment_min_or": 1.20,
            "top20_outlier_enrichment_max": 1.10,
        },
    }


def repair_gate_report(score_reports: dict[str, dict[str, float]]) -> dict[str, object]:
    candidates = ["B_pos_scale_stress", "r_smoothness_residual"]
    best_auc_name = None
    best_auc = -np.inf
    best_spearman_name = None
    best_abs_spearman = -np.inf
    for name in candidates:
        report = score_reports[name]
        auc = report["auc_outlier_initial_high_score_positive"]
        if np.isfinite(auc) and auc > best_auc:
            best_auc = float(auc)
            best_auc_name = name
        rho = report["spearman_error_delta"]
        if np.isfinite(rho) and rho < 0.0 and abs(rho) > best_abs_spearman:
            best_abs_spearman = float(abs(rho))
            best_spearman_name = name
    return {
        "passes": bool(best_auc >= 0.70 or best_abs_spearman >= 0.20),
        "best_auc_outlier_score": best_auc_name,
        "best_auc_outlier_initial": best_auc,
        "best_negative_spearman_score": best_spearman_name,
        "best_abs_negative_spearman_error_delta": best_abs_spearman,
        "auc_outlier_min_or": 0.70,
        "abs_negative_spearman_error_delta_min_or": 0.20,
    }


def bucket_summary(score: np.ndarray, masks: dict[str, np.ndarray]) -> dict[str, dict[str, float]]:
    output = {}
    for name, mask in masks.items():
        output[name] = summarize_values(score[mask]) if np.any(mask) else {}
    return output


def save_score_scatter(path: Path, score: np.ndarray, target: np.ndarray, title: str, ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.8), layout="constrained")
    ax.scatter(score, target, s=5, alpha=0.22, color="#22577a", edgecolors="none")
    ax.axhline(0.0, color="#222222", linewidth=1.0)
    ax.set_title(title)
    ax.set_xlabel("score")
    ax.set_ylabel(ylabel)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def write_rows_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("no rows to write")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_summary_md(path: Path, summary: dict[str, object]) -> None:
    score_reports = summary["score_reports"]
    strict_gates = summary["strict_protection_gates"]
    repair_gate = summary["repair_gate"]
    best_candidate = summary["best_protection_candidate"]
    lines = [
        "# Local Consistency Proxy Diagnostics",
        "",
        "This is a proxy diagnostic only. UE positions are used only for post-hoc labels and metrics.",
        "",
        "## Strict Gate Result",
        "",
        f"- Best protection candidate: `{best_candidate['name']}`",
        f"- Best protection gate pass: `{best_candidate['strict_gate']['passes']}`",
        f"- Repair gate pass: `{repair_gate['passes']}`",
        f"- Repair best outlier AUC: `{repair_gate['best_auc_outlier_score']}` = {repair_gate['best_auc_outlier_initial']:.4f}",
        "",
        "## Protection Gate Details",
        "",
        "| candidate | pass | AUC good | AUC strong worsened | rho(delta) | good AUC gain | strong AUC gain | top20 good enrich | top20 strong enrich | top20 outlier enrich |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, gate in strict_gates.items():
        report = score_reports[name]
        top = summary["top20_reports"][name]
        lines.append(
            f"| {name} | {gate['passes']} | "
            f"{report['auc_good_initial_high_score_positive']:.4f} | "
            f"{report['auc_strong_worsened_high_score_positive']:.4f} | "
            f"{report['spearman_error_delta']:.4f} | "
            f"{gate['auc_good_initial_gain_over_raw_A_nbr']:.4f} | "
            f"{gate['auc_strong_worsened_gain_over_raw_A_nbr']:.4f} | "
            f"{top['good_initial_enrichment']:.4f} | "
            f"{top['strong_worsened_enrichment']:.4f} | "
            f"{top['outlier_initial_enrichment']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Score Summary",
            "",
            "| score | rho(init_error) | rho(error_delta) | AUC good | AUC strong worsened | AUC outlier |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for name, report in score_reports.items():
        lines.append(
            f"| {name} | {report['spearman_init_error']:.4f} | "
            f"{report['spearman_error_delta']:.4f} | "
            f"{report['auc_good_initial_high_score_positive']:.4f} | "
            f"{report['auc_strong_worsened_high_score_positive']:.4f} | "
            f"{report['auc_outlier_initial_high_score_positive']:.4f} |"
        )

    lines.extend(["", "## Interpretation", ""])
    if best_candidate["strict_gate"]["passes"]:
        lines.append("- A revised protection proxy passed the strict gate, so a conservative adaptive-fidelity experiment is justified.")
    else:
        lines.append("- No revised protection proxy passed the strict gate. Do not implement adaptive fidelity from these protection scores yet.")
    if repair_gate["passes"]:
        lines.append("- Repair-needed / outlier-like regions still have usable signal from stress/residual proxies.")
    else:
        lines.append("- Repair proxy signal is also insufficient under the stricter gate.")
    lines.append("- Baseline displacement is kept diagnostic-only and is not included in the protection gate.")

    lines.extend(["", "## Output Files", ""])
    for key, value in summary["outputs"].items():
        lines.append(f"- `{key}`: `{value}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose local consistency protection proxies before adaptive fidelity refinement.")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA1)
    parser.add_argument("--distance-run", type=Path, default=Path("results/adp_old_fused/stride1_n20997_seed10"))
    parser.add_argument("--init-embedding", type=Path, default=Path("results/adp_old_siamese/stride1_n20997_seed10/embedding.npz"))
    parser.add_argument(
        "--refined-embedding",
        type=Path,
        default=Path("results/diffusion_refined_charting/stride1_n20997_k128_lambda0.15_seed10/embedding.npz"),
    )
    parser.add_argument("--out-dir", type=Path, default=Path("results/local_consistency_proxy_diagnostics"))
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--neighbor-k", type=int, default=32)
    parser.add_argument("--physical-k", type=int, default=128)
    parser.add_argument("--rank-k", type=int, default=256)
    parser.add_argument("--rank-tau", type=float, default=32.0)
    parser.add_argument("--anti-rho", type=float, default=0.25)
    parser.add_argument("--strong-worsen-threshold", type=float, default=0.05)
    parser.add_argument("--good-init-threshold", type=float, default=0.2)
    parser.add_argument("--outlier-init-threshold", type=float, default=1.0)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.neighbor_k > args.physical_k:
        raise ValueError("--neighbor-k must be <= --physical-k")
    if args.rank_k < max(args.neighbor_k, 64):
        raise ValueError("--rank-k must be at least max(neighbor-k, 64)")

    data = load_data1(args.data_path, seed=args.seed)
    indices = select_indices(len(data.h_by_array), args.stride, args.max_samples)
    positions = data.ue_pos[indices]

    distance_data = np.load(args.distance_run / "distance_data.npz")
    if not np.array_equal(distance_data["indices"], indices):
        raise ValueError("distance-run indices do not match requested data selection")

    init_data = np.load(args.init_embedding)
    refined_data = np.load(args.refined_embedding)
    if not np.array_equal(init_data["indices"], indices):
        raise ValueError("init embedding indices do not match requested data selection")
    if not np.array_equal(refined_data["indices"], indices):
        raise ValueError("refined embedding indices do not match requested data selection")
    y0 = init_data["embedding"].astype(np.float32)
    z_base = refined_data["embedding"].astype(np.float32)

    aligned_y0 = affine_transform(positions, y0).astype(np.float32)
    aligned_z = affine_transform(positions, z_base).astype(np.float32)
    init_error = np.linalg.norm(aligned_y0 - positions, axis=1)
    refined_error = np.linalg.norm(aligned_z - positions, axis=1)
    error_delta = refined_error - init_error

    physical_neighbors_path = args.distance_run / "geodesic_top512.npy"
    if not physical_neighbors_path.exists():
        raise FileNotFoundError(f"missing cached physical neighbors: {physical_neighbors_path}")
    physical_neighbors = np.load(physical_neighbors_path)
    if physical_neighbors.shape[1] < args.physical_k:
        raise ValueError("cached physical neighbors do not cover --physical-k")
    physical_neighbors = physical_neighbors[:, : args.physical_k].astype(np.int32)
    physical_eval_neighbors = physical_neighbors[:, : args.neighbor_k]

    y0_neighbors_rank = knn_indices(y0, args.rank_k)
    y0_neighbors_eval = y0_neighbors_rank[:, : args.neighbor_k]
    y0_neighbors_64 = y0_neighbors_rank[:, :64]

    geodesic = np.load(args.distance_run / "geodesic.npy", mmap_mode="r")
    scale_stress, global_eta = median_physical_scale_stress(y0, geodesic, physical_eval_neighbors)
    graph = build_local_scaled_graph(geodesic, physical_neighbors, args.physical_k)
    residual = graph_smoothness_residual(y0, graph)
    baseline_move = np.linalg.norm(z_base - y0, axis=1).astype(np.float32)

    raw_a = row_overlap(physical_eval_neighbors, y0_neighbors_eval, args.neighbor_k)
    mutual = mutual_support(physical_eval_neighbors, y0_neighbors_eval, args.neighbor_k)
    rank_support = rank_weighted_support(physical_eval_neighbors, y0_neighbors_rank, args.neighbor_k, args.rank_tau)
    stability = multi_k_stability(physical_neighbors, y0_neighbors_64, (16, 32, 64))
    anti = anti_repair_gate(scale_stress, residual, args.anti_rho)
    c_prot = (mutual * rank_support * stability).astype(np.float32)
    c_prot_anti = (c_prot * anti).astype(np.float32)

    z_b_pos = robust_zscore(np.maximum(scale_stress, 0.0))
    z_resid = robust_zscore(residual)
    c_prot_minus_repair = (
        robust_zscore(c_prot) - z_b_pos - args.anti_rho * z_resid
    ).astype(np.float32)

    scores = {
        "raw_A_nbr": raw_a.astype(np.float32),
        "C_mutual": mutual.astype(np.float32),
        "C_rank_weighted": rank_support.astype(np.float32),
        "C_multi_k_stability": stability.astype(np.float32),
        "C_prot": c_prot.astype(np.float32),
        "C_prot_anti": c_prot_anti.astype(np.float32),
        "C_prot_minus_repair_z": c_prot_minus_repair.astype(np.float32),
        "B_scale_stress": scale_stress.astype(np.float32),
        "B_pos_scale_stress": np.maximum(scale_stress, 0.0).astype(np.float32),
        "r_smoothness_residual": residual.astype(np.float32),
        "d_move_diagnostic_only": baseline_move.astype(np.float32),
    }

    good_initial = init_error < args.good_init_threshold
    strong_worsened = error_delta > args.strong_worsen_threshold
    outlier_initial = init_error >= args.outlier_init_threshold
    init_buckets = {
        "init_error_lt_0p2m": init_error < 0.2,
        "init_error_0p2m_to_0p5m": (init_error >= 0.2) & (init_error < 0.5),
        "init_error_0p5m_to_1p0m": (init_error >= 0.5) & (init_error < 1.0),
        "init_error_ge_1p0m": init_error >= 1.0,
    }

    score_reports = {
        name: score_report(score, init_error, error_delta, good_initial, strong_worsened, outlier_initial)
        for name, score in scores.items()
    }
    top20_reports = {
        name: top_fraction_report(score, good_initial, strong_worsened, outlier_initial, 0.20)
        for name, score in scores.items()
    }
    bucket_reports = {name: bucket_summary(score, init_buckets) for name, score in scores.items()}

    protection_candidates = [
        "C_mutual",
        "C_rank_weighted",
        "C_multi_k_stability",
        "C_prot",
        "C_prot_anti",
        "C_prot_minus_repair_z",
    ]
    strict_gates = {
        name: strict_gate_report(score_reports["raw_A_nbr"], score_reports[name], top20_reports[name])
        for name in protection_candidates
    }
    best_name = max(
        protection_candidates,
        key=lambda name: (
            strict_gates[name]["passes"],
            score_reports[name]["auc_good_initial_high_score_positive"]
            + score_reports[name]["auc_strong_worsened_high_score_positive"],
        ),
    )
    repair_gate = repair_gate_report(score_reports)

    out_dir = args.out_dir / f"stride{args.stride}_n{len(indices)}_seed{args.seed}_k{args.neighbor_k}_phys{args.physical_k}_revised"
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    for name, score in scores.items():
        save_score_scatter(
            plot_dir / f"{name}_vs_error_delta.png",
            score,
            error_delta,
            f"{name} vs error delta",
            "refined error - init error [m]",
        )
        save_score_scatter(
            plot_dir / f"{name}_vs_init_error.png",
            score,
            init_error,
            f"{name} vs initial error",
            "initial error [m]",
        )

    rows: list[dict[str, object]] = []
    for i in range(len(indices)):
        row: dict[str, object] = {
            "local_id": int(i),
            "dataset_index": int(indices[i]),
            "init_error_m": float(init_error[i]),
            "refined_error_m": float(refined_error[i]),
            "error_delta_m": float(error_delta[i]),
            "good_initial": bool(good_initial[i]),
            "strong_worsened": bool(strong_worsened[i]),
            "outlier_initial": bool(outlier_initial[i]),
        }
        for name, score in scores.items():
            row[name] = float(score[i])
        rows.append(row)
    write_rows_csv(out_dir / "node_proxy_scores.csv", rows)

    summary = {
        "diagnostic_only_uses_positions_for_labels": True,
        "positions_used_for_proxy_construction": False,
        "dataset": str(args.data_path),
        "distance_run": str(args.distance_run),
        "init_embedding": str(args.init_embedding),
        "refined_embedding": str(args.refined_embedding),
        "num_samples": int(len(indices)),
        "seed": int(args.seed),
        "neighbor_k": int(args.neighbor_k),
        "physical_k": int(args.physical_k),
        "rank_k": int(args.rank_k),
        "rank_tau": float(args.rank_tau),
        "anti_rho": float(args.anti_rho),
        "global_eta_y0_over_geodesic": float(global_eta),
        "label_counts": {
            "good_initial": int(np.sum(good_initial)),
            "strong_worsened": int(np.sum(strong_worsened)),
            "outlier_initial": int(np.sum(outlier_initial)),
        },
        "label_fractions": {
            "good_initial": float(np.mean(good_initial)),
            "strong_worsened": float(np.mean(strong_worsened)),
            "outlier_initial": float(np.mean(outlier_initial)),
        },
        "score_summaries": {name: summarize_values(score) for name, score in scores.items()},
        "score_reports": score_reports,
        "top20_reports": top20_reports,
        "score_by_init_error_bucket": bucket_reports,
        "strict_protection_gates": strict_gates,
        "best_protection_candidate": {
            "name": best_name,
            "score_report": score_reports[best_name],
            "top20_report": top20_reports[best_name],
            "strict_gate": strict_gates[best_name],
        },
        "repair_gate": repair_gate,
        "outputs": {
            "json": str(out_dir / "local_consistency_proxy_diagnostics.json"),
            "markdown": str(out_dir / "local_consistency_proxy_diagnostics.md"),
            "node_proxy_scores_csv": str(out_dir / "node_proxy_scores.csv"),
            "plots_dir": str(plot_dir),
        },
    }
    write_json(out_dir / "local_consistency_proxy_diagnostics.json", summary)
    write_summary_md(out_dir / "local_consistency_proxy_diagnostics.md", summary)
    print("LOCAL_CONSISTENCY_PROXY_DIAGNOSTICS", json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
