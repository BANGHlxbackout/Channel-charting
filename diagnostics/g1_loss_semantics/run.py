"""Offline test of whether saved D_geo behaves like physical path length.

GT is read only for this diagnostic. Nothing here trains or selects a model.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.sparse.csgraph import dijkstra
from scipy.stats import spearmanr

from common_metrics import affine_transform
from methods.g1_acf_graph_fcf.graph_utils import rebuild_directed_graph


ROOT = Path(__file__).resolve().parents[2]
DIST = ROOT / "results/adp_old_fused/stride1_n20997_seed10"
EMBED = ROOT / "results/g1_acf_graph_fcf/stride1_n20997_seed10_graph_full/embedding.npz"
OUT = ROOT / "results/g1_acf_graph_fcf/loss_semantic_gate"
SEED = 10
N_SOURCES = 64
TARGETS_PER_SOURCE = 256


def main() -> None:
    saved = np.load(EMBED)
    positions = np.asarray(saved["positions"], dtype=np.float64)
    chart = np.asarray(saved["embedding"], dtype=np.float64)
    aligned = affine_transform(positions, chart)
    distance_data = np.load(DIST / "distance_data.npz")
    if not np.array_equal(saved["indices"], distance_data["indices"]):
        raise ValueError("G1 indices do not match D_geo indices")
    fused = np.load(DIST / "fused.npy", mmap_mode="r")
    geo = np.load(DIST / "geodesic.npy", mmap_mode="r")
    graph = rebuild_directed_graph(fused, 20)
    rng = np.random.default_rng(SEED)
    sources = rng.choice(len(positions), size=N_SOURCES, replace=False)
    dist, predecessors = dijkstra(graph, directed=False, indices=sources,
                                  return_predecessors=True)

    rows = []
    max_geo_diff = 0.0
    max_path_cost_diff = 0.0
    for source_idx, source in enumerate(sources):
        targets = rng.choice(len(positions), size=TARGETS_PER_SOURCE, replace=False)
        for target in targets:
            if source == target:
                continue
            path = [int(target)]
            while path[-1] != source:
                parent = int(predecessors[source_idx, path[-1]])
                if parent < 0 or len(path) > len(positions):
                    raise ValueError("invalid shortest path")
                path.append(parent)
            path.reverse()
            g = float(geo[source, target])
            max_geo_diff = max(max_geo_diff, abs(float(dist[source_idx, target]) - g))
            physical_chord = float(np.linalg.norm(positions[source] - positions[target]))
            if physical_chord <= 0.5:
                continue
            path_array = np.asarray(path, dtype=np.int64)
            physical_edges = np.linalg.norm(np.diff(positions[path_array], axis=0), axis=1)
            # Reciprocal fused costs differ only at floating-point scale; the
            # minimum reproduces undirected Dijkstra's edge choice below.
            edge_costs = np.minimum(np.asarray(fused[path_array[:-1], path_array[1:]]),
                                    np.asarray(fused[path_array[1:], path_array[:-1]]))
            max_path_cost_diff = max(max_path_cost_diff, abs(float(edge_costs.sum()) - g))
            physical_path = float(physical_edges.sum())
            chart_chord = float(np.linalg.norm(aligned[source] - aligned[target]))
            chart_path = float(np.linalg.norm(np.diff(chart[path], axis=0), axis=1).sum())
            gt05 = physical_edges > 0.5
            gt1 = physical_edges > 1.0
            rows.append((source_idx, len(path) - 1, g, physical_chord, physical_path,
                         chart_chord, chart_path,
                         float(edge_costs[gt05].sum()), float(physical_edges[gt05].sum()),
                         int(gt05.sum()), float(edge_costs[gt1].sum()),
                         float(physical_edges[gt1].sum()), int(gt1.sum())))
    if max_geo_diff > 1e-4:
        raise ValueError(f"paths disagree with saved D_geo: {max_geo_diff}")
    if max_path_cost_diff > 1e-3:
        raise ValueError(f"path edge costs disagree with saved D_geo: {max_path_cost_diff}")
    a = np.asarray(rows, dtype=np.float64)
    (src, hops, g, chord, length, chart_chord, chart_path,
     cost_gt05, phys_gt05, count_gt05, cost_gt1, phys_gt1, count_gt1) = a.T
    curvature = length / chord
    calibration = src < N_SOURCES // 2
    test = ~calibration
    hop_low, hop_high = np.quantile(hops[calibration], [0.25, 0.75])
    curve_low, curve_high = np.quantile(curvature[calibration], [0.25, 0.75])
    straight_short = calibration & (hops <= hop_low) & (curvature <= curve_low)
    curved_long = test & (hops >= hop_high) & (curvature >= curve_high)
    straight_long = test & (hops >= hop_high) & (curvature <= curve_low)
    calibration_straight_long = calibration & (hops >= hop_high) & (curvature <= curve_low)
    if min(straight_short.sum(), curved_long.sum(), straight_long.sum(),
           calibration_straight_long.sum()) < 30:
        raise ValueError("insufficient fixed diagnostic groups")
    alpha = float(np.median(length[straight_short] / g[straight_short]))
    alpha_long = float(np.median(length[calibration_straight_long] /
                                 g[calibration_straight_long]))
    path_error = np.abs(alpha * g[curved_long] - length[curved_long])
    chord_error = np.abs(alpha * g[curved_long] - chord[curved_long])
    advantage = chord_error - path_error

    # Cluster bootstrap by source, since targets sharing one predecessor tree are dependent.
    test_source_ids = np.unique(src[curved_long].astype(np.int64))
    boot = np.random.default_rng(SEED + 1)
    boot_medians = []
    for _ in range(1000):
        sampled = boot.choice(test_source_ids, size=len(test_source_ids), replace=True)
        values = np.concatenate([advantage[src[curved_long] == s] for s in sampled])
        boot_medians.append(np.median(values))
    low_ci, high_ci = np.quantile(boot_medians, [0.025, 0.975])
    signed_stretch = (chart_chord - chord) / chord
    long_test = test & (hops >= hop_high)
    global_rng = np.random.default_rng(17)
    global_left = global_rng.integers(len(positions), size=60000)
    global_right = global_rng.integers(len(positions), size=60000)
    nonself = global_left != global_right
    global_left, global_right = global_left[nonself], global_right[nonself]
    global_geo = np.asarray(geo[global_left, global_right], dtype=np.float64)
    global_physical = np.linalg.norm(positions[global_left] - positions[global_right], axis=1)
    global_chart = np.linalg.norm(aligned[global_left] - aligned[global_right], axis=1)
    quantiles = np.quantile(global_geo, [0, 0.25, 0.5, 0.75, 1.0])
    global_bins = []
    for bin_idx in range(4):
        mask = (global_geo >= quantiles[bin_idx]) & (global_geo < quantiles[bin_idx + 1])
        global_bins.append({"dgeo_range": [float(quantiles[bin_idx]),
                                             float(quantiles[bin_idx + 1])],
                            "median_physical_chord_over_dgeo": float(np.median(
                                global_physical[mask] / global_geo[mask]))})
    timestamps = np.load(ROOT / "datasets/adp_old_shard4/timestamps.npy", mmap_mode="r")
    timestamps = np.asarray(timestamps[saved["indices"]], dtype=np.float64)
    order = np.argsort(timestamps)
    dt = np.diff(timestamps[order])
    valid_steps = (dt > 0) & (dt <= 1.0)
    valid_triplets = valid_steps[:-1] & valid_steps[1:]
    identity_saved = np.load(ROOT / "results/g1_acf_graph_fcf/stride1_n20997_seed10_identity_full/embedding.npz")
    if not np.array_equal(identity_saved["indices"], saved["indices"]):
        raise ValueError("Identity indices do not match Graph indices")
    identity_aligned = affine_transform(positions, identity_saved["embedding"])
    temporal = {"valid_steps": int(valid_steps.sum()),
                "valid_triplets": int(valid_triplets.sum()),
                "timestamp_dt_quantiles_s": [float(v) for v in np.quantile(dt, [0.5, 0.99, 0.999])]}
    for name, coords in (("gt", positions), ("graph", aligned), ("identity", identity_aligned)):
        speed = np.diff(np.asarray(coords[order], dtype=np.float64), axis=0) / dt[:, None]
        acceleration = np.linalg.norm(np.diff(speed, axis=0), axis=1)
        temporal[name] = {"median_velocity_change": float(np.median(acceleration[valid_triplets])),
                          "p95_velocity_change": float(np.quantile(acceleration[valid_triplets], 0.95))}
    result = {
        "status": "offline_diagnostic_only",
        "seed": SEED,
        "source_count": N_SOURCES,
        "targets_per_source": TARGETS_PER_SOURCE,
        "retained_pairs": int(len(a)),
        "calibration_pairs": int(straight_short.sum()),
        "calibration_straight_long_pairs": int(calibration_straight_long.sum()),
        "curved_long_test_pairs": int(curved_long.sum()),
        "straight_long_test_pairs": int(straight_long.sum()),
        "group_thresholds_from_calibration": {
            "hop_q25": float(hop_low), "hop_q75": float(hop_high),
            "curvature_q25": float(curve_low), "curvature_q75": float(curve_high),
        },
        "dgeo_to_meter_alpha_offline_only": alpha,
        "dgeo_to_meter_alpha_long_straight_sensitivity": alpha_long,
        "max_recomputed_vs_saved_dgeo": max_geo_diff,
        "max_path_edge_cost_sum_vs_saved_dgeo": max_path_cost_diff,
        "curved_long": {
            "median_alpha_dgeo_m": float(np.median(alpha * g[curved_long])),
            "median_gt_path_m": float(np.median(length[curved_long])),
            "median_gt_chord_m": float(np.median(chord[curved_long])),
            "median_gt_curvature": float(np.median(curvature[curved_long])),
            "median_abs_alpha_dgeo_minus_gt_path_m": float(np.median(path_error)),
            "median_abs_alpha_dgeo_minus_gt_chord_m": float(np.median(chord_error)),
            "median_paired_path_advantage_m": float(np.median(advantage)),
            "source_cluster_bootstrap_95pct_ci_m": [float(low_ci), float(high_ci)],
            "fraction_dgeo_closer_to_gt_path": float(np.mean(path_error < chord_error)),
            "long_straight_alpha_sensitivity": {
                "median_abs_alpha_dgeo_minus_gt_path_m": float(np.median(
                    np.abs(alpha_long * g[curved_long] - length[curved_long]))),
                "median_abs_alpha_dgeo_minus_gt_chord_m": float(np.median(
                    np.abs(alpha_long * g[curved_long] - chord[curved_long]))),
            },
            "median_g1_chord_stretch_ratio": float(np.median(signed_stretch[curved_long])),
            "median_g1_raw_path_over_dgeo": float(np.median(chart_path[curved_long] / g[curved_long])),
            "edge_deficit_decomposition_short_straight_alpha": {
                "all_edges_deficit_m": float(np.sum(length[curved_long] - alpha * g[curved_long])),
                "gt_0p5m_edge_fraction": float(np.sum(count_gt05[curved_long]) /
                                                np.sum(hops[curved_long])),
                "gt_0p5m_edges_deficit_m": float(np.sum(
                    phys_gt05[curved_long] - alpha * cost_gt05[curved_long])),
                "gt_1m_edge_fraction": float(np.sum(count_gt1[curved_long]) /
                                              np.sum(hops[curved_long])),
                "gt_1m_edges_deficit_m": float(np.sum(
                    phys_gt1[curved_long] - alpha * cost_gt1[curved_long])),
            },
        },
        "straight_long": {
            "median_gt_curvature": float(np.median(curvature[straight_long])),
            "median_g1_chord_stretch_ratio": float(np.median(signed_stretch[straight_long])),
        },
        "long_test_spearman_gt_curvature_vs_signed_g1_chord_stretch": float(
            spearmanr(curvature[long_test], signed_stretch[long_test]).statistic),
        "long_test_spearman_g1_path_residual_vs_signed_chord_stretch": float(
            spearmanr(np.abs(chart_path[long_test] / g[long_test] - 1),
                      signed_stretch[long_test]).statistic),
        "global_pair_diagnostic": {
            "seed": 17, "nonself_pairs": int(len(global_left)),
            "spearman_dgeo_vs_physical_chord": float(spearmanr(global_geo, global_physical).statistic),
            "spearman_g1_aligned_chord_vs_physical_chord": float(spearmanr(
                global_chart, global_physical).statistic),
            "distance_quartiles": global_bins,
        },
        "temporal_diagnostic_offline_gt_only": temporal,
        "inputs": {"embedding": str(EMBED), "distance_run": str(DIST)},
        "gt_use": "offline physical path/chord, global rank, and temporal diagnosis only; no training or checkpoint selection",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
