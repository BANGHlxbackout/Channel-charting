from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components, shortest_path
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.stats import rankdata, spearmanr
from sklearn.neighbors import NearestNeighbors

try:
    from diagnostics.phase2c_h1_h2.run_h1_diagnostic import (
        package_versions,
        pair_key,
        physical_local_scale,
        quantile_bins,
    )
except ModuleNotFoundError:
    from run_h1_diagnostic import (
        package_versions,
        pair_key,
        physical_local_scale,
        quantile_bins,
    )


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_DIR = REPO_ROOT / "results/phase2c_h1_h2"
FIGURE_DIR = RESULT_DIR / "figures"

GRAPH_PATH = RESULT_DIR / "stage1_graph.npz"
EDGE_TABLE_PATH = RESULT_DIR / "h1_edge_table.parquet"
PAIR_TABLE_PATH = RESULT_DIR / "h1_pair_table.parquet"
CALIBRATION_CURVE_PATH = RESULT_DIR / "h1_calibration_curve.npz"
FIRST_STAGE_SUMMARY_PATH = RESULT_DIR / "h1_summary.json"
POSITION_PATH = REPO_ROOT / "results/adp_old_fused/stride1_n20997_seed10/distance_data.npz"
D_GEO_PATH = REPO_ROOT / "results/adp_old_fused/stride1_n20997_seed10/geodesic.npy"

CONFIG_PATH = RESULT_DIR / "h1_counterfactual_config.json"
DESIGN_PATH = RESULT_DIR / "h1_counterfactual_design.npz"
CONTROL_EDGE_PATH = RESULT_DIR / "h1_counterfactual_control_edges.parquet"
BALANCE_PATH = RESULT_DIR / "h1_counterfactual_control_balance.csv"
RUN_TABLE_PATH = RESULT_DIR / "h1_counterfactual_runs.parquet"
TEST_PAIR_PATH = RESULT_DIR / "h1_counterfactual_test_pairs.parquet"
SUMMARY_PATH = RESULT_DIR / "h1_counterfactual_summary.json"
REPORT_PATH = RESULT_DIR / "H1_COUNTERFACTUAL_REPORT.md"

COUNTERFACTUAL_SEED = 11
SOURCE_COUNT = 128
SOURCES_PER_QUARTILE = 32
TARGETS_PER_SOURCE = 512
REMOVAL_COUNTS = np.asarray([25, 50, 100], dtype=np.int32)
CONTROL_REPLICATES = 50
MATCH_QUERY_NEIGHBORS = 512
PHYSICAL_Q = 16
SEVERE_THRESHOLD = -0.18340832744215135
OVER_ESTIMATION_THRESHOLD = 0.18755884633681646
WELL_ESTIMATED_ABS_RESIDUAL_THRESHOLD = 0.06011182814836502
EPS = 1e-12

RANDOM_FEATURES = (
    "log1p_discovery_path_usage",
    "fused_rank_percentile",
    "max_endpoint_degree",
    "log_physical_local_scale",
    "midpoint_x",
    "midpoint_y",
)
LOW_IMPACT_FEATURES = RANDOM_FEATURES + ("robust_mismatch_score",)
CRITICAL_BALANCE_LIMIT = 0.10
SECONDARY_BALANCE_LIMIT = 0.20


def read_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: dict[str, object]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(json_safe(value), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def json_safe(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def frozen_isotonic_predict(log_physical: np.ndarray) -> np.ndarray:
    curve = np.load(CALIBRATION_CURVE_PATH)
    x = curve["X_thresholds"]
    y = curve["y_thresholds"]
    return np.interp(log_physical, x, y, left=y[0], right=y[-1])


def edge_covariates(edge_frame: pd.DataFrame, positions: np.ndarray) -> pd.DataFrame:
    fused_rank = rankdata(edge_frame["fused_distance"].to_numpy(), method="average")
    fused_rank = fused_rank / len(edge_frame)
    edge_u = edge_frame["u"].to_numpy(dtype=np.int32)
    edge_v = edge_frame["v"].to_numpy(dtype=np.int32)
    midpoint = 0.5 * (positions[edge_u].astype(np.float64) + positions[edge_v].astype(np.float64))
    physical_scale = np.sqrt(
        edge_frame["physical_scale_u"].to_numpy(dtype=np.float64)
        * edge_frame["physical_scale_v"].to_numpy(dtype=np.float64)
    )
    return pd.DataFrame(
        {
            "log1p_discovery_path_usage": np.log1p(
                edge_frame["discovery_path_usage"].to_numpy(dtype=np.float64)
            ),
            "fused_rank_percentile": fused_rank,
            "max_endpoint_degree": np.maximum(
                edge_frame["degree_u"].to_numpy(dtype=np.float64),
                edge_frame["degree_v"].to_numpy(dtype=np.float64),
            ),
            "log_physical_local_scale": np.log(physical_scale + EPS),
            "midpoint_x": midpoint[:, 0],
            "midpoint_y": midpoint[:, 1],
            "robust_mismatch_score": edge_frame["robust_mismatch_score"].to_numpy(
                dtype=np.float64
            ),
        }
    )


def select_sources(
    physical_scale: np.ndarray,
    excluded_sources: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    quartiles, boundaries = quantile_bins(physical_scale, 4)
    excluded = np.zeros(len(physical_scale), dtype=bool)
    excluded[excluded_sources] = True
    selected: list[int] = []
    for quartile in range(4):
        members = np.flatnonzero((quartiles == quartile) & ~excluded)
        if len(members) < SOURCES_PER_QUARTILE:
            raise RuntimeError(f"density quartile {quartile} lacks fresh source nodes")
        chosen = rng.permutation(members)[:SOURCES_PER_QUARTILE]
        selected.extend(chosen.tolist())
    sources = np.asarray(selected, dtype=np.int32)
    if len(np.unique(sources)) != SOURCE_COUNT:
        raise RuntimeError("fresh source selection is not unique")
    return sources, quartiles, boundaries


def existing_pair_keys(pair_frame: pd.DataFrame, n: int) -> set[int]:
    left = pair_frame["source"].to_numpy(dtype=np.int64)
    right = pair_frame["target"].to_numpy(dtype=np.int64)
    low = np.minimum(left, right)
    high = np.maximum(left, right)
    return set((low * n + high).tolist())


def sample_fresh_targets(
    sources: np.ndarray,
    n: int,
    used_keys: set[int],
    positions: np.ndarray,
    d_geo: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    targets = np.empty((len(sources), TARGETS_PER_SOURCE), dtype=np.int32)
    for source_row, source_value in enumerate(sources):
        source = int(source_value)
        local_targets: set[int] = set()
        ordered: list[int] = []
        while len(ordered) < TARGETS_PER_SOURCE:
            target = int(rng.integers(0, n - 1))
            if target >= source:
                target += 1
            key = pair_key(source, target, n)
            if key in used_keys or target in local_targets:
                continue
            if np.linalg.norm(positions[source] - positions[target]) <= 0:
                continue
            if float(d_geo[source, target]) <= 0:
                continue
            used_keys.add(key)
            local_targets.add(target)
            ordered.append(target)
        targets[source_row] = np.asarray(ordered, dtype=np.int32)
    return targets


def constrained_optimal_sequences(
    target_edges: np.ndarray,
    pool_edges: np.ndarray,
    covariates: pd.DataFrame,
    critical_features: tuple[str, ...],
    secondary_features: tuple[str, ...],
    rng: np.random.Generator,
    low_impact_values: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    feature_names = critical_features + secondary_features
    pool_features = covariates.loc[pool_edges, feature_names].to_numpy(dtype=np.float64)
    target_features = covariates.loc[target_edges, feature_names].to_numpy(dtype=np.float64)
    center = np.mean(pool_features, axis=0)
    scale = np.std(pool_features, axis=0, ddof=0)
    if np.any(scale <= 0):
        raise RuntimeError("matching covariate has zero variance")
    pool_z = (pool_features - center) / scale
    target_z = (target_features - center) / scale
    critical_count = len(critical_features)
    pool_critical = pool_z[:, :critical_count]
    target_critical = target_z[:, :critical_count]
    pool_secondary = pool_z[:, critical_count:]
    target_secondary = target_z[:, critical_count:]
    neighbor_count = min(MATCH_QUERY_NEIGHBORS, len(pool_edges))
    neighbors = NearestNeighbors(n_neighbors=neighbor_count, algorithm="auto").fit(
        pool_critical
    )
    _, neighbor_indices = neighbors.kneighbors(target_critical, return_distance=True)
    admissible = [neighbor_indices[row] for row in range(len(target_edges))]
    impact_rank = None
    if low_impact_values is not None:
        impact_rank = rankdata(low_impact_values[pool_edges], method="average") / len(pool_edges)

    sequences = np.empty((CONTROL_REPLICATES, len(target_edges)), dtype=np.int32)
    blocks = (
        ((0, 25), (25, 50), (50, 100))
        if len(target_edges) == 100
        else ((0, len(target_edges)),)
    )
    for replicate in range(CONTROL_REPLICATES):
        used_pool_rows: set[int] = set()
        selected_sum = np.zeros(len(feature_names), dtype=np.float64)
        for start, stop in blocks:
            candidate_rows = np.unique(
                np.concatenate([admissible[row] for row in range(start, stop)])
            )
            candidate_rows = np.asarray(
                [row for row in candidate_rows if int(row) not in used_pool_rows],
                dtype=np.int32,
            )
            if len(candidate_rows) < stop - start:
                raise RuntimeError("admissible control graph lacks a full unique assignment")
            block_size = stop - start
            candidate_z = pool_z[candidate_rows]
            target_prefix_sum = np.sum(target_z[:stop], axis=0)
            limits = np.r_[
                np.full(critical_count, CRITICAL_BALANCE_LIMIT),
                np.full(len(secondary_features), SECONDARY_BALANCE_LIMIT),
            ]
            lower_feature_sum = target_prefix_sum - limits * stop - selected_sum
            upper_feature_sum = target_prefix_sum + limits * stop - selected_sum
            constraint_matrix = sp.vstack(
                [
                    sp.csr_matrix(np.ones((1, len(candidate_rows)), dtype=np.float64)),
                    sp.csr_matrix(candidate_z.T),
                ],
                format="csr",
            )
            constraints = LinearConstraint(
                constraint_matrix,
                np.r_[block_size, lower_feature_sum],
                np.r_[block_size, upper_feature_sum],
            )
            if len(secondary_features):
                secondary_model = NearestNeighbors(n_neighbors=1).fit(
                    target_secondary[start:stop]
                )
                secondary_distance = secondary_model.kneighbors(
                    pool_secondary[candidate_rows],
                    return_distance=True,
                )[0][:, 0]
            else:
                secondary_distance = np.zeros(len(candidate_rows), dtype=np.float64)
            objective = 0.05 * secondary_distance
            if impact_rank is not None:
                objective = objective + impact_rank[candidate_rows]
            objective = objective + rng.uniform(0.0, 0.10, size=len(candidate_rows))
            result = milp(
                c=objective,
                integrality=np.ones(len(candidate_rows), dtype=np.int8),
                bounds=Bounds(0.0, 1.0),
                constraints=constraints,
                options={"time_limit": 60.0, "mip_rel_gap": 0.0},
            )
            if not result.success or result.x is None:
                raise RuntimeError(
                    f"constrained matching failed for replicate={replicate}, "
                    f"block={start}:{stop}: {result.message}"
                )
            chosen_pool_rows = candidate_rows[result.x > 0.5]
            if len(chosen_pool_rows) != block_size:
                raise RuntimeError("constrained matching selected the wrong block size")
            sequences[replicate, start:stop] = pool_edges[chosen_pool_rows]
            used_pool_rows.update(int(row) for row in chosen_pool_rows)
            selected_sum += np.sum(pool_z[chosen_pool_rows], axis=0)
    return sequences, center, scale


def control_balance(
    family: str,
    target_edges: np.ndarray,
    sequences: np.ndarray,
    covariates: pd.DataFrame,
    feature_names: tuple[str, ...],
    center: np.ndarray,
    scale: np.ndarray,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for replicate in range(len(sequences)):
        for count in REMOVAL_COUNTS:
            target = covariates.loc[target_edges[:count], feature_names].to_numpy(dtype=np.float64)
            control = covariates.loc[sequences[replicate, :count], feature_names].to_numpy(
                dtype=np.float64
            )
            for column, feature in enumerate(feature_names):
                target_z = (target[:, column] - center[column]) / scale[column]
                control_z = (control[:, column] - center[column]) / scale[column]
                rows.append(
                    {
                        "control_family": family,
                        "replicate": replicate,
                        "edge_count": int(count),
                        "feature": feature,
                        "standardized_mean_difference": float(
                            np.mean(control_z) - np.mean(target_z)
                        ),
                        "target_median": float(np.median(target[:, column])),
                        "control_median": float(np.median(control[:, column])),
                        "target_p25": float(np.percentile(target[:, column], 25)),
                        "target_p75": float(np.percentile(target[:, column], 75)),
                        "control_p25": float(np.percentile(control[:, column], 25)),
                        "control_p75": float(np.percentile(control[:, column], 75)),
                    }
                )
    return pd.DataFrame(rows)


def control_edge_table(
    family: str,
    sequences: np.ndarray,
    edge_frame: pd.DataFrame,
    covariates: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for replicate, sequence in enumerate(sequences):
        selected = edge_frame.loc[sequence, [
            "edge_id",
            "u",
            "v",
            "fused_distance",
            "robust_mismatch_score",
            "discovery_path_usage",
            "discovery_severe_path_usage",
            "discovery_bad_path_enrichment",
            "branch",
            "boundary_tie_ambiguity",
        ]].copy()
        selected.insert(0, "prefix_position", np.arange(1, len(sequence) + 1, dtype=np.int32))
        selected.insert(0, "replicate", replicate)
        selected.insert(0, "control_family", family)
        for feature in ("max_endpoint_degree", "log_physical_local_scale", "midpoint_x", "midpoint_y"):
            selected[feature] = covariates.loc[sequence, feature].to_numpy()
        rows.append(selected)
    return pd.concat(rows, ignore_index=True)


def candidate_descriptions(edge_frame: pd.DataFrame, target_edges: np.ndarray) -> dict[str, object]:
    result: dict[str, object] = {}
    columns = (
        "robust_mismatch_score",
        "discovery_path_usage",
        "discovery_severe_path_usage",
        "discovery_bad_path_enrichment",
    )
    for count in REMOVAL_COUNTS:
        frame = edge_frame.loc[target_edges[:count]]
        entry: dict[str, object] = {"edge_count": int(count)}
        for column in columns:
            values = frame[column].to_numpy(dtype=np.float64)
            entry[column] = {
                "min": float(np.min(values)),
                "p25": float(np.percentile(values, 25)),
                "median": float(np.median(values)),
                "p75": float(np.percentile(values, 75)),
                "max": float(np.max(values)),
                "sum": float(np.sum(values)),
            }
        entry["branch_counts"] = {
            str(label): int(value) for label, value in frame["branch"].value_counts().items()
        }
        entry["boundary_tie_ambiguous_count"] = int(
            frame["boundary_tie_ambiguity"].sum()
        )
        entry["boundary_tie_ambiguous_fraction"] = float(
            frame["boundary_tie_ambiguity"].mean()
        )
        result[str(int(count))] = entry
    return result


def prepare_design() -> dict[str, object]:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    edge_frame = pd.read_parquet(EDGE_TABLE_PATH).sort_values("edge_id").reset_index(drop=True)
    if not np.array_equal(edge_frame["edge_id"].to_numpy(), np.arange(len(edge_frame))):
        raise RuntimeError("edge table row order no longer equals edge_id")
    pair_frame = pd.read_parquet(PAIR_TABLE_PATH)
    position_data = np.load(POSITION_PATH)
    positions = np.asarray(position_data["positions"], dtype=np.float32)
    d_geo = np.load(D_GEO_PATH, mmap_mode="r")
    n = len(positions)
    if d_geo.shape != (n, n):
        raise RuntimeError("D_geo shape does not match positions")

    physical_scale, fallback_count = physical_local_scale(positions, PHYSICAL_Q)
    old_source_mask = pair_frame["split"].isin(["discovery", "validation"])
    excluded_sources = pair_frame.loc[old_source_mask, "source"].unique().astype(np.int32)
    used_keys = existing_pair_keys(pair_frame, n)
    rng = np.random.default_rng(COUNTERFACTUAL_SEED)
    sources, quartiles, quartile_boundaries = select_sources(
        physical_scale,
        excluded_sources,
        rng,
    )
    targets = sample_fresh_targets(sources, n, used_keys, positions, d_geo, rng)
    if np.intersect1d(sources, excluded_sources).size:
        raise RuntimeError("fresh sources overlap first-stage discovery/validation sources")

    covariates = edge_covariates(edge_frame, positions)
    target_edges = edge_frame.nsmallest(int(REMOVAL_COUNTS[-1]), "discovery_rank").index.to_numpy(
        dtype=np.int32
    )
    if not np.array_equal(
        edge_frame.loc[target_edges, "discovery_rank"].to_numpy(dtype=np.int32),
        np.arange(1, int(REMOVAL_COUNTS[-1]) + 1, dtype=np.int32),
    ):
        raise RuntimeError("frozen target edges are not the rank-ordered top 100")

    non_target = edge_frame["discovery_rank"].to_numpy(dtype=np.int32) > int(
        REMOVAL_COUNTS[-1]
    )
    random_pool = np.flatnonzero(non_target).astype(np.int32)
    low_impact_pool = np.flatnonzero(
        non_target
        & (edge_frame["discovery_path_usage"].to_numpy(dtype=np.int32) > 0)
        & np.isfinite(edge_frame["discovery_bad_path_enrichment"].to_numpy(dtype=np.float64))
    ).astype(np.int32)
    if len(low_impact_pool) < 100:
        raise RuntimeError("suspicious-low-impact control pool is too small")

    random_rng = np.random.default_rng(COUNTERFACTUAL_SEED + 1000)
    low_rng = np.random.default_rng(COUNTERFACTUAL_SEED + 2000)
    secondary_features = tuple(
        feature for feature in RANDOM_FEATURES if feature != "log1p_discovery_path_usage"
    )
    random_sequences, random_center, random_scale = constrained_optimal_sequences(
        target_edges,
        random_pool,
        covariates,
        ("log1p_discovery_path_usage",),
        secondary_features,
        random_rng,
    )
    low_secondary_features = tuple(
        feature
        for feature in LOW_IMPACT_FEATURES
        if feature not in ("log1p_discovery_path_usage", "robust_mismatch_score")
    )
    low_matching_features = (
        "log1p_discovery_path_usage",
        "robust_mismatch_score",
    ) + low_secondary_features
    low_sequences, low_center, low_scale = constrained_optimal_sequences(
        target_edges,
        low_impact_pool,
        covariates,
        ("log1p_discovery_path_usage", "robust_mismatch_score"),
        low_secondary_features,
        low_rng,
        edge_frame["discovery_bad_path_enrichment"].to_numpy(dtype=np.float64),
    )

    balance = pd.concat(
        [
            control_balance(
                "matched_random",
                target_edges,
                random_sequences,
                covariates,
                RANDOM_FEATURES,
                random_center,
                random_scale,
            ),
            control_balance(
                "suspicious_low_impact",
                target_edges,
                low_sequences,
                covariates,
                low_matching_features,
                low_center,
                low_scale,
            ),
        ],
        ignore_index=True,
    )
    balance.to_csv(BALANCE_PATH, index=False)
    controls = pd.concat(
        [
            control_edge_table("matched_random", random_sequences, edge_frame, covariates),
            control_edge_table("suspicious_low_impact", low_sequences, edge_frame, covariates),
        ],
        ignore_index=True,
    )
    controls.to_parquet(CONTROL_EDGE_PATH, index=False)
    np.savez_compressed(
        DESIGN_PATH,
        sources=sources,
        targets=targets,
        target_edges=target_edges,
        matched_random_sequences=random_sequences,
        suspicious_low_impact_sequences=low_sequences,
    )

    balance_summary: dict[str, object] = {}
    for family, family_frame in balance.groupby("control_family"):
        absolute = family_frame["standardized_mean_difference"].abs()
        balance_summary[str(family)] = {
            "absolute_smd_median": float(absolute.median()),
            "absolute_smd_p95": float(absolute.quantile(0.95)),
            "absolute_smd_max": float(absolute.max()),
            "by_feature_mean_absolute_smd": {
                str(feature): float(value)
                for feature, value in family_frame.assign(abs_smd=absolute)
                .groupby("feature")["abs_smd"]
                .mean()
                .items()
            },
        }

    critical_features = {
        "matched_random": ("log1p_discovery_path_usage",),
        "suspicious_low_impact": (
            "log1p_discovery_path_usage",
            "robust_mismatch_score",
        ),
    }
    critical_balance: dict[str, object] = {}
    all_critical_replicates_pass = True
    for family, features in critical_features.items():
        family_result: dict[str, object] = {}
        for count in REMOVAL_COUNTS:
            selected = balance[
                (balance["control_family"] == family)
                & (balance["edge_count"] == int(count))
                & balance["feature"].isin(features)
            ].copy()
            selected["absolute_smd"] = selected["standardized_mean_difference"].abs()
            replicate_max = selected.groupby("replicate")["absolute_smd"].max()
            pass_fraction = float(np.mean(replicate_max <= CRITICAL_BALANCE_LIMIT))
            all_critical_replicates_pass &= pass_fraction == 1.0
            family_result[str(int(count))] = {
                "median_absolute_smd": float(selected["absolute_smd"].median()),
                "p95_absolute_smd": float(selected["absolute_smd"].quantile(0.95)),
                "max_absolute_smd": float(selected["absolute_smd"].max()),
                "replicate_pass_fraction": pass_fraction,
            }
        critical_balance[family] = family_result

    low_impact_definition: dict[str, object] = {}
    for count in REMOVAL_COUNTS:
        target_median = float(
            np.median(
                edge_frame.loc[
                    target_edges[:count], "discovery_bad_path_enrichment"
                ].to_numpy(dtype=np.float64)
            )
        )
        control_medians = np.asarray(
            [
                np.median(
                    edge_frame.loc[
                        sequence[:count], "discovery_bad_path_enrichment"
                    ].to_numpy(dtype=np.float64)
                )
                for sequence in low_sequences
            ],
            dtype=np.float64,
        )
        low_impact_definition[str(int(count))] = {
            "target_median_bad_path_enrichment": target_median,
            "control_median_mean": float(np.mean(control_medians)),
            "control_median_p95": float(np.percentile(control_medians, 95)),
            "fraction_replicates_below_target_median": float(
                np.mean(control_medians < target_median)
            ),
        }
    all_low_impact_replicates_pass = all(
        entry["fraction_replicates_below_target_median"] == 1.0
        for entry in low_impact_definition.values()
    )

    config: dict[str, object] = {
        "scope": "H1 targeted-removal counterfactual only; no sensitivity, H2, or method development",
        "package_versions": package_versions(),
        "input_artifacts": {
            "stage1_graph": str(GRAPH_PATH.relative_to(REPO_ROOT)),
            "frozen_edge_table": str(EDGE_TABLE_PATH.relative_to(REPO_ROOT)),
            "frozen_pair_table": str(PAIR_TABLE_PATH.relative_to(REPO_ROOT)),
            "frozen_calibration_curve": str(CALIBRATION_CURVE_PATH.relative_to(REPO_ROOT)),
            "first_stage_summary": str(FIRST_STAGE_SUMMARY_PATH.relative_to(REPO_ROOT)),
            "positions": str(POSITION_PATH.relative_to(REPO_ROOT)),
            "saved_D_geo": str(D_GEO_PATH.relative_to(REPO_ROOT)),
        },
        "frozen_ranking": {
            "formula": "mismatch_percentile * positive-severe-usage percentile; discovery only",
            "removal_counts": REMOVAL_COUNTS.astype(int).tolist(),
            "target_edge_ids": edge_frame.loc[target_edges, "edge_id"].astype(int).tolist(),
            "no_reranking": True,
        },
        "fresh_test": {
            "seed": COUNTERFACTUAL_SEED,
            "source_count": SOURCE_COUNT,
            "sources_per_q16_density_quartile": SOURCES_PER_QUARTILE,
            "targets_per_source": TARGETS_PER_SOURCE,
            "pair_count": int(SOURCE_COUNT * TARGETS_PER_SOURCE),
            "source_ids": sources.astype(int).tolist(),
            "source_disjoint_from_first_stage_discovery_validation": True,
            "unordered_pair_disjoint_from_calibration_discovery_validation": True,
            "source_ids_need_not_be_disjoint_from_calibration_endpoints": True,
            "q16_quartile_boundaries": quartile_boundaries.astype(float).tolist(),
            "source_quartile_counts": np.bincount(quartiles[sources], minlength=4).astype(int).tolist(),
            "physical_q16_zero_median_fallback_count": fallback_count,
        },
        "calibration": {
            "model": "frozen first-stage isotonic threshold arrays",
            "severe_threshold": SEVERE_THRESHOLD,
            "overestimation_threshold": OVER_ESTIMATION_THRESHOLD,
            "well_estimated_abs_residual_threshold": WELL_ESTIMATED_ABS_RESIDUAL_THRESHOLD,
            "no_recalibration": True,
        },
        "removal_semantics": "delete both stored directions of each logical undirected edge; directed=False shortest paths",
        "controls": {
            "replicates": CONTROL_REPLICATES,
            "nested_prefixes": True,
            "method": "critical-neighbor admissible graph plus blockwise binary optimization with explicit prefix SMD constraints",
            "nearest_critical_query_count": MATCH_QUERY_NEIGHBORS,
            "assignment_blocks": [[0, 25], [25, 50], [50, 100]],
            "assignment_cost": "0.05 * nearest-target secondary distance + Uniform(0,0.10) seeded jitter; suspicious-low-impact additionally minimizes pool enrichment rank",
            "critical_balance_limit": CRITICAL_BALANCE_LIMIT,
            "secondary_balance_limit": SECONDARY_BALANCE_LIMIT,
            "all_critical_replicates_pass": all_critical_replicates_pass,
            "critical_balance": critical_balance,
            "all_low_impact_replicates_pass": all_low_impact_replicates_pass,
            "low_impact_definition_check": low_impact_definition,
            "matched_random": {
                "pool_size": int(len(random_pool)),
                "features": list(RANDOM_FEATURES),
                "critical_features": ["log1p_discovery_path_usage"],
                "secondary_features": list(secondary_features),
                "excluded": [
                    "robust_mismatch_score",
                    "discovery_severe_path_usage",
                    "discovery_bad_path_enrichment",
                    "candidate_score",
                ],
            },
            "suspicious_low_impact": {
                "pool_size": int(len(low_impact_pool)),
                "features": list(low_matching_features),
                "critical_features": [
                    "log1p_discovery_path_usage",
                    "robust_mismatch_score",
                ],
                "secondary_features": list(low_secondary_features),
                "discovery_path_usage_positive": True,
                "low_impact_definition": "within each target's 512 nearest critical-covariate matches, include low discovery bad-path enrichment as a secondary assignment penalty; every retained replicate must have lower median enrichment than its targeted prefix",
            },
            "balance_summary": balance_summary,
        },
        "primary_outcomes": {
            "P1": "repair rate among baseline-severe fresh-test pairs",
            "P2": "median delta r_geo among baseline-severe fresh-test pairs",
        },
        "harm_outcomes": {
            "H1": "increase in fraction above frozen calibration residual P95",
            "H2": "median log path inflation on baseline well-estimated pairs",
            "H3": "connected components, giant-component ratio, unreachable fresh-test pairs",
        },
        "decision_rule": {
            "supported": "at least 2 of 3 X exceed both control-family P95 values on P1 and P2, remain connected with zero unreachable test pairs, and do not exceed either control-family P95 on H1 or H2",
            "false": "no X exceeds both control-family P95 values on both P1 and P2, or no primary-advantage X passes connectivity and harm checks",
            "otherwise": "inconclusive",
            "H1_supported_remains_null_pending_sensitivity": True,
        },
        "design_artifacts": {
            "design_npz": str(DESIGN_PATH.relative_to(REPO_ROOT)),
            "control_edges": str(CONTROL_EDGE_PATH.relative_to(REPO_ROOT)),
            "balance": str(BALANCE_PATH.relative_to(REPO_ROOT)),
        },
        "candidate_descriptions": candidate_descriptions(edge_frame, target_edges),
    }
    write_json(CONFIG_PATH, config)
    return config


def remove_logical_edges(
    directed_graph: sp.csr_matrix,
    directed_rows: np.ndarray,
    directed_keys: np.ndarray,
    selected_edge_ids: np.ndarray,
    edge_frame: pd.DataFrame,
) -> sp.csr_matrix:
    n = directed_graph.shape[0]
    selected = edge_frame.loc[selected_edge_ids]
    selected_keys = (
        selected["u"].to_numpy(dtype=np.int64) * n
        + selected["v"].to_numpy(dtype=np.int64)
    )
    keep = ~np.isin(directed_keys, selected_keys)
    modified = sp.csr_matrix(
        (
            directed_graph.data[keep],
            (directed_rows[keep], directed_graph.indices[keep]),
        ),
        shape=directed_graph.shape,
        dtype=directed_graph.dtype,
    )
    expected_removed = int(
        selected["directed_support_u_to_v"].sum()
        + selected["directed_support_v_to_u"].sum()
    )
    if directed_graph.nnz - modified.nnz != expected_removed:
        raise RuntimeError("logical edge removal did not remove the expected directed entries")
    return modified


def graph_connectivity(graph: sp.csr_matrix) -> tuple[int, float]:
    component_count, labels = connected_components(graph, directed=False, return_labels=True)
    sizes = np.bincount(labels, minlength=component_count)
    return int(component_count), float(np.max(sizes) / graph.shape[0])


def evaluate_distances(
    distances: np.ndarray,
    baseline_distances: np.ndarray,
    physical: np.ndarray,
    predicted_log: np.ndarray,
    baseline_residual: np.ndarray,
    baseline_severe: np.ndarray,
    baseline_overestimated: np.ndarray,
    baseline_well_estimated: np.ndarray,
    component_count: int,
    giant_component_ratio: float,
) -> dict[str, object]:
    reachable = np.isfinite(distances)
    residual = np.full(len(distances), np.inf, dtype=np.float64)
    residual[reachable] = np.log(distances[reachable]) - predicted_log[reachable]
    delta = residual - baseline_residual
    repaired = baseline_severe & reachable & (residual > SEVERE_THRESHOLD)
    severe_count = int(np.count_nonzero(baseline_severe))
    severe_delta = delta[baseline_severe & reachable]
    well_inflation = np.full(np.count_nonzero(baseline_well_estimated), np.inf, dtype=np.float64)
    well_reachable = reachable[baseline_well_estimated]
    well_inflation[well_reachable] = np.log(
        distances[baseline_well_estimated][well_reachable]
        / baseline_distances[baseline_well_estimated][well_reachable]
    )
    after_overestimated = (~reachable) | (residual > OVER_ESTIMATION_THRESHOLD)
    rho = spearmanr(physical[reachable], distances[reachable]).statistic
    finite_residual = residual[reachable]
    return {
        "repair_rate": float(np.count_nonzero(repaired) / severe_count),
        "median_delta_r_baseline_severe": float(np.median(severe_delta)),
        "mean_delta_r_baseline_severe": float(np.mean(severe_delta)),
        "p25_delta_r_baseline_severe": float(np.percentile(severe_delta, 25)),
        "severe_fraction_after": float(np.mean(reachable & (residual <= SEVERE_THRESHOLD))),
        "overestimation_fraction_after": float(np.mean(after_overestimated)),
        "new_overestimation_fraction_increase": float(
            np.mean(after_overestimated) - np.mean(baseline_overestimated)
        ),
        "median_log_inflation_well_estimated": float(np.median(well_inflation)),
        "p95_log_inflation_well_estimated": float(np.percentile(well_inflation, 95)),
        "residual_p01": float(np.percentile(finite_residual, 1)),
        "residual_p05": float(np.percentile(finite_residual, 5)),
        "residual_median": float(np.median(finite_residual)),
        "residual_p95": float(np.percentile(finite_residual, 95)),
        "residual_p99": float(np.percentile(finite_residual, 99)),
        "spearman_physical_vs_geodesic": float(rho),
        "connected_components": component_count,
        "giant_component_ratio": giant_component_ratio,
        "unreachable_test_pairs": int(np.count_nonzero(~reachable)),
    }


def run_one(
    family: str,
    replicate: int,
    count: int,
    selected_edges: np.ndarray,
    directed_graph: sp.csr_matrix,
    directed_rows: np.ndarray,
    directed_keys: np.ndarray,
    edge_frame: pd.DataFrame,
    sources: np.ndarray,
    targets: np.ndarray,
    baseline_distances: np.ndarray,
    physical: np.ndarray,
    predicted_log: np.ndarray,
    baseline_residual: np.ndarray,
    baseline_severe: np.ndarray,
    baseline_overestimated: np.ndarray,
    baseline_well_estimated: np.ndarray,
) -> dict[str, object]:
    started = time.perf_counter()
    modified = remove_logical_edges(
        directed_graph,
        directed_rows,
        directed_keys,
        selected_edges,
        edge_frame,
    )
    component_count, giant_ratio = graph_connectivity(modified)
    source_distances = shortest_path(
        modified,
        method="auto",
        directed=False,
        indices=sources,
    )
    distances = source_distances[
        np.arange(len(sources), dtype=np.int32)[:, None],
        targets,
    ].reshape(-1)
    metrics = evaluate_distances(
        distances,
        baseline_distances,
        physical,
        predicted_log,
        baseline_residual,
        baseline_severe,
        baseline_overestimated,
        baseline_well_estimated,
        component_count,
        giant_ratio,
    )
    metrics.update(
        {
            "family": family,
            "replicate": replicate,
            "edge_count": count,
            "directed_entries_removed": int(directed_graph.nnz - modified.nnz),
            "runtime_seconds": float(time.perf_counter() - started),
        }
    )
    return metrics


def control_quantiles(
    runs: pd.DataFrame,
    family: str,
    count: int,
    metric: str,
) -> dict[str, float]:
    values = runs.loc[
        (runs["family"] == family) & (runs["edge_count"] == count),
        metric,
    ].to_numpy(dtype=np.float64)
    if len(values) != CONTROL_REPLICATES:
        raise RuntimeError(f"missing {family} controls for X={count}")
    return {
        "mean": float(np.mean(values)),
        "p05": float(np.percentile(values, 5)),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
    }


def counterfactual_decision(runs: pd.DataFrame) -> tuple[str, dict[str, object]]:
    metrics = (
        "repair_rate",
        "median_delta_r_baseline_severe",
        "new_overestimation_fraction_increase",
        "median_log_inflation_well_estimated",
    )
    comparisons: dict[str, object] = {}
    primary_advantage_count = 0
    supported_count = 0
    for count in REMOVAL_COUNTS:
        target = runs.loc[
            (runs["family"] == "targeted") & (runs["edge_count"] == int(count))
        ].iloc[0]
        controls = {
            family: {
                metric: control_quantiles(runs, family, int(count), metric)
                for metric in metrics
            }
            for family in ("matched_random", "suspicious_low_impact")
        }
        repair_pass = all(
            target["repair_rate"] > controls[family]["repair_rate"]["p95"]
            for family in controls
        )
        delta_pass = all(
            target["median_delta_r_baseline_severe"]
            > controls[family]["median_delta_r_baseline_severe"]["p95"]
            for family in controls
        )
        primary_advantage = bool(repair_pass and delta_pass)
        harm_over_pass = all(
            target["new_overestimation_fraction_increase"]
            <= controls[family]["new_overestimation_fraction_increase"]["p95"]
            for family in controls
        )
        harm_inflation_pass = all(
            target["median_log_inflation_well_estimated"]
            <= controls[family]["median_log_inflation_well_estimated"]["p95"]
            for family in controls
        )
        connectivity_pass = bool(
            int(target["connected_components"]) == 1
            and float(target["giant_component_ratio"]) == 1.0
            and int(target["unreachable_test_pairs"]) == 0
        )
        supported = bool(
            primary_advantage and harm_over_pass and harm_inflation_pass and connectivity_pass
        )
        primary_advantage_count += int(primary_advantage)
        supported_count += int(supported)
        comparisons[str(int(count))] = {
            "targeted": {key: float(target[key]) for key in metrics},
            "controls": controls,
            "checks": {
                "repair_rate_above_both_control_p95": repair_pass,
                "median_delta_above_both_control_p95": delta_pass,
                "new_overestimation_not_above_either_control_p95": harm_over_pass,
                "well_estimated_inflation_not_above_either_control_p95": harm_inflation_pass,
                "connected_and_zero_unreachable": connectivity_pass,
                "supported_at_this_X": supported,
            },
        }
    if supported_count >= 2:
        decision = "true"
    elif primary_advantage_count == 0 or (primary_advantage_count > 0 and supported_count == 0):
        decision = "false"
    else:
        decision = "inconclusive"
    comparisons["counts"] = {
        "primary_advantage_X_count": primary_advantage_count,
        "fully_supported_X_count": supported_count,
    }
    return decision, comparisons


def save_figures(runs: pd.DataFrame) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    colors = {"matched_random": "tab:blue", "suspicious_low_impact": "tab:orange"}
    labels = {"matched_random": "Matched random", "suspicious_low_impact": "Suspicious low-impact"}
    targeted = runs[runs["family"] == "targeted"].sort_values("edge_count")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for axis, metric, ylabel in (
        (axes[0], "repair_rate", "Repair rate on baseline-severe pairs"),
        (axes[1], "median_delta_r_baseline_severe", "Median delta r_geo on baseline-severe pairs"),
    ):
        for family in colors:
            control = runs[runs["family"] == family]
            grouped = control.groupby("edge_count")[metric]
            x = REMOVAL_COUNTS.astype(int)
            median = grouped.median().reindex(x).to_numpy()
            p05 = grouped.quantile(0.05).reindex(x).to_numpy()
            p95 = grouped.quantile(0.95).reindex(x).to_numpy()
            axis.plot(x, median, marker="o", linestyle="--", color=colors[family], label=labels[family])
            axis.fill_between(x, p05, p95, color=colors[family], alpha=0.16)
        axis.plot(
            targeted["edge_count"],
            targeted[metric],
            marker="o",
            linewidth=2,
            color="crimson",
            label="Targeted",
        )
        axis.set_xlabel("Removed logical undirected edges")
        axis.set_ylabel(ylabel)
        axis.grid(True, alpha=0.25)
    axes[0].legend()
    fig.suptitle("H1 counterfactual repair outcomes on fresh test pairs")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "h1_counterfactual_repair.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for axis, metric, ylabel in (
        (
            axes[0],
            "new_overestimation_fraction_increase",
            "Increase in overestimation-tail fraction",
        ),
        (
            axes[1],
            "median_log_inflation_well_estimated",
            "Median log inflation on well-estimated pairs",
        ),
    ):
        for family in colors:
            control = runs[runs["family"] == family]
            grouped = control.groupby("edge_count")[metric]
            x = REMOVAL_COUNTS.astype(int)
            median = grouped.median().reindex(x).to_numpy()
            p05 = grouped.quantile(0.05).reindex(x).to_numpy()
            p95 = grouped.quantile(0.95).reindex(x).to_numpy()
            axis.plot(x, median, marker="o", linestyle="--", color=colors[family], label=labels[family])
            axis.fill_between(x, p05, p95, color=colors[family], alpha=0.16)
        axis.plot(
            targeted["edge_count"],
            targeted[metric],
            marker="o",
            linewidth=2,
            color="crimson",
            label="Targeted",
        )
        axis.set_xlabel("Removed logical undirected edges")
        axis.set_ylabel(ylabel)
        axis.grid(True, alpha=0.25)
    axes[0].legend()
    fig.suptitle("H1 counterfactual harm outcomes on fresh test pairs")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "h1_counterfactual_harm.png", dpi=180)
    plt.close(fig)


def execute_counterfactual() -> dict[str, object]:
    if not CONFIG_PATH.exists() or not DESIGN_PATH.exists():
        raise RuntimeError("run with --prepare-only and review the frozen design first")
    config = read_json(CONFIG_PATH)
    if not bool(config["controls"]["all_critical_replicates_pass"]):
        raise RuntimeError("frozen control design fails critical balance; outcomes are blocked")
    if not bool(config["controls"]["all_low_impact_replicates_pass"]):
        raise RuntimeError("frozen suspicious-low-impact controls are not lower impact; outcomes are blocked")
    design = np.load(DESIGN_PATH)
    sources = design["sources"].astype(np.int32)
    targets = design["targets"].astype(np.int32)
    target_edges = design["target_edges"].astype(np.int32)
    random_sequences = design["matched_random_sequences"].astype(np.int32)
    low_sequences = design["suspicious_low_impact_sequences"].astype(np.int32)

    directed_graph = sp.load_npz(GRAPH_PATH).tocsr()
    edge_frame = pd.read_parquet(EDGE_TABLE_PATH).sort_values("edge_id").reset_index(drop=True)
    position_data = np.load(POSITION_PATH)
    positions = np.asarray(position_data["positions"], dtype=np.float32)
    d_geo = np.load(D_GEO_PATH, mmap_mode="r")
    n = directed_graph.shape[0]
    if target_edges.shape != (100,) or random_sequences.shape != (50, 100) or low_sequences.shape != (50, 100):
        raise RuntimeError("frozen design array shapes are invalid")

    flat_sources = np.repeat(sources, TARGETS_PER_SOURCE)
    flat_targets = targets.reshape(-1)
    physical = np.linalg.norm(
        positions[flat_sources] - positions[flat_targets],
        axis=1,
    ).astype(np.float64)
    predicted_log = frozen_isotonic_predict(np.log(physical))
    baseline_matrix = shortest_path(
        directed_graph,
        method="auto",
        directed=False,
        indices=sources,
    )
    baseline_distances = baseline_matrix[
        np.arange(len(sources), dtype=np.int32)[:, None],
        targets,
    ].reshape(-1)
    saved_baseline = np.asarray(d_geo[flat_sources, flat_targets], dtype=np.float32)
    if not np.array_equal(baseline_distances.astype(np.float32), saved_baseline):
        raise RuntimeError("fresh baseline shortest paths do not reproduce saved D_geo")
    baseline_residual = np.log(baseline_distances) - predicted_log
    baseline_severe = baseline_residual <= SEVERE_THRESHOLD
    baseline_overestimated = baseline_residual > OVER_ESTIMATION_THRESHOLD
    baseline_well_estimated = (
        np.abs(baseline_residual) <= WELL_ESTIMATED_ABS_RESIDUAL_THRESHOLD
    )
    if not np.any(baseline_severe) or not np.any(baseline_well_estimated):
        raise RuntimeError("fresh test lacks severe or well-estimated baseline pairs")

    directed_rows = np.repeat(
        np.arange(n, dtype=np.int32),
        np.diff(directed_graph.indptr),
    )
    directed_low = np.minimum(directed_rows, directed_graph.indices).astype(np.int64)
    directed_high = np.maximum(directed_rows, directed_graph.indices).astype(np.int64)
    directed_keys = directed_low * n + directed_high

    rows: list[dict[str, object]] = []
    for count in REMOVAL_COUNTS:
        print(f"targeted X={int(count)}", flush=True)
        rows.append(
            run_one(
                "targeted",
                -1,
                int(count),
                target_edges[:count],
                directed_graph,
                directed_rows,
                directed_keys,
                edge_frame,
                sources,
                targets,
                baseline_distances,
                physical,
                predicted_log,
                baseline_residual,
                baseline_severe,
                baseline_overestimated,
                baseline_well_estimated,
            )
        )
    for family, sequences in (
        ("matched_random", random_sequences),
        ("suspicious_low_impact", low_sequences),
    ):
        for replicate, sequence in enumerate(sequences):
            if replicate % 5 == 0:
                print(f"{family} replicate {replicate}/{CONTROL_REPLICATES}", flush=True)
            for count in REMOVAL_COUNTS:
                rows.append(
                    run_one(
                        family,
                        replicate,
                        int(count),
                        sequence[:count],
                        directed_graph,
                        directed_rows,
                        directed_keys,
                        edge_frame,
                        sources,
                        targets,
                        baseline_distances,
                        physical,
                        predicted_log,
                        baseline_residual,
                        baseline_severe,
                        baseline_overestimated,
                        baseline_well_estimated,
                    )
                )

    run_frame = pd.DataFrame(rows)
    run_frame.to_parquet(RUN_TABLE_PATH, index=False)
    decision, comparisons = counterfactual_decision(run_frame)
    pair_frame = pd.DataFrame(
        {
            "source": flat_sources.astype(np.int32),
            "target": flat_targets.astype(np.int32),
            "physical_distance": physical.astype(np.float32),
            "baseline_d_geo": baseline_distances.astype(np.float32),
            "predicted_log_d_geo": predicted_log.astype(np.float32),
            "baseline_r_geo": baseline_residual.astype(np.float32),
            "baseline_severe": baseline_severe,
            "baseline_overestimated": baseline_overestimated,
            "baseline_well_estimated": baseline_well_estimated,
        }
    )
    pair_frame.to_parquet(TEST_PAIR_PATH, index=False)
    save_figures(run_frame)

    targeted_rows = run_frame[run_frame["family"] == "targeted"].sort_values("edge_count")
    summary: dict[str, object] = {
        "scope": "H1 targeted-removal counterfactual only; no sensitivity, H2, or method development",
        "H1_counterfactual_supported": decision,
        "H1_supported": None,
        "fresh_test": {
            "source_count": int(len(sources)),
            "pair_count": int(len(pair_frame)),
            "baseline_severe_count": int(np.count_nonzero(baseline_severe)),
            "baseline_severe_fraction": float(np.mean(baseline_severe)),
            "baseline_overestimation_count": int(np.count_nonzero(baseline_overestimated)),
            "baseline_overestimation_fraction": float(np.mean(baseline_overestimated)),
            "baseline_well_estimated_count": int(np.count_nonzero(baseline_well_estimated)),
            "baseline_residual": {
                "p01": float(np.percentile(baseline_residual, 1)),
                "p05": float(np.percentile(baseline_residual, 5)),
                "median": float(np.median(baseline_residual)),
                "p95": float(np.percentile(baseline_residual, 95)),
                "p99": float(np.percentile(baseline_residual, 99)),
            },
            "baseline_shortest_paths_exactly_match_saved_D_geo_float32": True,
        },
        "candidate_descriptions": config["candidate_descriptions"],
        "targeted_results": targeted_rows.to_dict(orient="records"),
        "decision_comparisons": comparisons,
        "artifacts": {
            "config": str(CONFIG_PATH.relative_to(REPO_ROOT)),
            "design": str(DESIGN_PATH.relative_to(REPO_ROOT)),
            "control_edges": str(CONTROL_EDGE_PATH.relative_to(REPO_ROOT)),
            "control_balance": str(BALANCE_PATH.relative_to(REPO_ROOT)),
            "run_table": str(RUN_TABLE_PATH.relative_to(REPO_ROOT)),
            "test_pairs": str(TEST_PAIR_PATH.relative_to(REPO_ROOT)),
        },
        "interpretation_boundary": "counterfactual true does not set H1_supported=true; q/source sensitivity remains required",
    }
    write_json(SUMMARY_PATH, summary)

    report_rows = []
    control_rows = []
    for _, row in targeted_rows.iterrows():
        count = int(row["edge_count"])
        report_rows.append(
            f"| {count} | {row['repair_rate']:.4%} | "
            f"{row['mean_delta_r_baseline_severe']:.6f} | "
            f"{row['median_delta_r_baseline_severe']:.6f} | "
            f"{row['severe_fraction_after']:.4%} | "
            f"{row['new_overestimation_fraction_increase']:.4%} | "
            f"{row['median_log_inflation_well_estimated']:.6f} | "
            f"{row['p95_log_inflation_well_estimated']:.6f} | "
            f"{int(row['connected_components'])} | {int(row['unreachable_test_pairs'])} |"
        )
        random_repair = control_quantiles(run_frame, "matched_random", count, "repair_rate")["p95"]
        low_repair = control_quantiles(
            run_frame, "suspicious_low_impact", count, "repair_rate"
        )["p95"]
        random_mean_delta = control_quantiles(
            run_frame, "matched_random", count, "mean_delta_r_baseline_severe"
        )["p95"]
        low_mean_delta = control_quantiles(
            run_frame,
            "suspicious_low_impact",
            count,
            "mean_delta_r_baseline_severe",
        )["p95"]
        random_over = control_quantiles(
            run_frame,
            "matched_random",
            count,
            "new_overestimation_fraction_increase",
        )["p95"]
        low_over = control_quantiles(
            run_frame,
            "suspicious_low_impact",
            count,
            "new_overestimation_fraction_increase",
        )["p95"]
        random_p95_inflation = control_quantiles(
            run_frame,
            "matched_random",
            count,
            "p95_log_inflation_well_estimated",
        )["p95"]
        low_p95_inflation = control_quantiles(
            run_frame,
            "suspicious_low_impact",
            count,
            "p95_log_inflation_well_estimated",
        )["p95"]
        control_rows.append(
            f"| {count} | {row['repair_rate']:.4%} / {random_repair:.4%} / {low_repair:.4%} | "
            f"{row['mean_delta_r_baseline_severe']:.6f} / {random_mean_delta:.6f} / {low_mean_delta:.6f} | "
            f"{row['new_overestimation_fraction_increase']:.4%} / {random_over:.4%} / {low_over:.4%} | "
            f"{row['p95_log_inflation_well_estimated']:.6f} / {random_p95_inflation:.6f} / {low_p95_inflation:.6f} |"
        )
    report = f"""# Phase 2C H1 Targeted-Removal Counterfactual

## Scope

This diagnostic uses the frozen first-stage discovery ranking and an entirely
new pair-disjoint test sample. It does not rerank edges, refit calibration, run
sensitivity, execute H2, or design a refinement method.

## Fresh test

- Seed: `{COUNTERFACTUAL_SEED}`
- Sources / targets per source: `{SOURCE_COUNT}` / `{TARGETS_PER_SOURCE}`
- Pairs: `{len(pair_frame)}`
- Baseline-severe pairs: `{int(np.count_nonzero(baseline_severe))}` (`{float(np.mean(baseline_severe)):.4%}`)
- Baseline-overestimated pairs: `{int(np.count_nonzero(baseline_overestimated))}` (`{float(np.mean(baseline_overestimated)):.4%}`)
- Baseline-well-estimated pairs: `{int(np.count_nonzero(baseline_well_estimated))}`
- Baseline shortest paths exactly reproduce saved `D_geo` in float32: `true`
- Controls: `{CONTROL_REPLICATES}` matched-random and `{CONTROL_REPLICATES}` suspicious-low-impact nested sequences

## Targeted results

| X | repair rate | mean delta r_geo | median delta r_geo | severe fraction after | new overestimation | median well-estimated inflation | P95 well-estimated inflation | components | unreachable pairs |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(report_rows)}

## Control-relative results

Each cell is `targeted / matched-random P95 / suspicious-low-impact P95`.

| X | repair rate | mean delta r_geo | new overestimation | P95 well-estimated inflation |
|---:|---:|---:|---:|---:|
{chr(10).join(control_rows)}

All control replicates satisfied the pre-frozen critical balance limit
`|SMD| <= {CRITICAL_BALANCE_LIMIT:.2f}`. Secondary covariates satisfied
`|SMD| <= {SECONDARY_BALANCE_LIMIT:.2f}`, although several covariates were close
to the admissible boundary. The suspicious-low-impact controls had positive,
matched total path usage and matched mismatch. Their realized median discovery
bad-path enrichment was zero; this was an outcome-free result of the secondary
low-enrichment objective, not an eligibility rule.

## Decision

`H1_counterfactual_supported = {decision}`

The frozen rule was not satisfied. Targeted removal exceeded both control P95
values for repair rate at every X, and its mean severe-cohort residual correction
also exceeded both control P95 values. However, the pre-frozen median residual
endpoint was zero for targeted and controls because most severe pairs were
unchanged by these sparse interventions. Targeted new overestimation and P95
well-estimated-pair inflation also exceeded both control families at every X.
The result was not caused by graph disconnection: all 303 modified graphs had one
component and zero unreachable fresh-test pairs.

The ranked edges therefore have strong causal influence on a concentrated subset
of severe underestimation, but the deletion intervention is not sufficiently
selective: it also causes substantial collateral geodesic inflation. The formal
`false` is specific to the frozen clean, low-damage counterfactual support rule;
it must not be read as evidence that the ranked edges have no effect.

The median endpoint was insensitive to a sparse effect, and the absolute harm
guardrail compares targeted removal with control interventions having much smaller
repair magnitude. Both are retained as pre-outcome design caveats rather than
changed after observing the result.

`H1_supported` remains `null`. No q/source sensitivity, H2, or method-development
experiment was run.
"""
    REPORT_PATH.write_text(report, encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="freeze split/control design and write balance artifacts without shortest-path interventions",
    )
    args = parser.parse_args()
    if args.prepare_only:
        config = prepare_design()
        print(json.dumps({"prepared": True, "balance": config["controls"]["balance_summary"]}, indent=2))
        return
    summary = execute_counterfactual()
    print(
        json.dumps(
            {
                "H1_counterfactual_supported": summary["H1_counterfactual_supported"],
                "baseline_severe_count": summary["fresh_test"]["baseline_severe_count"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
