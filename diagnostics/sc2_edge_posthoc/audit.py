"""Read-only, GT-assisted post-hoc audit of saved Stage-2 edge masks.

This script does not select edges, alter graph weights, solve refinement, or tune
parameters. Ground-truth positions are read only for offline assessment.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from common_metrics import affine_transform  # noqa: E402

SOURCE = Path(r"D:\diffusion_refined_charting")
DIST = SOURCE / "results/adp_old_fused/stride1_n20997_seed10"
Y0 = SOURCE / "results/adp_old_siamese/stride1_n20997_seed10/embedding.npz"
OUT = ROOT / "results/sc2_stage2_edge_consensus"


def as_json(value):
    if isinstance(value, dict):
        return {key: as_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [as_json(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    return value


def subset_stats(values: np.ndarray) -> dict:
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def outcome_stats(delta: np.ndarray) -> dict:
    repair, harm = delta > 0, delta < 0
    return {
        "repair_count": int(repair.sum()),
        "harm_count": int(harm.sum()),
        "repair_fraction": float(repair.mean()),
        "mean_repair_m_among_repaired": float(delta[repair].mean()),
        "mean_harm_m_among_harmed": float((-delta[harm]).mean()),
        "sum_repair_m": float(delta[repair].sum()),
        "sum_harm_m": float((-delta[harm]).sum()),
        "net_mean_improvement_m": float(delta.mean()),
        "delta_p05_p50_p95_m": [float(x) for x in np.percentile(delta, [5, 50, 95])],
    }


def sample_cases(
    delta: np.ndarray,
    baseline: np.ndarray,
    variant: np.ndarray,
    ids: np.ndarray,
    positions: np.ndarray,
    incident_count: np.ndarray,
    lost_weight: np.ndarray,
) -> dict:
    result = {}
    for label, selected in (
        ("top_repairs", np.argsort(delta)[-5:][::-1]),
        ("top_harms", np.argsort(delta)[:5]),
    ):
        result[label] = [
            {
                "sample_row": int(i),
                "original_sample_id": int(ids[i]),
                "physical_xy_m": [float(x) for x in positions[i]],
                "baseline_error_m": float(baseline[i]),
                "variant_error_m": float(variant[i]),
                "baseline_minus_variant_m": float(delta[i]),
                "selected_incident_edge_count": int(incident_count[i]),
                "incident_weight_loss": float(lost_weight[i]),
            }
            for i in selected
        ]
    return result


def main() -> None:
    saved = np.load(OUT / "edge_scores_and_masks.npz")
    emb = np.load(OUT / "embeddings.npz")
    y0_saved = np.load(Y0)
    ids = emb["indices"]
    pos = emb["positions"].astype(np.float32)
    if not np.array_equal(ids, y0_saved["indices"]):
        raise ValueError("Saved chart and Y0 sample order differ")
    if not np.array_equal(pos, y0_saved["positions"]):
        raise ValueError("Saved chart and Y0 positions differ")
    row, col = saved["row"].astype(np.int32), saved["col"].astype(np.int32)
    weight = saved["w"].astype(np.float64)
    score = saved["score"].astype(np.int32)
    eligible = saved["eligible"].astype(bool)
    residual = saved["first_order_residual"].astype(np.float64)
    sc2_ix = saved["sc2_selected"].astype(np.int64)
    random_ix = saved["shuffled_selected"].astype(np.int64)
    n_edges = len(row)
    if len({len(row), len(col), len(weight), len(score), len(eligible), len(residual)}) != 1:
        raise ValueError("Saved edge arrays have different lengths")
    if len(np.unique(sc2_ix)) != len(sc2_ix) or len(np.unique(random_ix)) != len(random_ix):
        raise ValueError("Duplicate selected edge index")
    if len(sc2_ix) != len(random_ix) or not eligible[sc2_ix].all() or not eligible[random_ix].all():
        raise ValueError("Selected mask does not match original eligible pool")
    geo = np.load(DIST / "geodesic.npy", mmap_mode="r")
    dgeo = np.asarray(geo[row, col], dtype=np.float64)
    y0 = y0_saved["embedding"].astype(np.float64)
    dy = np.linalg.norm(y0[row] - y0[col], axis=1)
    physical = np.linalg.norm(pos[row].astype(np.float64) - pos[col].astype(np.float64), axis=1)
    geo_cuts = np.quantile(dgeo, np.linspace(0, 1, 11)[1:-1])
    geo_bins = np.searchsorted(geo_cuts, dgeo, side="right")
    suspect = np.zeros(n_edges, dtype=bool)
    for bin_id in range(10):
        within = np.flatnonzero(eligible & (geo_bins == bin_id))
        threshold = np.quantile(physical[within], 0.9)
        suspect[within] = physical[within] >= threshold
    sc2_mask = np.zeros(n_edges, dtype=bool)
    random_mask = np.zeros(n_edges, dtype=bool)
    sc2_mask[sc2_ix] = True
    random_mask[random_ix] = True
    if not np.array_equal(np.bincount(geo_bins[sc2_ix], minlength=10),
                          np.bincount(geo_bins[random_ix], minlength=10)):
        raise ValueError("SC2 and random masks are not D_geo-decile matched")

    def mask_stats(ix: np.ndarray) -> dict:
        return {
            "count": int(len(ix)),
            "score": subset_stats(score[ix]),
            "first_order_residual": subset_stats(residual[ix]),
            "d_geo": subset_stats(dgeo[ix]),
            "d_y0": subset_stats(dy[ix]),
            "original_w": subset_stats(weight[ix]),
            "physical_endpoint_distance_m_GT_only": subset_stats(physical[ix]),
            "GT_suspect_fraction": float(suspect[ix].mean()),
            "total_original_weight": float(weight[ix].sum()),
            "lost_weight_at_multiplier_0_5": float(0.5 * weight[ix].sum()),
            "D_geo_decile_counts": np.bincount(geo_bins[ix], minlength=10).tolist(),
        }

    base = np.linalg.norm(affine_transform(pos, emb["uniform"]) - pos, axis=1)
    sc2 = np.linalg.norm(affine_transform(pos, emb["sc2"]) - pos, axis=1)
    random = np.linalg.norm(affine_transform(pos, emb["shuffled"]) - pos, axis=1)
    delta_sc2, delta_random = base - sc2, base - random

    def incident(ix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        endpoints = np.concatenate((row[ix], col[ix]))
        counts = np.bincount(endpoints, minlength=len(pos))
        losses = np.bincount(
            endpoints,
            weights=np.concatenate((0.5 * weight[ix], 0.5 * weight[ix])),
            minlength=len(pos),
        )
        return counts, losses

    sc2_incident, sc2_loss = incident(sc2_ix)
    random_incident, random_loss = incident(random_ix)
    order = np.argsort(base, kind="mergesort")
    error_quintiles = []
    for rank, members in enumerate(np.array_split(order, 5), 1):
        error_quintiles.append({
            "baseline_error_quintile": rank,
            "n": int(len(members)),
            "baseline_mean_error_m": float(base[members].mean()),
            "sc2_mean_improvement_m": float(delta_sc2[members].mean()),
            "random_mean_improvement_m": float(delta_random[members].mean()),
            "sc2_harm_fraction": float(np.mean(delta_sc2[members] < 0)),
            "random_harm_fraction": float(np.mean(delta_random[members] < 0)),
        })

    xcuts = np.quantile(pos[:, 0], [0.25, 0.5, 0.75])
    ycuts = np.quantile(pos[:, 1], [0.25, 0.5, 0.75])
    bx = np.searchsorted(xcuts, pos[:, 0], side="right")
    by = np.searchsorted(ycuts, pos[:, 1], side="right")
    spatial = []
    for ix in range(4):
        for iy in range(4):
            members = np.flatnonzero((bx == ix) & (by == iy))
            if len(members) == 0:
                continue
            spatial.append({
                "x_bin": ix, "y_bin": iy, "n": int(len(members)),
                "center_xy_m": [float(x) for x in pos[members].mean(axis=0)],
                "sc2_mean_improvement_m": float(delta_sc2[members].mean()),
                "random_mean_improvement_m": float(delta_random[members].mean()),
                "sc2_harm_fraction": float(np.mean(delta_sc2[members] < 0)),
                "random_harm_fraction": float(np.mean(delta_random[members] < 0)),
            })

    def edge_cases(ix: np.ndarray, delta: np.ndarray) -> dict:
        chosen_suspect = ix[suspect[ix]]
        chosen_other = ix[~suspect[ix]]
        # Deliberate GT-assisted case selection for illustration only.
        long_cases = chosen_suspect[np.argsort(physical[chosen_suspect])[-3:][::-1]]
        near_median = np.argsort(np.abs(physical[chosen_other] - np.median(physical[chosen_other])))[:3]
        ordinary_cases = chosen_other[near_median]
        def describe(edge: int) -> dict:
            i, j = int(row[edge]), int(col[edge])
            return {
                "edge_array_index": int(edge),
                "sample_rows": [i, j],
                "original_sample_ids": [int(ids[i]), int(ids[j])],
                "GT_endpoint_positions_m": [pos[i].tolist(), pos[j].tolist()],
                "score_common_witnesses": int(score[edge]),
                "first_order_residual": float(residual[edge]),
                "d_geo": float(dgeo[edge]),
                "d_y0": float(dy[edge]),
                "GT_physical_distance_m": float(physical[edge]),
                "original_W": float(weight[edge]),
                "GT_suspect_label": bool(suspect[edge]),
                "endpoint_improvement_m_after_full_mask": [float(delta[i]), float(delta[j])],
            }
        return {
            "GT_postselected_long_examples": [describe(int(edge)) for edge in long_cases],
            "GT_postselected_ordinary_examples": [describe(int(edge)) for edge in ordinary_cases],
        }

    original_degree = np.bincount(np.concatenate((row, col)), minlength=len(pos))

    def concentration(counts: np.ndarray, delta: np.ndarray) -> dict:
        k = max(1, len(counts) // 10)
        return {
            "nodes_touched": int(np.count_nonzero(counts)),
            "selected_incident_count_p50_p90_p95_max": [float(x) for x in np.percentile(counts, [50, 90, 95, 100])],
            "top_10pct_nodes_share_of_selected_edge_endpoints": float(np.sort(counts)[-k:].sum() / counts.sum()),
            "selected_fraction_of_original_degree_p50_p95_max": [float(x) for x in np.percentile(counts / original_degree, [50, 95, 100])],
            "corr_selected_count_vs_abs_point_error_change": float(np.corrcoef(counts, np.abs(delta))[0, 1]),
        }

    output = {
        "audit_only": True,
        "GT_role": "offline evaluation and post-selection of illustrative cases; never used to form existing edge masks",
        "sample_count": int(len(pos)),
        "candidate_undirected_edges": int(n_edges),
        "eligible_edges": int(eligible.sum()),
        "original_sc2_score_formula": "B_ij * sum_k B_ik B_kj on symmetric sparse B",
        "first_order_log_residual_formula": "abs(log(dY_ij)-log(Dgeo_ij)-median_directed(log(dY)-log(Dgeo)))",
        "first_order_threshold": float(json.loads((OUT / "result.json").read_text())["first_order_threshold"]),
        "edge_mask_overlap_count": int(np.intersect1d(sc2_ix, random_ix).size),
        "sc2_selected": mask_stats(sc2_ix),
        "seed10_random_selected": mask_stats(random_ix),
        "outcomes": {
            "sc2": outcome_stats(delta_sc2),
            "seed10_random": outcome_stats(delta_random),
            "baseline_error": subset_stats(base),
            "sc2_error": subset_stats(sc2),
            "seed10_random_error": subset_stats(random),
        },
        "selection_concentration": {
            "sc2": concentration(sc2_incident, delta_sc2),
            "seed10_random": concentration(random_incident, delta_random),
        },
        "by_baseline_error_quintile": error_quintiles,
        "by_physical_position_4x4": spatial,
        "sample_examples": {
            "sc2": sample_cases(delta_sc2, base, sc2, ids, pos, sc2_incident, sc2_loss),
            "seed10_random": sample_cases(delta_random, base, random, ids, pos, random_incident, random_loss),
        },
        "selected_edge_examples": {
            "sc2": edge_cases(sc2_ix, delta_sc2),
            "seed10_random": edge_cases(random_ix, delta_random),
        },
    }
    (OUT / "posthoc_edge_and_sample_audit.json").write_text(
        json.dumps(as_json(output), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    np.savez_compressed(
        OUT / "posthoc_sample_errors.npz",
        indices=ids, positions=pos,
        baseline_error=base, sc2_error=sc2, seed10_random_error=random,
        sc2_improvement=delta_sc2, seed10_random_improvement=delta_random,
        sc2_incident_edges=sc2_incident, seed10_random_incident_edges=random_incident,
        sc2_incident_weight_loss=sc2_loss, seed10_random_incident_weight_loss=random_loss,
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    max_color = float(np.percentile(np.abs(np.concatenate((delta_sc2, delta_random))), 98))
    max_color = max(max_color, 1e-6)
    for ax, delta, title in zip(
        axes, (delta_sc2, delta_random),
        ("SC2-selected edges", "Distance-stratified random edges, seed 10")
    ):
        points = ax.scatter(pos[:, 0], pos[:, 1], c=delta, s=5,
                            cmap="RdBu", vmin=-max_color, vmax=max_color,
                            linewidths=0, rasterized=True)
        ax.set_title(title)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("physical x (m)")
        ax.set_ylabel("physical y (m)")
        ax.grid(alpha=0.2)
    fig.colorbar(points, ax=axes, shrink=0.82,
                 label="baseline error - variant error (m); blue=repair, red=harm")
    fig.savefig(OUT / "posthoc_repair_harm_map.png", dpi=190)
    plt.close(fig)

    counts = np.bincount(geo_bins[sc2_ix], minlength=10)
    sc2_bad = [float(suspect[sc2_ix[geo_bins[sc2_ix] == b]].mean()) for b in range(10)]
    random_bad = [float(suspect[random_ix[geo_bins[random_ix] == b]].mean()) for b in range(10)]
    x = np.arange(10)
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    ax.bar(x - .2, sc2_bad, .4, label="SC2 selected")
    ax.bar(x + .2, random_bad, .4, label="random selected")
    ax.set_xlabel("D_geo distance decile (low to high)")
    ax.set_ylabel("GT-only long-physical-edge fraction")
    ax.set_xticks(x, [str(i + 1) for i in x])
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    fig.savefig(OUT / "posthoc_selected_edge_diagnostic.png", dpi=190)
    plt.close(fig)

    print("SC2 suspect fraction", output["sc2_selected"]["GT_suspect_fraction"])
    print("Random suspect fraction", output["seed10_random_selected"]["GT_suspect_fraction"])
    print("SC2 outcomes", output["outcomes"]["sc2"])
    print("Random outcomes", output["outcomes"]["seed10_random"])


if __name__ == "__main__":
    main()
