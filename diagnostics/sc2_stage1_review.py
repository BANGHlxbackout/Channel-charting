"""Post-result audit of the frozen Stage-1 common-neighbor pilot.

Ground-truth positions are used only here, never in the saved edge score or
intervention. This script does not rerun shortest paths, MDS, FCF, or Stage 2.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT / "results/sc2_stage1_consensus_pilot"
DISTANCE_RUN = ROOT / "results/adp_old_fused/stride1_n20997_seed10"


def aligned_errors(reference: np.ndarray, chart: np.ndarray) -> np.ndarray:
    design = np.column_stack([chart, np.ones(len(chart), dtype=chart.dtype)])
    transform, *_ = np.linalg.lstsq(design, reference, rcond=None)
    return np.linalg.norm(design @ transform - reference, axis=1)


def summarize_edge_group(indices: np.ndarray, physical: np.ndarray, adp_branch: np.ndarray, gaps: np.ndarray) -> dict:
    d = physical[indices]
    return {
        "count_directed": int(len(indices)),
        "physical_distance_median_m": float(np.median(d)),
        "physical_distance_lt_0p25m_fraction": float(np.mean(d < 0.25)),
        "physical_distance_gt_0p5m_fraction": float(np.mean(d > 0.5)),
        "adp_branch_fraction": float(np.mean(adp_branch[indices])),
        "time_gap_median_s": float(np.median(gaps[indices])),
    }


def distinct_examples(indices: np.ndarray, order: np.ndarray, rows: np.ndarray, cols: np.ndarray,
                      score: np.ndarray, original: np.ndarray, physical: np.ndarray,
                      gaps: np.ndarray, count: int = 3) -> list[dict]:
    examples = []
    seen = set()
    for idx in indices[order]:
        a, b = sorted((int(rows[idx]), int(cols[idx])))
        if (a, b) in seen:
            continue
        seen.add((a, b))
        examples.append({
            "sample_pair": [a, b],
            "common_support_score": float(score[idx]),
            "length_multiplier": float(1 + 0.5 * (1 - score[idx])),
            "original_fused_edge_length": float(original[idx]),
            "physical_distance_m": float(physical[idx]),
            "time_gap_s": float(gaps[idx]),
        })
        if len(examples) == count:
            break
    return examples


def main() -> None:
    saved = np.load(PILOT / "edge_scores.npz")
    rows = saved["rows"].astype(np.int64)
    cols = saved["cols"].astype(np.int64)
    original = saved["original_weights"].astype(np.float64)
    score = saved["support"].astype(np.float64)
    multipliers = saved["treatment_multiplier"].astype(np.float64)
    n = 20_997
    if not (len(rows) == len(cols) == len(original) == len(score) == len(multipliers) == n * 20):
        raise ValueError("frozen Stage-1 edge arrays are not the verified k=20 graph")
    if not np.allclose(multipliers, 1 + 0.5 * (1 - score), atol=1e-7):
        raise ValueError("saved multipliers differ from the frozen rule")

    data = np.load(DISTANCE_RUN / "distance_data.npz")
    positions = data["positions"]
    timestamps = np.load(ROOT / "datasets/adp_old_shard4/timestamps.npy")
    adp = np.load(DISTANCE_RUN / "adp.npy", mmap_mode="r")
    gamma = float(json.loads((DISTANCE_RUN / "result.json").read_text(encoding="utf-8"))["gamma"])
    physical = np.linalg.norm(positions[rows] - positions[cols], axis=1)
    gaps = np.abs(timestamps[rows] - timestamps[cols])
    adp_branch = np.asarray(adp[rows, cols]) <= gamma * gaps

    edge_bins = np.quantile(original, np.linspace(0, 1, 11))
    deciles = []
    for decile in range(10):
        in_bin = (original >= edge_bins[decile]) & (
            original <= edge_bins[decile + 1] if decile == 9 else original < edge_bins[decile + 1]
        )
        q1, q3 = np.quantile(score[in_bin], [0.25, 0.75])
        low = np.flatnonzero(in_bin & (score <= q1))
        high = np.flatnonzero(in_bin & (score >= q3))
        deciles.append({
            "fused_length_decile": decile + 1,
            "fused_length_min": float(edge_bins[decile]),
            "fused_length_max": float(edge_bins[decile + 1]),
            "low_support": summarize_edge_group(low, physical, adp_branch, gaps),
            "high_support": summarize_edge_group(high, physical, adp_branch, gaps),
        })
    top = deciles[-1]
    top_mask = (original >= edge_bins[-2]) & (original <= edge_bins[-1])
    q1, q3 = np.quantile(score[top_mask], [0.25, 0.75])
    low = np.flatnonzero(top_mask & (score <= q1))
    high = np.flatnonzero(top_mask & (score >= q3))

    mds = {name: np.load(PILOT / name / "mds.npz") for name in ("baseline", "shuffled", "consensus")}
    ids = mds["baseline"]["indices"].astype(np.int64)
    reference = mds["baseline"]["positions"]
    errors = {}
    for name, item in mds.items():
        if not np.array_equal(ids, item["indices"]) or not np.array_equal(reference, item["positions"]):
            raise ValueError("MDS sample order differs between variants")
        errors[name] = aligned_errors(reference, item["embedding"])
    sample_rows = []
    for index, sample_id in enumerate(ids):
        sample_rows.append({
            "sample_id": int(sample_id),
            "physical_x_m": float(reference[index, 0]),
            "physical_y_m": float(reference[index, 1]),
            "original_MDS_error_m": float(errors["baseline"][index]),
            "shuffled_MDS_error_m": float(errors["shuffled"][index]),
            "support_MDS_error_m": float(errors["consensus"][index]),
            "support_change_m_positive_is_improvement": float(errors["baseline"][index] - errors["consensus"][index]),
        })
    with (PILOT / "POSTHOC_SAMPLE_ERRORS.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(sample_rows[0]))
        writer.writeheader()
        writer.writerows(sample_rows)

    delta = errors["baseline"] - errors["consensus"]
    shuffled_delta = errors["baseline"] - errors["shuffled"]
    examples = {
        "low_support_but_physically_close": distinct_examples(low, np.argsort(physical[low]), rows, cols, score, original, physical, gaps),
        "high_support_but_physically_far": distinct_examples(high, np.argsort(physical[high])[::-1], rows, cols, score, original, physical, gaps),
        "largest_sample_harms": [sample_rows[i] for i in np.argsort(delta)[:5]],
        "largest_sample_repairs": [sample_rows[i] for i in np.argsort(delta)[-5:][::-1]],
    }
    review = {
        "status": "post-result GT-assisted diagnosis; no frozen outcome or method rule changed",
        "edge_count_directed": int(len(rows)),
        "score_definition": "common-neighbor score on binary undirected union of original directed k=20 graph",
        "intervention_definition": "original Stage-1 fused edge length * [1 + 0.5 * (1 - score)]",
        "edge_set_changed": False,
        "shuffled_control": "same multiplier distribution permuted within original edge-length deciles and endpoint-degree bins",
        "edge_score_quantiles": [float(x) for x in np.quantile(score, [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1])],
        "length_multiplier_quantiles": [float(x) for x in np.quantile(multipliers, [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1])],
        "deciles": deciles,
        "top_fused_length_decile": top,
        "top_decile_support_quartile_cutoffs": {"low_max": float(q1), "high_min": float(q3)},
        "mds_sample_count": int(len(ids)),
        "mds_sample_outcomes": {
            name: {
                "mean_error_m": float(np.mean(value)),
                "p95_error_m": float(np.percentile(value, 95)),
                "improved_count_vs_original": int(np.sum(errors["baseline"] - value > 0)),
                "harmed_count_vs_original": int(np.sum(errors["baseline"] - value < 0)),
            }
            for name, value in errors.items()
        },
        "support_mds_change_median_m": float(np.median(delta)),
        "support_better_than_shuffle_sample_count": int(np.sum(errors["consensus"] < errors["shuffled"])),
        "support_worse_than_shuffle_sample_count": int(np.sum(errors["consensus"] > errors["shuffled"])),
        "largest_harms_and_repairs_in_same_area_are_not_edge_causal_attribution": True,
        "examples": examples,
    }
    (PILOT / "POSTHOC_REVIEW.json").write_text(json.dumps(review, indent=2, ensure_ascii=False), encoding="utf-8")
    print("wrote posthoc review: 10 distance bins and", len(sample_rows), "MDS samples")


if __name__ == "__main__":
    main()
