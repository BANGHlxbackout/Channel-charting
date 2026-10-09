"""Diagnostic: match SC2 edge counts by D_geo decile and fixed Y0 chart tile.

This is an exploratory control of broad selection location, not a new method.
GT positions enter metrics only, after every mask has been fixed.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from common_metrics import chart_metrics, localization_error_summary  # noqa: E402
from methods.sc2_stage2_edge_consensus.run import graph_from_upper  # noqa: E402
from run_diffusion_refined_charting import solve_graph_harmonic_refinement  # noqa: E402

SOURCE = Path(r"D:\diffusion_refined_charting")
EDGE_RUN = ROOT / "results/sc2_stage2_edge_consensus"
OUT = ROOT / "results/sc2_edge_cause"
FIELDS = ("ct", "tw", "ks", "mean_m", "median_m", "p90_m", "p95_m", "max_m")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    saved = np.load(EDGE_RUN / "edge_scores_and_masks.npz")
    row, col, weight = saved["row"], saved["col"], saved["w"]
    eligible = saved["eligible"].astype(bool)
    selected = saved["sc2_selected"]
    chart = np.load(SOURCE / "results/adp_old_siamese/stride1_n20997_seed10/embedding.npz")
    y0 = chart["embedding"].astype(np.float32)
    pos = chart["positions"].astype(np.float32)
    fixed = np.load(EDGE_RUN / "embeddings.npz")
    if not np.array_equal(chart["indices"], fixed["indices"]):
        raise ValueError("Saved sample order differs")
    n = len(y0)
    if n != 20997 or len(selected) != 110862 or not eligible[selected].all():
        raise ValueError("Unexpected frozen experiment inputs")

    geo = np.load(SOURCE / "results/adp_old_fused/stride1_n20997_seed10/geodesic.npy", mmap_mode="r")
    d = np.asarray(geo[row, col], dtype=np.float64)
    distance_bin = np.searchsorted(np.quantile(d, np.linspace(0, 1, 11)[1:-1]), d, side="right")
    midpoint = (y0[row].astype(np.float64) + y0[col].astype(np.float64)) / 2
    x_bin = np.searchsorted(np.quantile(y0[:, 0], [0.25, 0.5, 0.75]), midpoint[:, 0])
    y_bin = np.searchsorted(np.quantile(y0[:, 1], [0.25, 0.5, 0.75]), midpoint[:, 1])
    stratum = distance_bin * 16 + x_bin * 4 + y_bin
    pool = np.flatnonzero(eligible)
    counts = np.bincount(stratum[selected], minlength=160)
    strata_pools = [pool[stratum[pool] == k] for k in range(160)]
    if any(len(strata_pools[k]) < counts[k] for k in range(160)):
        raise ValueError("SC2 stratum count exceeds eligible pool")

    upper = sp.coo_matrix((weight, (row, col)), shape=(n, n)).tocsr()
    random.seed(10)
    subset = np.asarray(random.sample(range(n), int(0.2 * n)), dtype=np.int64)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    original = json.loads((EDGE_RUN / "result.json").read_text(encoding="utf-8"))["metrics"]
    rows = []
    for seed in range(10, 30):
        rng = np.random.default_rng(seed)
        picked = np.concatenate([
            rng.choice(strata_pools[k], counts[k], replace=False)
            for k in range(160) if counts[k]
        ])
        if len(picked) != len(selected) or not np.array_equal(
            np.bincount(stratum[picked], minlength=160), counts
        ):
            raise ValueError("Matched mask count mismatch")
        factor = np.ones(len(weight), dtype=np.float32)
        factor[picked] = 0.5
        z = solve_graph_harmonic_refinement(
            y0, graph_from_upper(upper, factor), 0.15, 200, device
        )
        incident = np.bincount(np.concatenate((row[picked], col[picked])), minlength=n)
        rows.append({
            "seed": seed,
            **chart_metrics(pos[subset], z[subset]),
            **localization_error_summary(pos, z),
            "mask_overlap_with_sc2": int(np.intersect1d(picked, selected).size),
            "removed_undirected_weight": float(0.5 * weight[picked].astype(np.float64).sum()),
            "top_10pct_nodes_endpoint_share": float(
                np.sort(incident)[-(n // 10):].sum() / incident.sum()
            ),
            "incident_count_p95": float(np.percentile(incident, 95)),
        })
        print(f"region-matched seed {seed}: mean={rows[-1]['mean_m']:.6f} p95={rows[-1]['p95_m']:.6f}", flush=True)

    output = {
        "question": "Does broad Y0-region concentration explain SC2 versus distance-stratified random?",
        "status": "exploratory diagnostic, not a method ranking",
        "mask_rule": "B=1; exactly match SC2 counts in 10 D_geo bins x 4x4 Y0 midpoint-quantile tiles; random seeds 10..29",
        "gt_role": "metrics only; no GT in mask construction",
        "fixed": {"n": n, "selected_edges": len(selected), "lambda": 0.15, "iterations": 200},
        "baseline": original["uniform"],
        "sc2": original["sc2"],
        "runs": rows,
        "summary": {field: {
            "median": float(np.median([r[field] for r in rows])),
            "min": float(np.min([r[field] for r in rows])),
            "max": float(np.max([r[field] for r in rows])),
        } for field in FIELDS},
    }
    (OUT / "region_matched_random.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
