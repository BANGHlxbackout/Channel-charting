"""Existing-best-lambda cross-check for fixed eligible-edge random masks."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from common_metrics import localization_error_summary  # noqa: E402
from run_diffusion_refined_charting import solve_graph_harmonic_refinement  # noqa: E402
from methods.sc2_stage2_edge_consensus.run import graph_from_upper  # noqa: E402

SOURCE = Path(r"D:\diffusion_refined_charting")
OUT = ROOT / "results/sc2_stage2_edge_consensus"
DIST = SOURCE / "results/adp_old_fused/stride1_n20997_seed10/geodesic.npy"
Y0 = SOURCE / "results/adp_old_siamese/stride1_n20997_seed10/embedding.npz"
BEST = SOURCE / "results/lambda_sensitivity_refined_charting/stride1_n20997_k128_lambda0.1344_seed10"


def main() -> None:
    z = np.load(OUT / "edge_scores_and_masks.npz")
    row, col, w = z["row"], z["col"], z["w"]
    eligible = z["eligible"].astype(bool)
    selected = z["sc2_selected"]
    upper = sp.coo_matrix((w, (row, col)), shape=(20997, 20997)).tocsr()
    y = np.load(Y0)
    init = y["embedding"].astype(np.float32)
    pos = y["positions"].astype(np.float32)
    saved = np.load(BEST / "embedding.npz")
    if not np.array_equal(y["indices"], saved["indices"]):
        raise ValueError("Best lambda input order mismatch")
    baseline = localization_error_summary(pos, saved["embedding"])
    geo = np.load(DIST, mmap_mode="r")
    d = np.asarray(geo[row, col], dtype=np.float64)
    cuts = np.quantile(d, np.linspace(0, 1, 11)[1:-1])
    bins = np.searchsorted(cuts, d, side="right")
    pool = np.flatnonzero(eligible)
    counts = [int(np.sum(bins[selected] == b)) for b in range(10)]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    results = []
    for seed in range(10, 30):
        rng = np.random.default_rng(seed)
        picked = np.concatenate([
            rng.choice(pool[bins[pool] == b], counts[b], replace=False)
            for b in range(10)
        ])
        if seed == 10 and not np.array_equal(picked, z["shuffled_selected"]):
            raise ValueError("Seed-10 mask mismatch")
        factor = np.ones(len(w), dtype=np.float32)
        factor[picked] = 0.5
        graph = graph_from_upper(upper, factor)
        print(f"lambda=.1344 seed={seed}", flush=True)
        embedding = solve_graph_harmonic_refinement(init, graph, 0.1344, 200, device)
        m = localization_error_summary(pos, embedding)
        m["seed"] = seed
        results.append(m)
    means = np.asarray([m["mean_m"] for m in results])
    p95 = np.asarray([m["p95_m"] for m in results])
    summary = {
        "lambda": 0.1344,
        "baseline_original_W": baseline,
        "seed_range_inclusive": [10, 29],
        "random_B1_metrics": results,
        "mean_median_m": float(np.median(means)),
        "mean_min_m": float(means.min()),
        "mean_max_m": float(means.max()),
        "mean_std_m": float(means.std(ddof=1)),
        "p95_median_m": float(np.median(p95)),
        "seeds_beating_original_W": int(np.sum(means < baseline["mean_m"])),
        "seeds_beating_original_lambda_015": int(np.sum(means < 0.31874796748161316)),
    }
    (OUT / "best_lambda_followup.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("SUMMARY", json.dumps({k: v for k, v in summary.items() if k != "random_B1_metrics"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
