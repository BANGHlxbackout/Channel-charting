"""Pre-fixed repeated random control and pointwise harm audit for edge consensus."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from common_metrics import affine_transform, localization_error_summary  # noqa: E402
from run_diffusion_refined_charting import solve_graph_harmonic_refinement  # noqa: E402
from methods.sc2_stage2_edge_consensus.run import graph_from_upper  # noqa: E402

SOURCE = Path(r"D:\diffusion_refined_charting")
OUT = ROOT / "results/sc2_stage2_edge_consensus"
DIST = SOURCE / "results/adp_old_fused/stride1_n20997_seed10/geodesic.npy"
Y0 = SOURCE / "results/adp_old_siamese/stride1_n20997_seed10/embedding.npz"


def pointwise(reference: np.ndarray, embedding: np.ndarray) -> np.ndarray:
    return np.linalg.norm(affine_transform(reference, embedding) - reference, axis=1)


def harm_report(base: np.ndarray, candidate: np.ndarray) -> dict:
    delta = base - candidate
    better = delta > 0
    worse = delta < 0
    return {
        "better_count": int(better.sum()),
        "worse_count": int(worse.sum()),
        "same_count": int((delta == 0).sum()),
        "mean_repair_among_better_m": float(delta[better].mean()),
        "mean_harm_among_worse_m": float((-delta[worse]).mean()),
        "total_repair_m": float(delta[better].sum()),
        "total_harm_m": float((-delta[worse]).sum()),
        "net_mean_improvement_m": float(delta.mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", choices=("eligible", "all", "incompatible"), default="eligible")
    args = parser.parse_args()
    score = np.load(OUT / "edge_scores_and_masks.npz")
    wrow, wcol, wval = score["row"], score["col"], score["w"]
    eligible = score["eligible"].astype(bool)
    sc2_selected = score["sc2_selected"]
    original_shuffled = score["shuffled_selected"]
    n = 20997
    upper = sp.coo_matrix((wval, (wrow, wcol)), shape=(n, n)).tocsr()
    y = np.load(Y0)
    init = y["embedding"].astype(np.float32)
    pos = y["positions"].astype(np.float32)
    saved = np.load(OUT / "embeddings.npz")
    base_err = pointwise(pos, saved["uniform"])
    sc2_err = pointwise(pos, saved["sc2"])
    shuffled_err = pointwise(pos, saved["shuffled"])
    geo = np.load(DIST, mmap_mode="r")
    g = np.asarray(geo[wrow, wcol], dtype=np.float64)
    cuts = np.quantile(g, np.linspace(0, 1, 11)[1:-1])
    bins = np.searchsorted(cuts, g, side="right")
    pool = (np.flatnonzero(eligible) if args.pool == "eligible" else np.flatnonzero(~eligible) if args.pool == "incompatible" else np.arange(len(eligible)))
    counts = [int(np.sum(bins[sc2_selected] == b)) for b in range(10)]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    rows = []
    for seed in range(10, 30):
        rng = np.random.default_rng(seed)
        picked = np.concatenate([
            rng.choice(pool[bins[pool] == b], counts[b], replace=False)
            for b in range(10)
        ])
        if args.pool == "eligible" and seed == 10 and not np.array_equal(picked, original_shuffled):
            raise ValueError("Seed-10 mask does not reproduce original shuffled control")
        factor = np.ones(len(wval), dtype=np.float32)
        factor[picked] = 0.5
        graph = graph_from_upper(upper, factor)
        print(f"random seed {seed} of 10..29", flush=True)
        z = solve_graph_harmonic_refinement(init, graph, 0.15, 200, device)
        m = localization_error_summary(pos, z)
        m["seed"] = seed
        m["total_graph_weight"] = float(graph.data.astype(np.float64).sum())
        rows.append(m)
    means = np.asarray([row["mean_m"] for row in rows])
    p95s = np.asarray([row["p95_m"] for row in rows])
    report = {
        "seed_range_inclusive": [10, 29],
        "candidate_pool": args.pool,
        "random_metrics": rows,
        "random_mean_median_m": float(np.median(means)),
        "random_mean_min_m": float(means.min()),
        "random_mean_max_m": float(means.max()),
        "random_mean_std_m": float(means.std(ddof=1)),
        "random_p95_median_m": float(np.median(p95s)),
        "random_beating_sc2_count": int(np.sum(means < np.mean(sc2_err))),
        "random_beating_uniform_count": int(np.sum(means < np.mean(base_err))),
        "sc2_pointwise": harm_report(base_err, sc2_err),
        "seed10_shuffled_pointwise": harm_report(base_err, shuffled_err),
        "sc2_mean_m": float(np.mean(sc2_err)),
        "uniform_mean_m": float(np.mean(base_err)),
        "seed10_shuffled_mean_m": float(np.mean(shuffled_err)),
    }
    (OUT / ("random_followup.json" if args.pool == "eligible" else "random_all_edges_followup.json" if args.pool == "all" else "random_incompatible_followup.json")).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("SUMMARY", json.dumps({k: v for k, v in report.items() if k != "random_metrics"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
