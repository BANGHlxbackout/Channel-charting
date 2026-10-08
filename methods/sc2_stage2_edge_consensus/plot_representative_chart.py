"""Reproduce the fixed seed-18 compatible-edge chart and render it beside baseline."""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from common_metrics import affine_transform, chart_metrics, localization_error_summary
from run_diffusion_refined_charting import solve_graph_harmonic_refinement
from methods.sc2_stage2_edge_consensus.run import graph_from_upper

SOURCE = Path(r"D:\diffusion_refined_charting")
OUT = ROOT / "results/sc2_stage2_edge_consensus"
GEO = SOURCE / "results/adp_old_fused/stride1_n20997_seed10/geodesic.npy"
Y0 = SOURCE / "results/adp_old_siamese/stride1_n20997_seed10/embedding.npz"
SEED = 18


def main() -> None:
    fixed = np.load(OUT / "edge_scores_and_masks.npz")
    row, col, weights = fixed["row"], fixed["col"], fixed["w"]
    upper = sp.coo_matrix((weights, (row, col)), shape=(20997, 20997)).tocsr()
    eligible = fixed["eligible"].astype(bool)
    sc2_selected = fixed["sc2_selected"]
    dgeo = np.load(GEO, mmap_mode="r")
    edge_distance = np.asarray(dgeo[row, col], dtype=np.float64)
    cuts = np.quantile(edge_distance, np.linspace(0, 1, 11)[1:-1])
    bins = np.searchsorted(cuts, edge_distance, side="right")
    pool = np.flatnonzero(eligible)
    counts = [int(np.sum(bins[sc2_selected] == b)) for b in range(10)]
    rng = np.random.default_rng(SEED)
    selected = np.concatenate([
        rng.choice(pool[bins[pool] == b], counts[b], replace=False)
        for b in range(10)
    ])
    factor = np.ones(len(weights), dtype=np.float32)
    factor[selected] = 0.5
    graph = graph_from_upper(upper, factor)

    saved_y = np.load(Y0)
    y0 = saved_y["embedding"].astype(np.float32)
    positions = saved_y["positions"].astype(np.float32)
    random.seed(10)
    torch.manual_seed(10)
    subset = np.asarray(random.sample(range(len(y0)), int(0.2 * len(y0))), dtype=np.int64)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    candidate = solve_graph_harmonic_refinement(y0, graph, 0.15, 200, device)

    recorded = json.loads((OUT / "complete_metrics_followup.json").read_text(encoding="utf-8"))
    expected = next(run for run in recorded["groups"]["compatible_random"]["runs"] if run["seed"] == SEED)
    measured = {
        **chart_metrics(positions[subset], candidate[subset]),
        **localization_error_summary(positions, candidate),
    }
    if abs(measured["mean_m"] - expected["mean_m"]) > 1e-5 or abs(measured["p95_m"] - expected["p95_m"]) > 1e-5:
        raise ValueError(f"Seed-18 chart does not reproduce recorded metrics: {measured} vs {expected}")
    original = np.load(OUT / "embeddings.npz")["uniform"]
    original_metrics = localization_error_summary(positions, original)
    np.savez_compressed(
        OUT / "compatible_random_seed18_channel_chart.npz",
        indices=saved_y["indices"],
        positions=positions,
        embedding=candidate,
        seed=SEED,
    )

    colors = np.zeros((len(positions), 3), dtype=np.float32)
    colors[:, 0] = (positions[:, 0] - positions[:, 0].min()) / np.ptp(positions[:, 0])
    colors[:, 1] = (positions[:, 1] - positions[:, 1].min()) / np.ptp(positions[:, 1])
    colors[:, 2] = 0.25
    aligned_original = affine_transform(positions, original)
    aligned_candidate = affine_transform(positions, candidate)
    charts = (
        (positions, "Physical positions"),
        (aligned_original, f"Original Stage-2  |  mean {original_metrics['mean_m']:.6f} m"),
        (aligned_candidate, f"Compatible-edge random, seed 18  |  mean {measured['mean_m']:.6f} m"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    xlim = (min(chart[:, 0].min() for chart, _ in charts), max(chart[:, 0].max() for chart, _ in charts))
    ylim = (min(chart[:, 1].min() for chart, _ in charts), max(chart[:, 1].max() for chart, _ in charts))
    for ax, (chart, title) in zip(axes, charts):
        ax.scatter(chart[:, 0], chart[:, 1], c=colors, s=3, linewidths=0, rasterized=True)
        ax.set_title(title, fontsize=10)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.2)
        ax.set_xlabel("aligned x (m)")
        ax.set_ylabel("aligned y (m)")
    path = OUT / "channel_chart_seed18_vs_baseline.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print("CHART", path)
    print("METRICS", json.dumps(measured, indent=2))
    print("EXPECTED_MEAN", expected["mean_m"])


if __name__ == "__main__":
    main()
