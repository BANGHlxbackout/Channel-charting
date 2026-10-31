"""Generate GT-only and channel-chart-only PNGs for diffusion and euclidean results."""
from __future__ import annotations
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from common_metrics import affine_transform

RUNS = [
    (
        "results/diffusion_refined_charting/stride1_n20997_k128_lambda0.15_seed10/embedding.npz",
        "results/diffusion_refined_charting/stride1_n20997_k128_lambda0.15_seed10/result.json",
        "Diffusion Refined",
        "diffusion_refined",
    ),
    (
        "results/euclidean_knn_refined_charting/stride1_n20997_euclidean_k128_lambda0.15_seed10/embedding.npz",
        "results/euclidean_knn_refined_charting/stride1_n20997_euclidean_k128_lambda0.15_seed10/result.json",
        "Euclidean-KNN Refined",
        "euclidean_knn_refined",
    ),
    (
        "results/triplet_baseline_notebook/stride4_n20997_seed10/embedding.npz",
        "results/triplet_baseline_notebook/stride4_n20997_seed10/result.json",
        "Triplet",
        "triplet",
    ),
    (
        "results/adp_old_siamese/stride1_n20997_seed10/embedding.npz",
        "results/adp_old_siamese/stride1_n20997_seed10/result.json",
        "ADP Siamese",
        "adp_siamese",
    ),
    (
        "results/adp_old_fused/stride1_n20997_seed10/embedding.npz",
        "results/adp_old_fused/stride1_n20997_seed10/result.json",
        "ADP MDS",
        "adp_mds",
    ),
]

OUT = Path(__file__).parent


def _colors(reference: np.ndarray) -> np.ndarray:
    c = np.zeros((len(reference), 3), dtype=np.float32)
    c[:, 0] = (reference[:, 0] - reference[:, 0].min()) / np.ptp(reference[:, 0])
    c[:, 1] = (reference[:, 1] - reference[:, 1].min()) / np.ptp(reference[:, 1])
    c[:, 2] = 0.25
    return c


def _save(path: Path, x: np.ndarray, y: np.ndarray, colors: np.ndarray, title: str) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(4.5, 4))
    ax.scatter(x, y, c=colors, s=5, linewidths=0)
    ax.set_title(title)
    ax.axis("equal")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


CHART_ONLY = [
    (
        "results/g1_acf_graph_fcf/stride1_n20997_seed10_graph_full/embedding.npz",
        "results/g1_acf_graph_fcf/stride1_n20997_seed10_graph_full/result.json",
        "Graph-FCF",
        "graph_fcf",
    ),
]

for npz_rel, json_rel, label, stem in RUNS:
    data = np.load(ROOT / npz_rel)
    ref = data["positions"]
    emb = data["embedding"]
    colors = _colors(ref)
    aligned = affine_transform(ref, emb)

    with open(ROOT / json_rel) as f:
        result = json.load(f)
    m = result["metrics"]
    chart_title = f"{label}\nCT={m['ct']:.3f}, TW={m['tw']:.3f}, KS={m['ks']:.3f}"

    _save(OUT / f"{stem}_gt_only.png", ref[:, 0], ref[:, 1], colors, "Ground Truth")
    _save(OUT / f"{stem}_chart_only.png", aligned[:, 0], aligned[:, 1], colors, chart_title)
    print(f"saved {stem}")

for npz_rel, json_rel, label, stem in CHART_ONLY:
    data = np.load(ROOT / npz_rel)
    ref = data["positions"]
    emb = data["embedding"]
    colors = _colors(ref)
    aligned = affine_transform(ref, emb)

    with open(ROOT / json_rel) as f:
        result = json.load(f)
    m = result["metrics"]
    chart_title = f"{label}\nCT={m['ct']:.3f}, TW={m['tw']:.3f}, KS={m['ks']:.3f}"

    _save(OUT / f"{stem}_chart_only.png", aligned[:, 0], aligned[:, 1], colors, chart_title)
    print(f"saved {stem}")
