"""CDF of localization errors for ADP Siamese and Graph-FCF."""
from __future__ import annotations
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from common_metrics import affine_transform

METHODS = [
    ("results/adp_old_siamese/stride1_n20997_seed10/embedding.npz", "ADP-Siamese"),
    ("results/adp_old_fused/stride1_n20997_seed10/embedding.npz", "ADP-MDS"),
    ("results/euclidean_knn_refined_charting/stride1_n20997_euclidean_k128_lambda0.15_seed10/embedding.npz", "Euclidean-KNN Refined"),
    ("results/diffusion_refined_charting/stride1_n20997_k128_lambda0.15_seed10/embedding.npz", "Laplacian Refined"),
    ("results/g1_acf_graph_fcf/stride1_n20997_seed10_graph_full/embedding.npz", "Graph-FCF"),
]

fig, ax = plt.subplots(1, 1, figsize=(8, 5))

for npz_rel, label in METHODS:
    data = np.load(ROOT / npz_rel)
    ref = data["positions"]
    emb = data["embedding"]
    aligned = affine_transform(ref, emb)
    errors = np.linalg.norm(aligned - ref, axis=1)

    count, bins_count = np.histogram(errors, bins=200)
    pdf = count / sum(count)
    cdf = np.cumsum(pdf)
    bins_count[0] = 0
    cdf = np.append([0], cdf)

    ax.plot(bins_count, cdf, label=label)

ax.set_xlim((0, 2))
ax.set_xlabel("Absolute Localization Error [m]")
ax.set_ylabel("CDF")
ax.set_title("CDF of Localization Error")
ax.legend()
ax.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig(ROOT / "tmp" / "cdf_all_methods.png", dpi=180)
plt.close(fig)
print("saved cdf_all_methods.png")
