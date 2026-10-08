"""Complete CT/TW/KS and affine-error comparison for the fixed random masks."""
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
from run_diffusion_refined_charting import solve_graph_harmonic_refinement  # noqa: E402
from methods.sc2_stage2_edge_consensus.run import graph_from_upper  # noqa: E402

SOURCE = Path(r"D:\diffusion_refined_charting")
OUT = ROOT / "results/sc2_stage2_edge_consensus"
GEO = SOURCE / "results/adp_old_fused/stride1_n20997_seed10/geodesic.npy"
Y0 = SOURCE / "results/adp_old_siamese/stride1_n20997_seed10/embedding.npz"
GROUPS = (
    ("compatible_random", "eligible", 0.15),
    ("all_edge_random", "all", 0.15),
    ("incompatible_random", "incompatible", 0.15),
    ("compatible_random_best_lambda", "eligible", 0.1344),
)
FIELDS = ("ct", "tw", "ks", "mean_m", "median_m", "p90_m", "p95_m", "max_m")


def main() -> None:
    saved = np.load(OUT / "edge_scores_and_masks.npz")
    row, col, weight = saved["row"], saved["col"], saved["w"]
    eligible = saved["eligible"].astype(bool)
    selected = saved["sc2_selected"]
    upper = sp.coo_matrix((weight, (row, col)), shape=(20997, 20997)).tocsr()
    chart = np.load(Y0)
    y0 = chart["embedding"].astype(np.float32)
    pos = chart["positions"].astype(np.float32)
    if len(y0) != 20997:
        raise ValueError("Unexpected chart size")
    random.seed(10)
    subset = np.asarray(random.sample(range(len(y0)), int(.2 * len(y0))), dtype=np.int64)
    geo = np.load(GEO, mmap_mode="r")
    d = np.asarray(geo[row, col], dtype=np.float64)
    bins = np.searchsorted(np.quantile(d, np.linspace(0, 1, 11)[1:-1]), d, side="right")
    counts = [int(np.sum(bins[selected] == b)) for b in range(10)]
    pools = {
        "eligible": np.flatnonzero(eligible),
        "all": np.arange(len(eligible)),
        "incompatible": np.flatnonzero(~eligible),
    }
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    output = {"protocol": "same N=20997, random.sample seed=10 and 4199 metrics subset, 200 Jacobi iterations", "groups": {}}
    for group_name, pool_name, lam in GROUPS:
        pool = pools[pool_name]
        rows = []
        for seed in range(10, 30):
            rng = np.random.default_rng(seed)
            picked = np.concatenate([
                rng.choice(pool[bins[pool] == b], counts[b], replace=False)
                for b in range(10)
            ])
            if pool_name == "eligible" and seed == 10 and not np.array_equal(picked, saved["shuffled_selected"]):
                raise ValueError("Seed-10 eligible mask mismatch")
            factor = np.ones(len(weight), dtype=np.float32)
            factor[picked] = .5
            graph = graph_from_upper(upper, factor)
            print(f"group={group_name}, seed={seed}, lambda={lam}", flush=True)
            z = solve_graph_harmonic_refinement(y0, graph, lam, 200, device)
            row_metrics = {
                "seed": seed,
                **chart_metrics(pos[subset], z[subset]),
                **localization_error_summary(pos, z),
            }
            rows.append(row_metrics)
            output["groups"][group_name] = {"pool": pool_name, "lambda": lam, "runs": rows}
            (OUT / "complete_metrics_followup.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
        summary = {}
        for field in FIELDS:
            values = np.asarray([r[field] for r in rows])
            summary[field] = {
                "median": float(np.median(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }
        output["groups"][group_name]["summary"] = summary
        (OUT / "complete_metrics_followup.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    print("SUMMARY", json.dumps({
        name: data["summary"] for name, data in output["groups"].items()
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
