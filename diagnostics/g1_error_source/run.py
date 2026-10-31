"""Read-only diagnosis of Stage-1 target, model fit, and physical chart error."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from common_metrics import affine_transform, write_json
from methods.g1_acf_graph_fcf.graph_utils import rebuild_directed_graph


ROOT = Path("results/g1_acf_graph_fcf")
DIST = Path("results/adp_old_fused/stride1_n20997_seed10")
OUT = ROOT / "error_source_diagnostic"
SEED = 20261027
DRAW = 400000


def aggregate(group: str, mask: np.ndarray, geo: np.ndarray, physical: np.ndarray,
              beta: float, native: dict, aligned: dict) -> dict:
    target_delta = beta * geo[mask] - physical[mask]
    row = {"group": group, "count": int(mask.sum()),
           "physical_median_m": float(np.median(physical[mask])),
           "Dgeo_median": float(np.median(geo[mask])),
           "target_bias_mean_m": float(np.mean(target_delta)),
           "target_abs_error_mean_m": float(np.mean(np.abs(target_delta))),
           "target_abs_error_p95_m": float(np.quantile(np.abs(target_delta), .95))}
    for name in ("graph", "identity"):
        fit = native[name][mask] - geo[mask]
        chart = aligned[name][mask] - physical[mask]
        row[f"{name}_Dgeo_bias_mean"] = float(np.mean(fit))
        row[f"{name}_Dgeo_abs_mean"] = float(np.mean(np.abs(fit)))
        row[f"{name}_weighted_stress"] = float(np.mean(fit**2 / (geo[mask] + MARGIN)))
        row[f"{name}_physical_bias_mean_m"] = float(np.mean(chart))
        row[f"{name}_physical_abs_mean_m"] = float(np.mean(np.abs(chart)))
    return row


def main() -> None:
    global OUT
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=OUT)
    OUT = parser.parse_args().out_dir
    if OUT.exists():
        raise FileExistsError(OUT)
    OUT.mkdir(parents=True)
    saved = np.load(DIST / "distance_data.npz")
    indices, positions = saved["indices"], saved["positions"].astype(np.float64)
    n = len(indices)
    embeddings = {}
    for name in ("graph", "identity"):
        x = np.load(ROOT / f"stride1_n20997_seed10_{name}_full/embedding.npz")
        if not np.array_equal(x["indices"], indices) or not np.array_equal(x["positions"], saved["positions"]):
            raise ValueError(f"{name} index or position mismatch")
        embeddings[name] = {"native": x["embedding"].astype(np.float64),
                            "aligned": affine_transform(positions, x["embedding"].astype(np.float64))}
    fused = np.load(DIST / "fused.npy", mmap_mode="r")
    adp = np.load(DIST / "adp.npy", mmap_mode="r")
    dgeo = np.load(DIST / "geodesic.npy", mmap_mode="r")
    global MARGIN
    MARGIN = float(np.quantile(dgeo, .01))
    rng = np.random.default_rng(SEED)
    left, right = rng.integers(n, size=DRAW), rng.integers(n, size=DRAW)
    keep = left != right
    left, right = left[keep], right[keep]
    np.savez(OUT / "random_pairs.npz", left=left, right=right)
    graph = rebuild_directed_graph(fused, 20)
    rows = np.repeat(np.arange(n, dtype=np.int64), np.diff(graph.indptr))
    cols = graph.indices.astype(np.int64)
    key = np.unique(np.minimum(rows, cols) * n + np.maximum(rows, cols))
    edge_l, edge_r = key // n, key % n
    edge_fused = np.asarray(fused[edge_l, edge_r], dtype=np.float64)
    edge_adp = np.asarray(adp[edge_l, edge_r], dtype=np.float64)
    edge_time = edge_fused < edge_adp - 1e-5
    np.savez(OUT / "edge_groups.npz", left=edge_l, right=edge_r,
             time_source=edge_time)

    def values(a: np.ndarray, b: np.ndarray):
        geo = np.asarray(dgeo[a, b], dtype=np.float64)
        physical = np.linalg.norm(positions[a] - positions[b], axis=1)
        native = {name: np.linalg.norm(x["native"][a] - x["native"][b], axis=1)
                  for name, x in embeddings.items()}
        aligned = {name: np.linalg.norm(x["aligned"][a] - x["aligned"][b], axis=1)
                   for name, x in embeddings.items()}
        return geo, physical, native, aligned

    geo, physical, native, aligned = values(left, right)
    # One global scalar is fitted using GT for *offline diagnosis only*.
    beta = float(np.dot(geo, physical) / np.dot(geo, geo))
    groups = [aggregate("random_all", np.ones(len(left), bool), geo, physical,
                        beta, native, aligned)]
    edges = np.quantile(geo, [0, .1, .4, .7, .9, 1])
    for j in range(len(edges) - 1):
        mask = (geo >= edges[j]) & (geo < edges[j + 1] if j < 4 else geo <= edges[j + 1])
        groups.append(aggregate(f"random_Dgeo_bin_{j}", mask, geo, physical,
                                beta, native, aligned))
    egeo, ephysical, enative, ealigned = values(edge_l, edge_r)
    for label, mask in (("fused_edge_all", np.ones(len(edge_l), bool)),
                        ("fused_edge_time_source", edge_time),
                        ("fused_edge_adp_source", ~edge_time)):
        groups.append(aggregate(label, mask, egeo, ephysical, beta, enative, ealigned))
    xedges = np.quantile(positions[:, 0], [0, .25, .5, .75, 1])
    yedges = np.quantile(positions[:, 1], [0, .25, .5, .75, 1])
    xbin = np.minimum(np.searchsorted(xedges[1:-1], positions[:, 0], side="right"), 3)
    ybin = np.minimum(np.searchsorted(yedges[1:-1], positions[:, 1], side="right"), 3)
    region = xbin * 4 + ybin
    node_errors = {name: np.linalg.norm(x["aligned"] - positions, axis=1)
                   for name, x in embeddings.items()}
    node_rows = []
    for k in range(16):
        node_mask = region == k
        pair_mask = region[left] == k
        if not node_mask.any() or not pair_mask.any():
            continue
        group = aggregate(f"origin_region_{k}", pair_mask, geo, physical,
                          beta, native, aligned)
        group["node_count"] = int(node_mask.sum())
        for name in ("graph", "identity"):
            group[f"{name}_node_error_mean_m"] = float(node_errors[name][node_mask].mean())
            group[f"{name}_node_error_p95_m"] = float(np.quantile(node_errors[name][node_mask], .95))
        node_rows.append(group)

    def incident_mean(a: np.ndarray, b: np.ndarray, value: np.ndarray) -> np.ndarray:
        count = np.bincount(np.r_[a, b], minlength=n)
        total = np.bincount(np.r_[a, b], weights=np.r_[value, value], minlength=n)
        return total / np.maximum(count, 1)

    random_target_error = incident_mean(left, right, np.abs(beta * geo - physical))
    edge_target_error = incident_mean(edge_l, edge_r,
                                      np.abs(beta * egeo - ephysical))
    associations = {}
    for name in ("graph", "identity"):
        error = node_errors[name]
        fit_error = incident_mean(left, right, np.abs(native[name] - geo))
        associations[name] = {
            "spearman_node_error_vs_random_target_abs": float(spearmanr(error, random_target_error).statistic),
            "spearman_node_error_vs_edge_target_abs": float(spearmanr(error, edge_target_error).statistic),
            "spearman_node_error_vs_random_model_Dgeo_abs": float(spearmanr(error, fit_error).statistic),
        }
    write_json(OUT / "node_associations.json", associations)

    def write_csv(path: Path, records: list[dict]) -> None:
        keys = list(dict.fromkeys(k for row in records for k in row))
        with path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=keys)
            writer.writeheader()
            writer.writerows(records)

    write_csv(OUT / "pair_groups.csv", groups)
    write_csv(OUT / "spatial_regions.csv", node_rows)
    summary = {"seed": SEED, "random_pair_count": len(left), "edge_count": len(edge_l),
               "fused_edge_source_counts": {"time": int(edge_time.sum()),
                                            "adp": int((~edge_time).sum())},
               "beta_m_per_Dgeo": beta, "Dgeo_quantile_edges": edges.tolist(),
               "spearman_Dgeo_physical": float(spearmanr(geo, physical).statistic),
               "margin": MARGIN, "input_paths": {"distance": str(DIST.resolve()),
               **{name: str((ROOT / f"stride1_n20997_seed10_{name}_full/embedding.npz").resolve())
                  for name in embeddings}},
               "GT_usage": "offline beta fitting, attribution, and evaluation only; no training or checkpoint selection"}
    write_json(OUT / "summary.json", summary)
    print(json.dumps({"summary": summary, "groups": groups[:9]}, indent=2))


if __name__ == "__main__":
    main()
