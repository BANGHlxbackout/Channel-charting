"""Probe the proposed Stage-1 loss on saved charts without training a model."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import sparse

from common_metrics import affine_transform, localization_error_summary, write_json
from methods.g1_acf_graph_fcf.compare import compare_paths
from methods.g1_acf_graph_fcf.graph_utils import rebuild_directed_graph
from run_adp_siamese import plot_chart


ROOT = Path("results/g1_acf_graph_fcf")
DIST = Path("results/adp_old_fused/stride1_n20997_seed10")
OUT = ROOT / "loss_direction_diagnostic"
PAIR_SEED = 20261026
PAIR_COUNT = 400000
LAMBDA = 1.0
STEP_FRACTION = 0.05


def diffusion_operator(fused: np.ndarray, geo: np.ndarray):
    directed = rebuild_directed_graph(fused, 20)
    n = directed.shape[0]
    row = np.repeat(np.arange(n, dtype=np.int64), np.diff(directed.indptr))
    col = directed.indices.astype(np.int64)
    lo, hi = np.minimum(row, col), np.maximum(row, col)
    keys = np.unique(lo * n + hi)
    lo, hi = keys // n, keys % n
    rho = np.maximum.reduceat(directed.data, directed.indptr[:-1]).astype(np.float64)
    cost = np.asarray(fused[lo, hi], dtype=np.float64)
    weight = np.exp(-cost**2 / np.maximum(rho[lo] * rho[hi], 1e-12))
    rows = np.r_[lo, hi, np.arange(n)]
    cols = np.r_[hi, lo, np.arange(n)]
    affinity = sparse.coo_matrix((np.r_[weight, weight, np.ones(n)], (rows, cols)),
                                 shape=(n, n)).tocsr()
    inv_degree = 1 / np.asarray(affinity.sum(axis=1)).ravel()
    transition = sparse.diags(inv_degree) @ affinity
    edge_geo = np.asarray(geo[lo, hi], dtype=np.float64)
    reverse_geo = np.asarray(geo[hi, lo], dtype=np.float64)
    scale = np.sqrt(np.bincount(np.r_[lo, hi],
                               weights=np.r_[weight * edge_geo**2,
                                             weight * reverse_geo**2],
                               minlength=n) * inv_degree)
    if not np.all(np.isfinite(scale)) or not np.allclose(
            np.asarray(transition.sum(axis=1)).ravel(), 1.0, atol=1e-12):
        raise ValueError("invalid fixed diffusion operator")
    return transition, scale, float(np.median(edge_geo)), len(keys)


def geo_loss_grad(z: np.ndarray, left: np.ndarray, right: np.ndarray,
                  target: np.ndarray, margin: float):
    difference = z[left] - z[right]
    predicted = np.linalg.norm(difference, axis=1)
    residual = predicted - target
    loss = float(np.mean(residual**2 / (target + margin)))
    coefficient = 2 * residual / (len(left) * (target + margin) * np.maximum(predicted, 1e-9))
    edge_gradient = coefficient[:, None] * difference
    gradient = np.zeros_like(z)
    np.add.at(gradient, left, edge_gradient)
    np.add.at(gradient, right, -edge_gradient)
    return loss, gradient


def diff_loss_grad(z: np.ndarray, transition: sparse.csr_matrix,
                   scale: np.ndarray, margin: float):
    residual = z - transition @ z
    weight = residual / (scale + margin)[:, None]
    loss = float(np.mean(np.sum(residual**2, axis=1) / (scale + margin)))
    gradient = 2 / len(z) * (weight - transition.T @ weight)
    return loss, gradient


def detail(z: np.ndarray, moved: np.ndarray, positions: np.ndarray):
    aligned = affine_transform(positions, z)
    after = affine_transform(positions, moved)
    old_error = np.linalg.norm(aligned - positions, axis=1)
    new_error = np.linalg.norm(after - positions, axis=1)
    change = new_error - old_error
    fit = np.linalg.lstsq(np.column_stack((z, np.ones(len(z)))), positions, rcond=None)[0]
    projected_move = (moved - z) @ fit[:2]
    toward_gt = np.sum((aligned - positions) * projected_move, axis=1) < 0
    first_order_mean_change = np.mean(np.sum((aligned - positions) * projected_move,
                                              axis=1) / np.maximum(old_error, 1e-9))
    q = np.quantile(old_error, [0, .2, .4, .6, .8, 1])
    groups = []
    for group in range(5):
        mask = (old_error >= q[group]) & (old_error < q[group + 1] if group < 4
                                           else old_error <= q[group + 1])
        groups.append({"initial_error_m": [float(q[group]), float(q[group + 1])],
                       "count": int(mask.sum()), "mean_change_m": float(change[mask].mean()),
                       "improved_fraction": float(np.mean(change[mask] < 0))})
    return {"first_order_mean_change_m": float(first_order_mean_change),
            "first_order_toward_GT_fraction": float(np.mean(toward_gt)),
            "mean_change_m": float(change.mean()),
            "improved_fraction": float(np.mean(change < 0)),
            "worsened_fraction": float(np.mean(change > 0)),
            "change_p95_m": float(np.quantile(change, .95)),
            "aligned_movement_median_m": float(np.median(np.linalg.norm(after - aligned, axis=1))),
            "by_initial_error_quintile": groups}


def main():
    global OUT
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=OUT)
    OUT = parser.parse_args().out_dir
    OUT.mkdir(parents=True, exist_ok=False)
    fused = np.load(DIST / "fused.npy", mmap_mode="r")
    geo = np.load(DIST / "geodesic.npy", mmap_mode="r")
    transition, scale, median_edge_geo, edge_count = diffusion_operator(fused, geo)
    n = len(scale)
    rng = np.random.default_rng(PAIR_SEED)
    left = rng.integers(n, size=PAIR_COUNT)
    right = rng.integers(n, size=PAIR_COUNT)
    keep = left != right
    left, right = left[keep], right[keep]
    target = np.asarray(geo[left, right], dtype=np.float64)
    np.savez(OUT / "loss_pair_sample.npz", left=left, right=right)
    margin = float(np.quantile(geo, .01))
    step_rms = STEP_FRACTION * median_edge_geo
    variants = {}
    analysis = {}
    source = {}
    for mode, title in (("identity", "G1-Identity"), ("graph", "G1-Graph")):
        original = ROOT / f"stride1_n20997_seed10_{mode}_full/embedding.npz"
        saved = np.load(original)
        indices, positions = saved["indices"], saved["positions"]
        z = saved["embedding"].astype(np.float64)
        if len(z) != n or not np.array_equal(indices, np.arange(n)):
            raise ValueError(f"{mode} saved index mismatch")
        geo_loss, geo_grad = geo_loss_grad(z, left, right, target, margin)
        diff_loss, diff_grad = diff_loss_grad(z, transition, scale, margin)
        if not np.all(np.isfinite(geo_grad + diff_grad)):
            raise ValueError(f"{mode} nonfinite gradient")
        source[title] = str(original.resolve())
        variants[title] = original
        control = {"geo": geo_grad, "diff": diff_grad, "new": geo_grad + LAMBDA * diff_grad}
        rows = {}
        for kind, gradient in control.items():
            rms = float(np.sqrt(np.mean(np.sum(gradient**2, axis=1))))
            moved = z - gradient * (step_rms / rms)
            after_geo, _ = geo_loss_grad(moved, left, right, target, margin)
            after_diff, _ = diff_loss_grad(moved, transition, scale, margin)
            result = {"mode": mode, "phase": "offline_embedding_gradient_probe",
                      "status": "diagnostic_only", "method": f"{title}/{kind}-step",
                      "source_embedding": str(original.resolve()),
                      "lambda": LAMBDA, "step_rms_raw": step_rms,
                      "loss_before": {"geo": geo_loss, "diff": diff_loss,
                                      "new": geo_loss + LAMBDA * diff_loss},
                      "loss_after": {"geo": after_geo, "diff": after_diff,
                                     "new": after_geo + LAMBDA * after_diff},
                      "pointwise": detail(z, moved, positions)}
            folder = OUT / f"{mode}_{kind}_step"
            folder.mkdir()
            np.savez(folder / "embedding.npz", indices=indices, positions=positions,
                     embedding=moved.astype(np.float32))
            before_aligned = affine_transform(positions, z)
            after_aligned = affine_transform(positions, moved)
            before_error = np.linalg.norm(before_aligned - positions, axis=1)
            after_error = np.linalg.norm(after_aligned - positions, axis=1)
            np.savez(folder / "pointwise_change.npz", indices=indices, positions=positions,
                     aligned_before=before_aligned, aligned_after=after_aligned,
                     error_before_m=before_error, error_after_m=after_error,
                     error_change_m=after_error - before_error)
            write_json(folder / "result.json", result)
            variants[result["method"]] = folder / "embedding.npz"
            rows[kind] = result
        analysis[mode] = {"initial_loss": {"geo": geo_loss, "diff": diff_loss,
                                             "new": geo_loss + LAMBDA * diff_loss},
                          "gradient_rms": {key: float(np.sqrt(np.mean(np.sum(value**2, axis=1))))
                                           for key, value in control.items()},
                          "gradient_cosine_geo_diff": float(np.sum(geo_grad * diff_grad) /
                              (np.linalg.norm(geo_grad) * np.linalg.norm(diff_grad))),
                          "steps": rows}
    b0 = Path("results/adp_old_siamese/stride1_n20997_seed10/embedding.npz")
    comparison = {"B0": b0, "G1-Identity": variants["G1-Identity"],
                  "G1-Graph": variants["G1-Graph"],
                  "G1-Identity/geo-step": variants["G1-Identity/geo-step"],
                  "G1-Graph/geo-step": variants["G1-Graph/geo-step"],
                  "G1-Identity/diff-step": variants["G1-Identity/diff-step"],
                  "G1-Graph/diff-step": variants["G1-Graph/diff-step"],
                  "G1-Identity/new-step": variants["G1-Identity/new-step"],
                  "G1-Graph/new-step": variants["G1-Graph/new-step"]}
    subset = ROOT / "comparison/metric_subset_seed10.npy"
    metrics = compare_paths(comparison, OUT / "comparison", seed=10, subset_path=subset)
    protocol_path = OUT / "comparison/eval_protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    protocol["colors"] = "same GT x/y mapping in all ten panels"
    write_json(protocol_path, protocol)
    for mode in ("identity", "graph"):
        title = f"G1-{mode.capitalize()}"
        focus = {title: variants[title],
                 f"{title}/geo-step": variants[f"{title}/geo-step"],
                 f"{title}/new-step": variants[f"{title}/new-step"]}
        compare_paths(focus, OUT / "comparison" / f"{mode}_focus", seed=10,
                      subset_path=subset)
    for name, path in comparison.items():
        if "-step" not in name:
            continue
        data = np.load(path)
        metric = metrics[name]
        plot_chart(path.parent / "chart.png", data["positions"], data["embedding"],
                   f"{name} (offline)\nCT={metric['ct']:.3f}, TW={metric['tw']:.3f}, KS={metric['ks']:.3f}")
        result_path = path.parent / "result.json"
        result = json.loads(result_path.read_text(encoding="utf-8"))
        result["metrics"] = {k: metric[k] for k in ("ct", "tw", "ks")}
        result["localization_errors_after_affine"] = {
            k: metric[k] for k in ("mean_m", "median_m", "p90_m", "p95_m", "max_m")}
        write_json(result_path, result)
    write_json(OUT / "direction_analysis.json", analysis)
    write_json(OUT / "run_config.json", {"phase": "offline_embedding_gradient_probe",
               "source_distance_run": str(DIST.resolve()), "source_embeddings": source,
               "pair_seed": PAIR_SEED, "pair_count": int(len(left)), "lambda": LAMBDA,
               "step_fraction_of_median_fused_edge_geodesic": STEP_FRACTION,
               "step_rms_raw": step_rms, "median_fused_edge_geodesic": median_edge_geo,
               "undirected_support_edges": edge_count,
               "operator": "row-normalized symmetric original fused Gaussian affinity plus self-loops",
               "model_training": False, "GT_usage": "post-step offline evaluation only",
               "metric_subset_source": str(subset.resolve())})
    print(json.dumps({"step_rms_raw": step_rms, "analysis": {
        mode: {"initial_loss": item["initial_loss"],
               "gradient_cosine_geo_diff": item["gradient_cosine_geo_diff"],
               "step_mean_m": {kind: metrics[f"G1-{mode.capitalize()}/{kind}-step"]["mean_m"]
                               for kind in ("geo", "diff", "new")}}
        for mode, item in analysis.items()}}, indent=2))


if __name__ == "__main__":
    main()
