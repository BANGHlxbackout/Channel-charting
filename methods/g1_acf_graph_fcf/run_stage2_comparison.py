"""E1: apply the unchanged advisor Stage-2 operator to three saved charts."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy import sparse

from common_metrics import write_json
from methods.g1_acf_graph_fcf.compare import compare_paths
from methods.g1_acf_graph_fcf.graph_utils import file_identity
from run_adp_siamese import plot_chart
from run_diffusion_refined_charting import (
    build_local_scaled_graph, compute_neighbor_indices, solve_graph_harmonic_refinement,
)

S2 = {"top_k": 512, "graph_k": 128, "lambda_smooth": 0.15, "iterations": 200}
DISTANCE = Path("results/adp_old_fused/stride1_n20997_seed10")
SUBSET = Path("results/g1_acf_graph_fcf/comparison/metric_subset_seed10.npy")
INITIAL = {
    "B0": Path("results/adp_old_siamese/stride1_n20997_seed10/embedding.npz"),
    "G1-Identity": Path("results/g1_acf_graph_fcf/stride1_n20997_seed10_identity_full/embedding.npz"),
    "G1-Graph": Path("results/g1_acf_graph_fcf/stride1_n20997_seed10_graph_full/embedding.npz"),
}


def digest(*arrays: np.ndarray) -> str:
    hasher = hashlib.sha256()
    for array in arrays:
        hasher.update(np.ascontiguousarray(array).tobytes())
    return hasher.hexdigest()


def load_initial_charts() -> tuple[dict, np.ndarray, np.ndarray]:
    indices = np.load(DISTANCE / "distance_data.npz")["indices"]
    charts = {name: np.load(path) for name, path in INITIAL.items()}
    positions = charts["B0"]["positions"]
    for name, saved in charts.items():
        if not np.array_equal(saved["indices"], indices):
            raise ValueError(f"{name}: sample order mismatch")
        if not np.array_equal(saved["positions"], positions):
            raise ValueError(f"{name}: position fields mismatch")
        y = saved["embedding"]
        if y.shape != (20997, 2) or not np.isfinite(y).all() or np.std(y, axis=0).max() < 1e-8:
            raise ValueError(f"{name}: invalid chart")
    return charts, indices, positions


def refine_charts(embeddings: dict[str, np.ndarray], graph, device: torch.device) -> dict:
    # This function receives no positions. The exact same graph object is reused.
    return {name: solve_graph_harmonic_refinement(y, graph, S2["lambda_smooth"],
                                                 S2["iterations"], device)
            for name, y in embeddings.items()}


def finish_evaluation(paths: dict[str, Path], output: Path) -> dict:
    results = compare_paths(paths, output / "comparison", seed=10, subset_path=SUBSET)
    for name, path in paths.items():
        if path.parent.parent != output:
            continue  # Original result JSONs and charts are read-only.
        result_path = path.parent / "result.json"
        record = json.loads(result_path.read_text())
        row = results[name]
        record.update(status="completed", metrics={key: row[key] for key in ("ct", "tw", "ks")},
                      localization_errors_after_affine={key: row[key] for key in
                          ("mean_m", "median_m", "p90_m", "p95_m", "max_m")})
        write_json(result_path, record)
        saved = np.load(path)
        plot_chart(path.parent / "chart.png", saved["positions"], saved["embedding"],
                   f"{name}\nCT={row['ct']:.3f}, TW={row['tw']:.3f}, KS={row['ks']:.3f}")
    return results


def check_protected_inputs(root: Path) -> None:
    preflight = json.loads((root / "preflight.json").read_text())
    before = preflight["protected_inputs"]
    changed = [item["path"] for item in before
               if file_identity(Path(item["path"])) != item]
    if changed:
        raise ValueError(f"protected input files changed: {changed}")
    if sorted(path.name for path in DISTANCE.iterdir()) != preflight["distance_dir_listing"]:
        raise ValueError("source distance-run directory contents changed")


def run(args: argparse.Namespace) -> None:
    started = time.perf_counter()
    output = args.out_root / "e1_stage2"
    output.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)
    charts, indices, positions = load_initial_charts()
    geo = np.load(DISTANCE / "geodesic.npy", mmap_mode="r")
    cache = output / "cache"
    cache.mkdir()
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    neighbors = compute_neighbor_indices(geo, S2["top_k"], device, cache / "geodesic_top512.npy")
    repeated = compute_neighbor_indices(geo, S2["top_k"], device, cache / "repeat_top512.npy")
    if not np.array_equal(neighbors, repeated):
        raise ValueError("fresh top-k neighbor calculation is not reproducible")
    (cache / "repeat_top512.npy").unlink()
    if np.any(neighbors[:, :S2["graph_k"]] == np.arange(len(indices))[:, None]):
        raise ValueError("Stage-2 graph has unexpected self-neighbors")
    graph = build_local_scaled_graph(geo, neighbors, S2["graph_k"])
    graph_csr = graph.tocsr()
    sparse.save_npz(cache / "stage2_graph.npz", graph_csr)
    graph_hash = digest(graph_csr.indptr, graph_csr.indices, graph_csr.data)
    source_cache = DISTANCE / "geodesic_top512.npy"
    old_cache = np.load(source_cache, mmap_mode="r") if source_cache.exists() else None
    manifest = {"status": "computing", "parameters": S2, "graph_hash": graph_hash,
                "graph_entries": int(graph.nnz), "neighbor_hash": digest(neighbors),
                "fresh_topk_repeat_exact": True,
                "old_neighbor_cache": str(source_cache.resolve()) if old_cache is not None else None,
                "old_neighbor_different_entries": int(np.count_nonzero(neighbors != old_cache))
                    if old_cache is not None and old_cache.shape == neighbors.shape else None,
                "input_paths": {name: str(path.resolve()) for name, path in INITIAL.items()},
                "command": [sys.executable, *sys.argv], "device": str(device)}
    write_json(output / "manifest.json", manifest)
    paths = dict(INITIAL)
    solved = {}
    for name, chart in charts.items():
        torch.cuda.reset_peak_memory_stats(device)
        begin = time.perf_counter()
        z = refine_charts({name: chart["embedding"]}, graph, device)[name]
        elapsed = time.perf_counter() - begin
        solved[name] = z
        if not np.isfinite(z).all():
            raise ValueError(f"{name}+S2: nonfinite output")
        folder = output / name.lower().replace("-", "_")
        folder.mkdir()
        path = folder / "embedding.npz"
        np.savez(path, indices=indices, positions=positions, embedding=z)
        residual = z + S2["lambda_smooth"] * (
            np.asarray(graph_csr.sum(axis=1)) * z - graph_csr @ z) - chart["embedding"]
        record = {"status": "computed_awaiting_evaluation", "phase": "full_postprocessing",
                  "method": name + "+S2", "num_samples": len(indices), "seed": 10,
                  **S2, "graph_hash": graph_hash, "input_embedding": str(INITIAL[name].resolve()),
                  "postprocess_time_s": elapsed, "peak_vram_bytes": torch.cuda.max_memory_allocated(device),
                  "relative_system_residual": float(np.linalg.norm(residual) / np.linalg.norm(chart["embedding"])),
                  "protocol": "same-node transductive; GT used only after refinement"}
        write_json(folder / "result.json", record)
        write_json(folder / "run_config.json", {**S2, "source": str(INITIAL[name].resolve()),
                                                "device": str(device), "seed": 10})
        paths[name + "+S2"] = path
    repeat_z = refine_charts({"B0": charts["B0"]["embedding"]}, graph, device)["B0"]
    manifest["repeat_solver_max_abs_diff"] = float(np.max(np.abs(repeat_z - solved["B0"])))
    if not np.allclose(repeat_z, solved["B0"], atol=1e-4, rtol=1e-5):
        raise ValueError("Stage-2 solver repeat exceeds numerical tolerance")
    results = finish_evaluation(paths, output)
    metrics = ("ct", "tw", "ks", "mean_m", "median_m", "p90_m", "p95_m", "max_m")
    manifest["graph_s2_minus_identity_s2"] = {key: results["G1-Graph+S2"][key] -
                                              results["G1-Identity+S2"][key] for key in metrics}
    old_z_path = Path("results/diffusion_refined_charting/stride1_n20997_k128_lambda0.15_seed10/embedding.npz")
    if old_z_path.exists():
        old = np.load(old_z_path)
        if not np.array_equal(old["indices"], indices):
            raise ValueError("historical Stage-2 indices mismatch")
        manifest["historical_b0_s2_raw_max_abs_diff"] = float(np.max(np.abs(old["embedding"] - solved["B0"])))
    check_protected_inputs(args.out_root)
    manifest.update(status="completed", total_time_s=time.perf_counter() - started,
                    source_inputs_unchanged=True, results=results)
    write_json(output / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, default=Path("results/g1_acf_graph_fcf_followup"))
    parser.add_argument("--device", default="cuda:0")
    run(parser.parse_args())
