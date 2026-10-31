"""Gate 0: reproduce the fused Stage-1 graph and score its existing edges."""

from __future__ import annotations

import argparse
import ast
import json
import platform
import time
from pathlib import Path

import numpy as np
import scipy
import torch
from scipy import sparse
from scipy.sparse.csgraph import connected_components, dijkstra
from sklearn.neighbors import NearestNeighbors, kneighbors_graph

from run_adp_fused import normalize_csi_vectors


def view_mapping(preprocess_path: Path, spec_path: Path) -> list[str]:
    tree = ast.parse(preprocess_path.read_text(encoding="utf-8"))
    assignments = next(
        ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "ASSIGNMENTS" for target in node.targets)
    )
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    official = {
        item["name"].removeprefix("Antenna Array "): {
            antenna for group in item["assignments"] for antenna in group
        }
        for item in spec["antennas"]
    }
    names = [next((name for name, indices in official.items() if set(group) == indices), None)
             for group in assignments]
    if names != ["C", "B", "A", "D"]:
        raise ValueError(f"preprocessing/spec array mapping mismatch: {names}")
    return names


def load_scoring_inputs(data_path: Path) -> tuple[np.ndarray, np.ndarray]:
    return (np.load(data_path / "H.npy", mmap_mode="r"),
            np.load(data_path / "timestamps.npy", mmap_mode="r"))


def raw_csi_spotcheck(repo_root: Path, h: np.ndarray,
                      timestamps: np.ndarray) -> tuple[float, float]:
    """Check the first two retained cf02 samples without reading UE positions."""
    from preprocess_adp_old_shard import process_csi
    from preprocess_dichasus_numpy import decode_tensor
    from tfrecord import example_pb2
    from tfrecord.reader import tfrecord_iterator

    offsets = json.loads((repo_root / "datasets/reftx-offsets-dichasus-cf02.json")
                         .read_text(encoding="utf-8"))
    max_h_error = 0.0
    max_t_error = 0.0
    for raw_index, raw in enumerate(tfrecord_iterator(
            str(repo_root / "datasets/dichasus-cf02.tfrecords"), None)):
        if raw_index in (0, 4):
            example = example_pb2.Example()
            example.ParseFromString(bytes(raw))
            features = example.features.feature
            csi = decode_tensor(features["csi"].bytes_list.value[0])
            reconstructed = process_csi(csi, offsets).reshape(32, 13)
            saved_index = raw_index // 4
            max_h_error = max(max_h_error,
                              float(np.max(np.abs(reconstructed - h[saved_index]))))
            max_t_error = max(max_t_error,
                              abs(float(features["time"].float_list.value[0])
                                  - float(timestamps[saved_index])))
        if raw_index == 4:
            break
    return max_h_error, max_t_error


def rebuild_graph(fused: np.ndarray, neighbors: int) -> sparse.csr_matrix:
    model = NearestNeighbors(n_neighbors=neighbors, metric="precomputed", n_jobs=-1).fit(fused)
    return kneighbors_graph(
        model, neighbors, metric="precomputed", mode="distance", n_jobs=-1
    ).tocsr()


def view_adp_block(view_tensor: torch.Tensor, start: int, stop: int) -> np.ndarray:
    dots = torch.einsum("ilm,jlm->ijl", torch.conj(view_tensor[start:stop]), view_tensor)
    block = (13.0 - torch.sum(torch.abs(dots) ** 2, dim=-1)).cpu().numpy()
    block[np.arange(stop - start), np.arange(start, stop)] = 0.0
    return block


def kth_nonself(distances: np.ndarray, start: int, k: int) -> np.ndarray:
    rows = np.arange(len(distances))
    self_columns = np.arange(start, start + len(distances))
    distances[rows, self_columns] = np.inf
    return np.partition(distances, k - 1, axis=1)[:, k - 1]


def score_edges(edge_adp: np.ndarray, scales: np.ndarray, left: np.ndarray,
                right: np.ndarray, adp: np.ndarray, timestamps: np.ndarray,
                gamma: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    d = np.maximum(edge_adp.astype(np.float64), 0.0)
    scale = np.sqrt(np.maximum(scales[left].astype(np.float64), 0.0)
                    * np.maximum(scales[right].astype(np.float64), 0.0))
    q = (d + 1e-12) / (scale + 1e-12)
    x = np.log(q)
    center = np.median(x, axis=1)
    s = 1.4826 * np.median(np.abs(x - center[:, None]), axis=1)
    s95 = float(np.quantile(s, 0.95))
    if not np.isfinite(s95) or s95 <= 0:
        raise ValueError(f"invalid disagreement normalization s95={s95}")
    u = np.minimum(s / (s95 + 1e-12), 1.0)
    time_cost = np.abs(timestamps[left] - timestamps[right]).astype(np.float32)
    time_cost *= np.float32(gamma)
    source_adp = np.asarray(adp[left, right]) <= time_cost
    return q, s, u, source_adp


def apply_risk_cost(graph: sparse.csr_matrix, pair_index: np.ndarray,
                    u: np.ndarray, source_adp: np.ndarray, kappa: float) -> sparse.csr_matrix:
    changed = graph.copy()
    changed.data = (graph.data * (1.0 + np.float32(kappa) * u[pair_index]
                                  * source_adp[pair_index])).astype(graph.data.dtype)
    return changed


def write_result(path: Path, result: dict) -> None:
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict:
    start_time = time.time()
    output = args.out_dir / "edge_risk"
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "gate0_regression.json"
    result: dict = {
        "status": "running",
        "input": {"data_path": str(args.data_path), "distance_run": str(args.distance_run)},
        "config": {"block_size": args.block_size, "device": args.device, "k_view": 20,
                   "kappa_check": 0.5, "source_indices": np.linspace(0, 20996, 32, dtype=int).tolist()},
        "software": {"python": platform.python_version(), "numpy": np.__version__,
                     "scipy": scipy.__version__, "torch": torch.__version__},
        "checks": {},
    }

    def check(name: str, passed: bool, **details: object) -> None:
        result["checks"][name] = {"passed": bool(passed), **details}
        write_result(report_path, result)
        if not passed:
            raise ValueError(f"Gate 0 check failed: {name}")

    try:
        mapping = view_mapping(args.repo_root / "preprocess_adp_old_shard.py",
                               args.repo_root / "datasets/specv2.0.json")
        check("view_mapping", True, stored_views=mapping)

        h, timestamps = load_scoring_inputs(args.data_path)
        adp = np.load(args.distance_run / "adp.npy", mmap_mode="r")
        fused = np.load(args.distance_run / "fused.npy", mmap_mode="r")
        geodesic = np.load(args.distance_run / "geodesic.npy", mmap_mode="r")
        metadata = json.loads((args.distance_run / "result.json").read_text(encoding="utf-8"))
        with np.load(args.distance_run / "distance_data.npz") as saved:
            indices = saved["indices"]
        n = len(h)
        data_ok = (h.shape == (20997, 32, 13) and h.dtype == np.complex64
                   and timestamps.shape == (n,) and timestamps.dtype == np.float32
                   and all(a.shape == (n, n) and a.dtype == np.float32
                           for a in (adp, fused, geodesic))
                   and np.array_equal(indices, np.arange(n))
                   and metadata["neighbors"] == 20 and metadata["num_samples"] == n)
        check("input_shapes_and_order", data_ok, n=n, h_shape=list(h.shape),
              matrix_shape=list(fused.shape), neighbors=metadata["neighbors"],
              gamma=metadata["gamma"])
        h_error, t_error = raw_csi_spotcheck(args.repo_root, h, timestamps)
        check("raw_csi_order_spotcheck", h_error == 0 and t_error == 0,
              checked_raw_indices=[0, 4], max_h_abs=h_error, max_timestamp_abs=t_error)

        print("Rebuilding baseline 20-NN graph", flush=True)
        graph = rebuild_graph(fused, metadata["neighbors"])
        rows = np.repeat(np.arange(n, dtype=np.int32), np.diff(graph.indptr))
        cols = graph.indices
        components = int(connected_components(graph, directed=False)[0])
        graph_ok = (graph.nnz == n * 20 and np.all(np.diff(graph.indptr) == 20)
                    and not np.any(rows == cols) and components == 1
                    and np.all(np.isfinite(graph.data)))
        check("baseline_graph", graph_ok, nnz=int(graph.nnz), components=components,
              self_edges=int(np.count_nonzero(rows == cols)))

        sources = np.linspace(0, n - 1, 32, dtype=int)
        distances = dijkstra(graph, directed=False, indices=sources).astype(np.float32)
        reference = np.asarray(geodesic[sources])
        diff = np.abs(distances - reference)
        path_ok = np.allclose(distances, reference, rtol=1e-5, atol=1e-4)
        check("baseline_shortest_paths", path_ok, sources=sources.tolist(),
              max_abs=float(np.max(diff)),
              max_rel=float(np.max(diff / np.maximum(np.abs(reference), 1e-12))),
              mismatches=int(np.count_nonzero(~np.isclose(distances, reference, rtol=1e-5, atol=1e-4))))
        del distances, reference, diff

        directed_keys = np.minimum(rows, cols).astype(np.int64) * n + np.maximum(rows, cols)
        keys, pair_index = np.unique(directed_keys, return_inverse=True)
        left, right = keys // n, keys % n
        baseline_min = np.full(len(keys), np.inf)
        baseline_max = np.full(len(keys), -np.inf)
        np.minimum.at(baseline_min, pair_index, graph.data)
        np.maximum.at(baseline_max, pair_index, graph.data)
        baseline_asym = baseline_max - baseline_min
        check("baseline_reciprocal_weights", True,
              differing_pairs=int(np.count_nonzero(baseline_asym)),
              max_abs_difference=float(np.max(baseline_asym)))
        edge_adp = np.empty((len(keys), 4), dtype=np.float32)
        scales = np.empty((n, 4), dtype=np.float32)
        z = normalize_csi_vectors(np.asarray(h).reshape(n, 4, 8, 13)).reshape(n, 4, 13, 8)
        z_device = torch.from_numpy(z).to(args.device)
        view_tensors = [z_device[:, view].contiguous() for view in range(4)]
        max_regression = 0.0
        regression_mismatches = 0
        min_view = np.full(4, np.inf)
        max_view = np.full(4, -np.inf)
        with torch.no_grad():
            for block_start in range(0, n, args.block_size):
                block_stop = min(block_start + args.block_size, n)
                sum_views = np.zeros((block_stop - block_start, n), dtype=np.float32)
                first = np.searchsorted(left, block_start)
                last = np.searchsorted(left, block_stop)
                for view, view_tensor in enumerate(view_tensors):
                    d = view_adp_block(view_tensor, block_start, block_stop)
                    min_view[view] = min(min_view[view], float(np.min(d)))
                    max_view[view] = max(max_view[view], float(np.max(d)))
                    edge_adp[first:last, view] = d[left[first:last] - block_start, right[first:last]]
                    sum_views += d
                    scales[block_start:block_stop, view] = kth_nonself(d, block_start, 20)
                    del d
                cached = np.asarray(adp[block_start:block_stop])
                error = np.abs(sum_views - cached)
                max_regression = max(max_regression, float(np.max(error)))
                regression_mismatches += int(np.count_nonzero(error > 1e-5 + 1e-5 * np.abs(cached)))
                if block_stop == n or block_stop % (args.block_size * 20) == 0:
                    print(f"ADP rows {block_stop}/{n}, regression mismatches {regression_mismatches}", flush=True)
        check("view_adp_range_and_sum", regression_mismatches == 0
              and np.all(min_view >= -1e-5) and np.all(max_view <= 13 + 1e-5),
              min_by_view=min_view.tolist(), max_by_view=max_view.tolist(),
              max_abs_regression=max_regression, regression_mismatches=regression_mismatches)
        check("local_scales", bool(np.all(np.isfinite(scales)) and np.all(scales >= -1e-5)),
              min_by_view=np.min(scales, axis=0).tolist(),
              max_by_view=np.max(scales, axis=0).tolist())

        # Symmetry is checked independently on a fixed sample; stored edge values
        # use canonical unordered pairs so reciprocal entries always share one score.
        sampled = np.linspace(0, len(keys) - 1, min(1024, len(keys)), dtype=int)
        symmetry_error = 0.0
        with torch.no_grad():
            for view, view_tensor in enumerate(view_tensors):
                lhs = view_tensor[left[sampled]]
                rhs = view_tensor[right[sampled]]
                forward = 13.0 - torch.sum(torch.abs(torch.sum(torch.conj(lhs) * rhs, dim=-1)) ** 2, dim=-1)
                reverse = 13.0 - torch.sum(torch.abs(torch.sum(torch.conj(rhs) * lhs, dim=-1)) ** 2, dim=-1)
                symmetry_error = max(symmetry_error, float(torch.max(torch.abs(forward - reverse)).item()))
        check("view_symmetry_sample", symmetry_error <= 1e-5,
              tested_pairs=len(sampled), max_abs=symmetry_error)

        q, disagreement, u, source_adp = score_edges(
            edge_adp, scales, left, right, adp, timestamps, metadata["gamma"]
        )
        check("score_finite_and_permutation", bool(
            np.all(np.isfinite(q)) and np.all(np.isfinite(u)) and np.all((u >= 0) & (u <= 1))
            and np.allclose(np.median(np.abs(np.log(q[:, ::-1])
                                             - np.median(np.log(q[:, ::-1]), axis=1)[:, None]), axis=1),
                            np.median(np.abs(np.log(q)
                                             - np.median(np.log(q), axis=1)[:, None]), axis=1))
        ), s95=float(np.quantile(disagreement, 0.95)),
              adp_source_edges=int(np.count_nonzero(source_adp)),
              time_source_edges=int(np.count_nonzero(~source_adp)))
        time_cost = np.abs(timestamps[left] - timestamps[right]).astype(np.float32)
        time_cost *= np.float32(metadata["gamma"])
        fused_expected = np.minimum(np.asarray(adp[left, right]), time_cost)
        fused_actual = np.asarray(fused[left, right])
        directed_time_cost = np.abs(timestamps[rows] - timestamps[cols]).astype(np.float32)
        directed_time_cost *= np.float32(metadata["gamma"])
        directed_source = np.asarray(adp[rows, cols]) <= directed_time_cost
        source_mismatches = int(np.count_nonzero(directed_source != source_adp[pair_index]))
        check("fused_source_and_cost", bool(np.allclose(fused_expected, fused_actual,
                                                       rtol=1e-6, atol=1e-6)
                                            and source_mismatches == 0),
              max_abs=float(np.max(np.abs(fused_expected - fused_actual))),
              source_mismatches=source_mismatches)

        baseline0 = apply_risk_cost(graph, pair_index, u, source_adp, 0.0)
        same_graph = (np.array_equal(graph.indptr, baseline0.indptr)
                      and np.array_equal(graph.indices, baseline0.indices)
                      and np.array_equal(graph.data, baseline0.data))
        check("kappa_zero_exact_graph", same_graph)
        risk_graph = apply_risk_cost(graph, pair_index, u, source_adp, 0.5)
        expected_factor = 1.0 + np.float32(0.5) * u[pair_index] * source_adp[pair_index]
        observed_factor = np.ones(graph.nnz, dtype=np.float64)
        nonzero = graph.data != 0
        observed_factor[nonzero] = risk_graph.data[nonzero] / graph.data[nonzero]
        factor_min = np.full(len(keys), np.inf)
        factor_max = np.full(len(keys), -np.inf)
        np.minimum.at(factor_min, pair_index, observed_factor)
        np.maximum.at(factor_max, pair_index, observed_factor)
        factor_difference = float(np.max(factor_max - factor_min))
        risk_components = int(connected_components(risk_graph, directed=False)[0])
        risk_ok = (np.array_equal(graph.indptr, risk_graph.indptr)
                   and np.array_equal(graph.indices, risk_graph.indices)
                   and np.all(np.isfinite(risk_graph.data))
                   and np.all(risk_graph.data >= graph.data)
                   and np.array_equal(risk_graph.data[~source_adp[pair_index]],
                                      graph.data[~source_adp[pair_index]])
                   and np.allclose(observed_factor[nonzero], expected_factor[nonzero],
                                   rtol=1e-6, atol=1e-6)
                   and factor_difference <= 1e-6 and risk_components == 1)
        check("risk_graph_invariants", bool(risk_ok), components=risk_components,
              max_reciprocal_factor_difference=factor_difference,
              max_baseline_reciprocal_cost_difference=float(np.max(baseline_asym)),
              changed_entries=int(np.count_nonzero(risk_graph.data != graph.data)))

        np.savez_compressed(output / "edge_observations.npz", left=left, right=right,
                            per_view_adp=edge_adp, view_scales=scales, q=q.astype(np.float32),
                            disagreement=disagreement.astype(np.float32), u=u.astype(np.float32),
                            source_adp=source_adp, directed_pair_index=pair_index)
        result["status"] = "confirmed"
        result["elapsed_s"] = time.time() - start_time
        result["artifacts"] = ["edge_risk/edge_observations.npz", "edge_risk/gate0_regression.json"]
        write_result(report_path, result)
        return result
    except Exception as exc:
        result["status"] = "failed"
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["elapsed_s"] = time.time() - start_time
        write_result(report_path, result)
        raise


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument("--data-path", type=Path, default=repo_root / "datasets/adp_old_shard4")
    parser.add_argument("--distance-run", type=Path,
                        default=repo_root / "results/adp_old_fused/stride1_n20997_seed10")
    parser.add_argument("--out-dir", type=Path,
                        default=repo_root / "results/multiview_edge_uncertainty")
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
