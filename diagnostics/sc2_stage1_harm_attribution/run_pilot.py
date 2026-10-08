"""Stage-1 common-neighbor penalty harm attribution diagnostic.

The intervention is frozen from the previous pilot. This script only separates
its old edge-length multiplier into T and R groups; it does not invent a score,
change graph support, or use GT before the label-free grouping is saved.
"""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components, shortest_path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

N_EXPECTED = 20_997
K_EXPECTED = 20
SEED = 10
SOURCES = np.arange(0, N_EXPECTED, 10, dtype=np.int64)
TIME_THRESHOLD_S = 60.0
OUTPUT_ROOT = ROOT / "results/sc2_stage1_harm_attribution"
GRAPH_PATH = ROOT / "results/phase2c_h1_h2/stage1_graph.npz"
DISTANCE_PATH = ROOT / "results/adp_old_fused/stride1_n20997_seed10"
OLD_PILOT = ROOT / "results/sc2_stage1_consensus_pilot"
DATA_PATH = ROOT / "datasets/adp_old_shard4"


def jsonable(value: object) -> object:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(value), indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")


def summarize(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {"count": 0}
    return {
        "count": int(values.size),
        "min": float(np.min(values)),
        "p01": float(np.percentile(values, 1)),
        "p25": float(np.percentile(values, 25)),
        "median": float(np.median(values)),
        "p75": float(np.percentile(values, 75)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
    }


def load_graph() -> sp.csr_matrix:
    saved = np.load(GRAPH_PATH, allow_pickle=True)
    required = {"indices", "indptr", "shape", "data"}
    if not required.issubset(saved.files):
        raise ValueError(f"Stage-1 graph missing {required - set(saved.files)}")
    shape = tuple(int(x) for x in saved["shape"])
    graph = sp.csr_matrix((saved["data"], saved["indices"], saved["indptr"]), shape=shape)
    if shape != (N_EXPECTED, N_EXPECTED):
        raise ValueError(f"unexpected graph shape: {shape}")
    degrees = np.diff(graph.indptr)
    if not np.all(degrees == K_EXPECTED) or np.any(graph.diagonal() != 0):
        raise ValueError("graph is not the verified directed k=20 graph")
    if np.any(graph.data <= 0) or not np.all(np.isfinite(graph.data)):
        raise ValueError("graph has non-positive or non-finite weights")
    if int(connected_components(graph, directed=False, return_labels=False)) != 1:
        raise ValueError("Stage-1 graph is disconnected")
    return graph


def edge_arrays(graph: sp.csr_matrix) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = np.repeat(np.arange(graph.shape[0], dtype=np.int64), np.diff(graph.indptr))
    cols = graph.indices.astype(np.int64, copy=False)
    return rows, cols, graph.data.astype(np.float32, copy=False)


def align_saved_edge_artifact(graph: sp.csr_matrix) -> dict[str, np.ndarray | float | bool]:
    """Align old edge_scores arrays to graph CSR order by (row, col) key."""
    saved = np.load(OLD_PILOT / "edge_scores.npz")
    rows, cols, weights = edge_arrays(graph)
    old_rows = np.asarray(saved["rows"], dtype=np.int64)
    old_cols = np.asarray(saved["cols"], dtype=np.int64)
    if len(old_rows) != graph.nnz or not np.array_equal(np.sort(old_rows), np.sort(rows)):
        raise ValueError("old edge artifact row support does not match graph")
    old_keys = old_rows * N_EXPECTED + old_cols
    graph_keys = rows * N_EXPECTED + cols
    old_order = np.argsort(old_keys, kind="mergesort")
    graph_order = np.argsort(graph_keys, kind="mergesort")
    if not np.array_equal(old_keys[old_order], graph_keys[graph_order]):
        raise ValueError("old edge artifact and graph edge sets differ")
    if not np.array_equal(np.asarray(saved["original_weights"])[old_order], weights[graph_order]):
        raise ValueError("old edge artifact weights differ from graph weights")
    # graph_order[k] is the graph index for sorted key k; old_order[k] is old index.
    graph_to_old = np.empty(graph.nnz, dtype=np.int64)
    graph_to_old[graph_order] = old_order
    result: dict[str, np.ndarray | float | bool] = {
        "rows": rows,
        "cols": cols,
        "original_weights": weights,
        "support": np.asarray(saved["support"], dtype=np.float32)[graph_to_old],
        "treatment_multiplier": np.asarray(saved["treatment_multiplier"], dtype=np.float32)[graph_to_old],
        "shuffled_multiplier": np.asarray(saved["shuffled_multiplier"], dtype=np.float32)[graph_to_old],
        "common_neighbor_count": np.asarray(saved["common_neighbor_count"], dtype=np.int32)[graph_to_old],
        "undirected_degree": np.asarray(saved["undirected_degree"], dtype=np.int32),
        "edge_set_match": True,
        "max_weight_abs_diff": 0.0,
    }
    expected_multiplier = 1.0 + 0.5 * (1.0 - result["support"])
    if not np.allclose(result["treatment_multiplier"], expected_multiplier, atol=1e-6):
        raise ValueError("old treatment multiplier does not match frozen rule")
    return result


def build_label_free_groups(graph: sp.csr_matrix, edge: dict[str, np.ndarray | float | bool]) -> dict[str, object]:
    rows = edge["rows"]
    cols = edge["cols"]
    timestamps = np.load(DATA_PATH / "timestamps.npy").astype(np.float64)
    adp = np.load(DISTANCE_PATH / "adp.npy", mmap_mode="r")
    result_json = json.loads((DISTANCE_PATH / "result.json").read_text(encoding="utf-8")) if (DISTANCE_PATH / "result.json").exists() else json.loads((DISTANCE_PATH / "../result.json").read_text(encoding="utf-8"))
    gamma = float(result_json["gamma"])
    dt = np.abs(timestamps[rows] - timestamps[cols])
    adp_edge = np.asarray(adp[rows, cols], dtype=np.float64)
    adp_branch = adp_edge <= gamma * dt
    t_mask = (dt > TIME_THRESHOLD_S) & adp_branch
    r_mask = ~t_mask
    if np.any(t_mask & r_mask) or not np.all(t_mask | r_mask):
        raise AssertionError("T/R groups are not a partition")
    group = {
        "dt": dt,
        "adp_edge": adp_edge,
        "adp_branch": adp_branch,
        "T_mask": t_mask,
        "R_mask": r_mask,
        "gamma": gamma,
        "timestamps_range_s": [float(timestamps.min()), float(timestamps.max())],
    }
    return group


def variant_graph(graph: sp.csr_matrix, original: np.ndarray, multiplier: np.ndarray, mask: np.ndarray | None) -> sp.csr_matrix:
    if mask is None:
        data = original
    else:
        data = original * np.where(mask, multiplier, 1.0).astype(np.float32)
    variant = sp.csr_matrix((data.astype(np.float32), graph.indices.copy(), graph.indptr.copy()), shape=graph.shape)
    if not np.array_equal(variant.indices, graph.indices) or not np.array_equal(variant.indptr, graph.indptr):
        raise AssertionError("variant changed graph support")
    return variant


def run_source_shortest_path(graph: sp.csr_matrix, variant_name: str) -> np.ndarray:
    t0 = time.time()
    distances = shortest_path(graph, method="auto", directed=False, indices=SOURCES)
    subset = np.asarray(distances[:, SOURCES], dtype=np.float32)
    del distances
    variant_dir = OUTPUT_ROOT / "variants" / variant_name
    variant_dir.mkdir(parents=True, exist_ok=True)
    np.save(variant_dir / "source_geodesic_subset.npy", subset)
    write_json(variant_dir / "shortest_path_summary.json", {"variant": variant_name, "source_count": int(len(SOURCES)), "target_count": N_EXPECTED, "directed": False, "dtype_saved": "float32", "elapsed_s": time.time() - t0, "subset_shape": list(subset.shape)})
    return subset


def evaluate_mds(distance_subset: np.ndarray, name: str, positions: np.ndarray) -> tuple[dict[str, object], np.ndarray]:
    from common_metrics import affine_transform, chart_metrics, localization_error_summary
    from run_adp_fused import run_mds

    t0 = time.time()
    embedding = run_mds(distance_subset, seed=SEED, max_iter=80, n_init=4)
    aligned = affine_transform(positions, embedding)
    errors = np.linalg.norm(aligned - positions, axis=1)
    record = {
        "variant": name,
        "indices_definition": "np.arange(0,20997,10)",
        "num_samples": int(len(SOURCES)),
        "seed": SEED,
        "max_iter": 80,
        "n_init": 4,
        "metrics": chart_metrics(positions, embedding),
        "localization_errors_after_affine": localization_error_summary(positions, embedding),
        "per_sample_error_summary": summarize(errors),
        "elapsed_s": time.time() - t0,
    }
    variant_dir = OUTPUT_ROOT / "variants" / name
    variant_dir.mkdir(parents=True, exist_ok=True)
    np.savez(variant_dir / "mds.npz", indices=SOURCES, positions=positions, embedding=embedding, aligned=aligned, errors=errors)
    write_json(variant_dir / "mds_result.json", record)
    return record, errors


def load_old_mds(name: str, positions: np.ndarray) -> tuple[dict[str, object], np.ndarray]:
    from common_metrics import affine_transform, chart_metrics, localization_error_summary

    data = np.load(OLD_PILOT / name / "mds.npz")
    if not np.array_equal(np.asarray(data["indices"], dtype=np.int64), SOURCES):
        raise ValueError(f"old {name} MDS sample order differs from fixed sources")
    embedding = np.asarray(data["embedding"], dtype=np.float64)
    aligned = affine_transform(positions, embedding)
    errors = np.linalg.norm(aligned - positions, axis=1)
    record = {
        "variant": name,
        "indices_definition": "np.arange(0,20997,10)",
        "num_samples": int(len(SOURCES)),
        "seed": SEED,
        "max_iter": 80,
        "n_init": 4,
        "metrics": chart_metrics(positions, embedding),
        "localization_errors_after_affine": localization_error_summary(positions, embedding),
        "per_sample_error_summary": summarize(errors),
        "source": str((OLD_PILOT / name / "mds.npz").relative_to(ROOT)),
    }
    variant_dir = OUTPUT_ROOT / "variants" / name
    variant_dir.mkdir(parents=True, exist_ok=True)
    np.savez(variant_dir / "mds_reused.npz", indices=SOURCES, positions=positions, embedding=embedding, aligned=aligned, errors=errors)
    write_json(variant_dir / "mds_result_reused.json", record)
    return record, errors


def compare_baseline_reproduction(recomputed: dict[str, object], old_gate: dict[str, object], old_errors: np.ndarray, new_errors: np.ndarray) -> dict[str, object]:
    old = old_gate["variants"]["baseline"]
    metric_deltas = {}
    for key in ("mean_m", "p95_m", "max_m"):
        metric_deltas[key] = float(recomputed["localization_errors_after_affine"][key] - old["localization_errors_after_affine"][key])
    return {
        "old_gate_metrics": old["localization_errors_after_affine"],
        "recomputed_metrics": recomputed["localization_errors_after_affine"],
        "metric_deltas": metric_deltas,
        "max_per_sample_error_abs_delta": float(np.max(np.abs(new_errors - old_errors))),
        "reproduction_ok": bool(max(abs(v) for v in metric_deltas.values()) <= 1e-4 and np.max(np.abs(new_errors - old_errors)) <= 1e-4),
    }


def save_label_free_artifacts(edge: dict[str, np.ndarray | float | bool], group: dict[str, object]) -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUTPUT_ROOT / "edge_groups.npz", rows=edge["rows"], cols=edge["cols"], original_weights=edge["original_weights"], common_neighbor_count=edge["common_neighbor_count"], support=edge["support"], treatment_multiplier=edge["treatment_multiplier"], shuffled_multiplier=edge["shuffled_multiplier"], dt=group["dt"], adp_edge=group["adp_edge"], adp_branch=group["adp_branch"], T_mask=group["T_mask"], R_mask=group["R_mask"])
    write_json(OUTPUT_ROOT / "pre_gt_manifest.json", {"N": N_EXPECTED, "directed_edges": int(len(edge["rows"])), "T_count": int(np.count_nonzero(group["T_mask"])), "R_count": int(np.count_nonzero(group["R_mask"])), "gamma": group["gamma"], "time_threshold_s": TIME_THRESHOLD_S, "gt_loaded": False, "edge_set_match": bool(edge["edge_set_match"]), "edge_weight_max_abs_diff": float(edge["max_weight_abs_diff"]), "sources_definition": "np.arange(0,20997,10)", "shortest_path_directed": False})


def edge_group_summary(edge: dict[str, np.ndarray | float | bool], group: dict[str, object]) -> dict[str, object]:
    original = np.asarray(edge["original_weights"], dtype=np.float64)
    multiplier = np.asarray(edge["treatment_multiplier"], dtype=np.float64)
    result = {}
    for name, mask in (("T", group["T_mask"]), ("R", group["R_mask"])):
        mask = np.asarray(mask, dtype=bool)
        delta = original[mask] * (multiplier[mask] - 1.0)
        result[name] = {"count": int(mask.sum()), "fraction": float(mask.mean()), "original_weight_sum": float(original[mask].sum()), "treatment_weight_sum": float((original[mask] * multiplier[mask]).sum()), "weighted_increment_sum": float(delta.sum()), "weighted_increment_fraction_of_total": None, "multiplier": summarize(multiplier[mask]), "support": summarize(np.asarray(edge["support"])[mask]), "dt_s": summarize(np.asarray(group["dt"])[mask]), "adp": summarize(np.asarray(group["adp_edge"])[mask])}
    total_increment = float(np.sum(original * (multiplier - 1.0)))
    for name in ("T", "R"):
        result[name]["weighted_increment_fraction_of_total"] = float(result[name]["weighted_increment_sum"] / total_increment) if total_increment else None
    result["all"] = {"count": int(len(original)), "original_weight_sum": float(original.sum()), "treatment_weight_sum": float((original * multiplier).sum()), "weighted_increment_sum": total_increment, "multiplier": summarize(multiplier)}
    return result


def fixed_example_errors(error_by_variant: dict[str, np.ndarray]) -> list[dict[str, object]]:
    rows = []
    for sample_id in (4540, 5720):
        index = int(np.flatnonzero(SOURCES == sample_id)[0])
        rows.append({"sample_id": sample_id, **{name: float(errors[index]) for name, errors in error_by_variant.items()}})
    return rows


def gt_path_diagnostic(edge: dict[str, np.ndarray | float | bool], group: dict[str, object], positions: np.ndarray) -> dict[str, object]:
    physical = np.linalg.norm(positions[edge["rows"]] - positions[edge["cols"]], axis=1)
    t_mask = np.asarray(group["T_mask"], dtype=bool)
    support = np.asarray(edge["support"], dtype=np.float64)
    cutoff = float(np.quantile(support[t_mask], 0.25))
    selected = t_mask & (support <= cutoff) & (physical < 0.25)
    base_geo = np.load(OLD_PILOT / "baseline" / "geodesic.npy", mmap_mode="r")
    cons_geo = np.load(OLD_PILOT / "consensus" / "geodesic.npy", mmap_mode="r")
    delta = np.asarray(cons_geo[edge["rows"][selected], edge["cols"][selected]], dtype=np.float64) - np.asarray(base_geo[edge["rows"][selected], edge["cols"][selected]], dtype=np.float64)
    selected_indices = np.flatnonzero(selected)
    order = np.lexsort((edge["cols"][selected_indices], edge["rows"][selected_indices]))
    examples = []
    for index in selected_indices[order[:5]]:
        examples.append({"sample_i": int(edge["rows"][index]), "sample_j": int(edge["cols"][index]), "physical_distance_m": float(physical[index]), "support": float(support[index]), "treatment_multiplier": float(edge["treatment_multiplier"][index]), "baseline_geodesic": float(base_geo[edge["rows"][index], edge["cols"][index]]), "consensus_geodesic": float(cons_geo[edge["rows"][index], edge["cols"][index]]), "delta": float(cons_geo[edge["rows"][index], edge["cols"][index]] - base_geo[edge["rows"][index], edge["cols"][index]])})
    return {"T_low_support_quartile_cutoff": cutoff, "selected_count": int(selected.sum()), "physical_near_definition_m": 0.25, "physical_summary_m": summarize(physical[selected]), "geodesic_delta_consensus_minus_baseline": summarize(delta), "delta_positive_fraction": float(np.mean(delta > 0)) if len(delta) else None, "delta_negative_fraction": float(np.mean(delta < 0)) if len(delta) else None, "examples": examples}


def write_report(result: dict[str, object]) -> None:
    m = result["mds_metrics"]
    a = result["attribution"]
    lines = [
        "# Stage-1 共同邻居边长惩罚：损害归因检查",
        "",
        "这轮实验只回答一个问题：旧的全边加长惩罚造成的 MDS 变差，是否主要来自跨时间且由 ADP 分支选中的连接。没有设计新评分，没有加入天线视角，没有训练 FCF，也没有运行 Stage 2。",
        "",
        "## 怎么分组",
        "",
        f"旧实验的 419,940 条有向连接按固定规则分成两组：T 组是时间差超过 60 秒、且 `ADP <= gamma*时间差` 的连接，共 {result['groups']['T']['count']:,} 条；R 组是其余 {result['groups']['R']['count']:,} 条。两组都保留原图的边，只对对应组使用旧倍率 `1+0.5*(1-support)`。",
        "",
        f"T 组占全部连接 {result['groups']['T']['fraction']:.2%}，但只占本次加权改变量 {result['groups']['T']['weighted_increment_fraction_of_total']:.2%}；这说明 T 组规模较小，不能仅凭“它被改了多少”判断它是不是损害来源。",
        "",
        "旧 edge_scores 文件和原始图的存储顺序不同，本次按 `(row, col)` 边键重新对齐；419,940 条边和权重逐条核对一致。",
        "",
        "## 原图复现检查",
        "",
        f"固定使用 `np.arange(0,20997,10)` 的 2,100 个源。原图重新计算的 source-to-all 最短路抽取矩阵与旧 baseline 测地距离的最大误差为 `{result['reproduction']['geodesic_subset_max_abs_error']:.3g}`；重新运行 baseline MDS 后，逐样本误差相对旧 baseline 的最大差为 `{result['reproduction']['max_per_sample_error_abs_delta']:.3g}`。复现判定：**{'通过' if result['reproduction']['reproduction_ok'] else '失败'}**。",
        "",
        "## 四组 MDS 结果",
        "",
        "| 组别 | 平均误差(m) | P95(m) | CT | TW | KS | 相对 baseline 改善/恶化样本数 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ("baseline", "T_only", "R_only", "consensus"):
        item = m[name]
        lines.append(f"| {name} | {item['mean_m']:.6f} | {item['p95_m']:.6f} | {item['ct']:.6f} | {item['tw']:.6f} | {item['ks']:.6f} | +{item['improved_count_vs_baseline']} / -{item['harmed_count_vs_baseline']} |")
    lines += [
        "",
        f"旧全边惩罚相对 baseline 的平均误差增量为 `{a['full_mean_increment_m']:.6f} m`。T_only 的增量为 `{a['T_only_mean_increment_m']:.6f} m`，R_only 的增量为 `{a['R_only_mean_increment_m']:.6f} m`。T_only 是否达到全惩罚增量一半且比 R_only 至少多 0.01 m：`{a['mean_attribution_condition']}`。P95 是否与全惩罚同方向变差：`{a['p95_same_direction_condition']}`。",
        "",
        f"按事前标准，本轮归因判定为：**{'支持 T 组是主要损害来源的证据' if a['T_is_major_harm_source_evidence'] else '不支持把 T 组列为主要损害来源'}**。这是类别层面的归因证据，不是对每一条 T 连接的精确因果结论。",
        "",
        "固定样本 4540 和 5720 的误差如下。单个样本的变化不能归因到某一条边，只能作为局部现象示例。",
        "",
        "| 样本 | baseline | T_only | R_only | consensus |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in result["fixed_examples"]:
        lines.append(f"| {row['sample_id']} | {row['baseline']:.6f} | {row['T_only']:.6f} | {row['R_only']:.6f} | {row['consensus']:.6f} |")
    d = result["gt_path_diagnostic"]
    lines += [
        "",
        "## T 组低支持且物理上很近的边：事后路径检查",
        "",
        f"T 组内支持分数最低四分之一的阈值为 `{d['T_low_support_quartile_cutoff']:.6f}`；其中真实距离小于 0.25 m 的连接有 {d['selected_count']:,} 条。全惩罚图相对原图的端点 D_geo 变化中位数为 `{d['geodesic_delta_consensus_minus_baseline']['median']:.6f}`（D_geo 单位），增加的比例为 `{d['delta_positive_fraction']:.2%}`，减少的比例为 `{d['delta_negative_fraction']:.2%}`。因为本次旧干预只会加长边，D_geo 增大是机械上预期的结果；这里不把 100% 增大本身当作独立的损害证据，只列出低支持且物理上很近的受影响关系。真实位置只用于这项事后检查。",
        "",
        "| i | j | 真实距离(m) | support | 旧倍率 | baseline D_geo | consensus D_geo | 变化(D_geo单位) |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in d["examples"]:
        lines.append(f"| {row['sample_i']} | {row['sample_j']} | {row['physical_distance_m']:.4f} | {row['support']:.4f} | {row['treatment_multiplier']:.4f} | {row['baseline_geodesic']:.4f} | {row['consensus_geodesic']:.4f} | {row['delta']:.4f} |")
    lines += [
        "",
        "## 结论边界",
        "",
        "这轮只判断旧惩罚的损害来源，不能证明新的关系评分有效，也不能直接推出应该保留或删除 T 组连接。T/R 作用可能有交互，因此两个单独变体的误差增量不能被解释成精确的因果贡献率。T 组定义由先前同一数据集的事后观察启发，因此这不是独立数据复现。GT 只用于评价和事后样本检查。",
        "",
        "机器文件：`pre_gt_manifest.json`、`edge_groups.npz`、`group_summary.json`、`mds_metrics.json`、`attribution.json`、`fixed_examples.json`、`gt_path_diagnostic.json`，以及 `variants/` 下的源距离子矩阵和 MDS 结果。",
    ]
    (OUTPUT_ROOT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    graph = load_graph()
    edge = align_saved_edge_artifact(graph)
    group = build_label_free_groups(graph, edge)
    save_label_free_artifacts(edge, group)
    summaries = edge_group_summary(edge, group)
    write_json(OUTPUT_ROOT / "group_summary.json", summaries)

    # Build and solve all label-free graph variants before reading GT.
    base_graph = variant_graph(graph, edge["original_weights"], edge["treatment_multiplier"], None)
    t_graph = variant_graph(graph, edge["original_weights"], edge["treatment_multiplier"], group["T_mask"])
    r_graph = variant_graph(graph, edge["original_weights"], edge["treatment_multiplier"], group["R_mask"])

    base_subset = run_source_shortest_path(base_graph, "baseline_recomputed")
    old_base_geo = np.load(OLD_PILOT / "baseline" / "geodesic.npy", mmap_mode="r")
    old_base_subset = np.asarray(old_base_geo[np.ix_(SOURCES, SOURCES)], dtype=np.float32)
    geodesic_subset_max_abs_error = float(np.max(np.abs(base_subset - old_base_subset)))
    if geodesic_subset_max_abs_error > 1e-4:
        raise RuntimeError(f"baseline source geodesic reproduction failed: {geodesic_subset_max_abs_error}")

    # GT is loaded only after graph construction and baseline source-path reproduction.
    positions = np.asarray(np.load(DISTANCE_PATH / "distance_data.npz")["positions"], dtype=np.float32)
    if positions.shape != (N_EXPECTED, 2):
        raise ValueError(f"unexpected evaluation position shape: {positions.shape}")
    mds_positions = positions[SOURCES]

    old_gate = json.loads((OLD_PILOT / "mds_gate.json").read_text(encoding="utf-8"))
    base_record, base_errors = evaluate_mds(base_subset, "baseline_recomputed", mds_positions)
    old_base_record, old_base_errors = load_old_mds("baseline", mds_positions)
    reproduction = compare_baseline_reproduction(base_record, old_gate, old_base_errors, base_errors)
    reproduction["geodesic_subset_max_abs_error"] = geodesic_subset_max_abs_error
    write_json(OUTPUT_ROOT / "reproduction_check.json", reproduction)
    if not reproduction["reproduction_ok"]:
        raise RuntimeError("baseline MDS reproduction failed; stopped before T/R variants")

    t_subset = run_source_shortest_path(t_graph, "T_only")
    r_subset = run_source_shortest_path(r_graph, "R_only")
    t_record, t_errors = evaluate_mds(t_subset, "T_only", mds_positions)
    r_record, r_errors = evaluate_mds(r_subset, "R_only", mds_positions)
    consensus_record, consensus_errors = load_old_mds("consensus", mds_positions)
    variant_records = {"baseline": base_record, "T_only": t_record, "R_only": r_record, "consensus": consensus_record}
    variant_errors = {"baseline": base_errors, "T_only": t_errors, "R_only": r_errors, "consensus": consensus_errors}
    mds_metrics = {}
    baseline_mean = base_record["localization_errors_after_affine"]["mean_m"]
    for name in ("baseline", "T_only", "R_only", "consensus"):
        rec = variant_records[name]
        errors = variant_errors[name]
        mds_metrics[name] = {"mean_m": rec["localization_errors_after_affine"]["mean_m"], "p95_m": rec["localization_errors_after_affine"]["p95_m"], "ct": rec["metrics"]["ct"], "tw": rec["metrics"]["tw"], "ks": rec["metrics"]["ks"], "improved_count_vs_baseline": int(np.count_nonzero(base_errors - errors > 0)), "harmed_count_vs_baseline": int(np.count_nonzero(base_errors - errors < 0))}
    write_json(OUTPUT_ROOT / "mds_metrics.json", mds_metrics)
    error_by_variant = {"baseline": base_errors, "T_only": t_errors, "R_only": r_errors, "consensus": consensus_errors}
    fixed_examples = fixed_example_errors(error_by_variant)
    write_json(OUTPUT_ROOT / "fixed_examples.json", fixed_examples)

    full_mean_increment = float(mds_metrics["consensus"]["mean_m"] - mds_metrics["baseline"]["mean_m"])
    t_increment = float(mds_metrics["T_only"]["mean_m"] - mds_metrics["baseline"]["mean_m"])
    r_increment = float(mds_metrics["R_only"]["mean_m"] - mds_metrics["baseline"]["mean_m"])
    full_p95_increment = float(mds_metrics["consensus"]["p95_m"] - mds_metrics["baseline"]["p95_m"])
    t_p95_increment = float(mds_metrics["T_only"]["p95_m"] - mds_metrics["baseline"]["p95_m"])
    p95_same_direction = bool(full_p95_increment > 0 and t_p95_increment > 0)
    attribution = {"full_mean_increment_m": full_mean_increment, "T_only_mean_increment_m": t_increment, "R_only_mean_increment_m": r_increment, "full_p95_increment_m": full_p95_increment, "T_only_p95_increment_m": t_p95_increment, "R_only_p95_increment_m": float(mds_metrics["R_only"]["p95_m"] - mds_metrics["baseline"]["p95_m"]), "mean_attribution_condition": bool(full_mean_increment > 0 and t_increment >= 0.5 * full_mean_increment and t_increment >= r_increment + 0.01), "p95_same_direction_condition": p95_same_direction}
    attribution["T_is_major_harm_source_evidence"] = bool(attribution["mean_attribution_condition"] and attribution["p95_same_direction_condition"])
    write_json(OUTPUT_ROOT / "attribution.json", attribution)

    gt_diag = gt_path_diagnostic(edge, group, positions)
    write_json(OUTPUT_ROOT / "gt_path_diagnostic.json", gt_diag)
    result = {"groups": summaries, "reproduction": reproduction, "mds_metrics": mds_metrics, "attribution": attribution, "fixed_examples": fixed_examples, "gt_path_diagnostic": gt_diag}
    write_json(OUTPUT_ROOT / "result.json", result)
    write_report(result)
    print(json.dumps({"reproduction": reproduction, "mds_metrics": mds_metrics, "attribution": attribution}, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
