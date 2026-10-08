"""Frozen Stage-1 relation-only multi-view consistency diagnostic.

This script never modifies the graph and never uses UE positions until all
candidate relations, scores, strata, and shuffle controls have been fixed.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

N_EXPECTED = 20_997
K_EXPECTED = 20
SEED = 10
BOOTSTRAP_SEED = 11
TIME_GAP_S = 60.0
NEAR_M = 0.25
FAR_M = 0.5
ARRAY_A = (0, 2)
ARRAY_B = (1, 3)

DISTANCE_DIR = ROOT / "results/adp_old_fused/stride1_n20997_seed10"
GRAPH_PATH = ROOT / "results/phase2c_h1_h2/stage1_graph.npz"
DATA_DIR = ROOT / "datasets/adp_old_shard4"
OUTPUT_DIR = ROOT / "results/sc2_stage1_multiview_check"


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


def rank_bins(values: np.ndarray, bins: int) -> np.ndarray:
    """Assign deterministic equal-count rank bins without using labels."""
    if bins < 1 or len(values) < bins:
        raise ValueError("rank_bins requires at least one item per bin")
    order = np.argsort(np.asarray(values), kind="mergesort")
    result = np.empty(len(values), dtype=np.int16)
    for label, chunk in enumerate(np.array_split(order, bins)):
        result[chunk] = label
    return result


def rank_score(values: np.ndarray, higher_is_better: bool = True) -> np.ndarray:
    order = np.argsort(-values if higher_is_better else values, kind="mergesort")
    result = np.empty(len(values), dtype=np.float64)
    # Larger returned scores always mean better: high input value gets high rank.
    result[order] = np.arange(len(values) - 1, -1, -1, dtype=np.float64)
    return result


def load_graph_and_candidates() -> dict[str, object]:
    saved = np.load(GRAPH_PATH, allow_pickle=True)
    shape = tuple(int(x) for x in saved["shape"])
    if shape != (N_EXPECTED, N_EXPECTED):
        raise ValueError(f"unexpected Stage-1 graph shape: {shape}")
    import scipy.sparse as sp

    graph = sp.csr_matrix((saved["data"], saved["indices"], saved["indptr"]), shape=shape)
    graph.sum_duplicates()
    degrees = np.diff(graph.indptr).astype(np.int64)
    if not np.all(degrees == K_EXPECTED) or graph.diagonal().any():
        raise ValueError("saved Stage-1 graph is not the verified directed k=20 graph")

    fused = np.load(DISTANCE_DIR / "fused.npy", mmap_mode="r")
    adp = np.load(DISTANCE_DIR / "adp.npy", mmap_mode="r")
    if fused.shape != shape or adp.shape != shape:
        raise ValueError("saved distance matrices do not match the graph")
    rows = np.repeat(np.arange(N_EXPECTED, dtype=np.int64), degrees)
    cols = graph.indices.astype(np.int64, copy=False)
    if not np.array_equal(graph.data, np.asarray(fused[rows, cols])):
        raise ValueError("Stage-1 graph weights do not equal saved fused distances")

    distance_data = np.load(DISTANCE_DIR / "distance_data.npz")
    indices = np.asarray(distance_data["indices"])
    if not np.array_equal(indices, np.arange(N_EXPECTED, dtype=indices.dtype)):
        raise ValueError("saved distance sample order is not stride-1 arange(N)")
    timestamps = np.load(DATA_DIR / "timestamps.npy").astype(np.float64)
    if timestamps.shape != (N_EXPECTED,):
        raise ValueError("unexpected timestamp shape")
    gamma = float(json.loads((DISTANCE_DIR / "result.json").read_text(encoding="utf-8"))["gamma"])

    dt_directed = np.abs(timestamps[rows] - timestamps[cols])
    adp_directed = np.asarray(adp[rows, cols], dtype=np.float64)
    branch_directed = adp_directed <= gamma * dt_directed
    lower = np.minimum(rows, cols)
    upper = np.maximum(rows, cols)
    keys = lower.astype(np.int64) * N_EXPECTED + upper
    _, first = np.unique(keys, return_index=True)
    keep = (dt_directed[first] > TIME_GAP_S) & branch_directed[first]
    selected = first[keep]
    left = lower[selected]
    right = upper[selected]
    d_full = adp_directed[selected]
    dt = dt_directed[selected]

    endpoint_density = np.bincount(rows, weights=np.asarray(graph.data, dtype=np.float64), minlength=N_EXPECTED) / K_EXPECTED
    pair_density = 0.5 * (endpoint_density[left] + endpoint_density[right])
    full_bin = rank_bins(d_full, 10)
    time_bin = rank_bins(dt, 4)
    density_bin = rank_bins(pair_density, 3)
    strata = full_bin.astype(np.int16) * 12 + time_bin.astype(np.int16) * 3 + density_bin.astype(np.int16)

    return {
        "graph": graph,
        "rows": rows,
        "cols": cols,
        "adp": adp,
        "fused": fused,
        "timestamps": timestamps,
        "gamma": gamma,
        "endpoint_density": endpoint_density,
        "left": left,
        "right": right,
        "d_full": d_full,
        "dt": dt,
        "pair_density": pair_density,
        "full_bin": full_bin,
        "time_bin": time_bin,
        "density_bin": density_bin,
        "strata": strata,
        "directed_count": int(len(rows)),
        "directed_qualifying_count": int(np.count_nonzero((dt_directed > TIME_GAP_S) & branch_directed)),
        "unique_undirected_count": int(len(np.unique(keys))),
    }


def normalized_components(h: np.ndarray) -> np.ndarray:
    if h.ndim != 4 or h.shape[1:] != (4, 8, 13):
        raise ValueError(f"expected H shape (N,4,8,13), got {h.shape}")
    z = np.transpose(h, (0, 1, 3, 2)).reshape(h.shape[0], 52, 8)
    norms = np.linalg.norm(z, axis=-1, keepdims=True)
    return (z / np.maximum(norms, 1e-12)).astype(np.complex64)


def view_adp(z: np.ndarray, left: np.ndarray, right: np.ndarray, arrays: tuple[int, ...]) -> np.ndarray:
    component_ids = np.concatenate([np.arange(array * 13, (array + 1) * 13) for array in arrays])
    zl = z[left][:, component_ids, :]
    zr = z[right][:, component_ids, :]
    dots = np.einsum("ecm,ecm->ec", np.conj(zl), zr, optimize=True)
    return (len(component_ids) - np.sum(np.abs(dots) ** 2, axis=1)).astype(np.float64)


def shuffle_within_strata(values: np.ndarray, strata: np.ndarray) -> np.ndarray:
    shuffled = np.empty_like(values)
    rng = np.random.default_rng(SEED)
    for label in np.unique(strata):
        members = np.flatnonzero(strata == label)
        shuffled[members] = values[rng.permutation(members)]
    return shuffled


def fixed_scores(candidate: dict[str, object]) -> dict[str, object]:
    h_flat = np.load(DATA_DIR / "H.npy", mmap_mode="r")
    h = np.asarray(h_flat).reshape(N_EXPECTED, 4, 8, 13)
    z = normalized_components(h)
    left = candidate["left"]
    right = candidate["right"]
    d_a = view_adp(z, left, right, ARRAY_A)
    d_b = view_adp(z, left, right, ARRAY_B)
    d_full = candidate["d_full"]
    decomposition_error = d_a + d_b - d_full
    agreement = 1.0 - np.abs(d_a - d_b) / np.maximum(d_full, 1e-12)
    agreement_shuffled = shuffle_within_strata(agreement, candidate["strata"])
    full_score = -d_full
    combined_score = rank_score(full_score) + rank_score(agreement)
    shuffled_combined_score = rank_score(full_score) + rank_score(agreement_shuffled)
    return {
        "d_a": d_a,
        "d_b": d_b,
        "decomposition_error": decomposition_error,
        "agreement": agreement,
        "agreement_shuffled": agreement_shuffled,
        "full_score": full_score,
        "combined_score": combined_score,
        "shuffled_combined_score": shuffled_combined_score,
    }


def auc_unambiguous(score: np.ndarray, physical: np.ndarray) -> tuple[float | None, int]:
    mask = (physical < NEAR_M) | (physical > FAR_M)
    labels = (physical[mask] < NEAR_M).astype(np.int8)
    if labels.min(initial=0) == labels.max(initial=0):
        return None, int(mask.sum())
    return float(roc_auc_score(labels, score[mask])), int(mask.sum())


def precision_at_top(score: np.ndarray, physical: np.ndarray, fraction: float = 0.05) -> dict[str, float | int]:
    k = max(1, int(np.ceil(len(score) * fraction)))
    chosen = np.argsort(-score, kind="mergesort")[:k]
    return {
        "k": int(k),
        "coverage_fraction": float(k / len(score)),
        "near_fraction": float(np.mean(physical[chosen] < NEAR_M)),
        "far_fraction": float(np.mean(physical[chosen] > FAR_M)),
        "middle_fraction": float(np.mean((physical[chosen] >= NEAR_M) & (physical[chosen] <= FAR_M))),
    }


def stratified_agreement_groups(agreement: np.ndarray, strata: np.ndarray, min_group: int = 30) -> tuple[np.ndarray, np.ndarray, list[dict[str, object]]]:
    top: list[np.ndarray] = []
    bottom: list[np.ndarray] = []
    records: list[dict[str, object]] = []
    for label in np.unique(strata):
        members = np.flatnonzero(strata == label)
        if len(members) < 4:
            records.append({"stratum": int(label), "count": int(len(members)), "bottom_count": 0, "top_count": 0, "eligible": False})
            continue
        quartile = rank_bins(agreement[members], 4)
        bot = members[quartile == 0]
        hi = members[quartile == 3]
        eligible = len(bot) >= min_group and len(hi) >= min_group
        records.append({"stratum": int(label), "count": int(len(members)), "bottom_count": int(len(bot)), "top_count": int(len(hi)), "eligible": bool(eligible)})
        if eligible:
            top.append(hi)
            bottom.append(bot)
    if not top:
        raise RuntimeError("no stratified agreement group met the fixed minimum size")
    return np.concatenate(top), np.concatenate(bottom), records


def endpoint_bootstrap_difference(top: np.ndarray, bottom: np.ndarray, left: np.ndarray, right: np.ndarray, near: np.ndarray, n_nodes: int) -> dict[str, object]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    diffs: list[float] = []
    for _ in range(200):
        sampled_nodes = rng.integers(0, n_nodes, size=n_nodes)
        counts = np.bincount(sampled_nodes, minlength=n_nodes).astype(np.float64)
        top_weights = counts[left[top]] * counts[right[top]]
        bottom_weights = counts[left[bottom]] * counts[right[bottom]]
        if top_weights.sum() <= 0 or bottom_weights.sum() <= 0:
            continue
        top_rate = float(np.sum(top_weights * near[top]) / top_weights.sum())
        bottom_rate = float(np.sum(bottom_weights * near[bottom]) / bottom_weights.sum())
        diffs.append(top_rate - bottom_rate)
    if not diffs:
        raise RuntimeError("all endpoint bootstrap replicates had zero group mass")
    return {
        "replicates_requested": 200,
        "replicates_valid": len(diffs),
        "seed": BOOTSTRAP_SEED,
        "mean_difference": float(np.mean(diffs)),
        "ci95_low": float(np.percentile(diffs, 2.5)),
        "ci95_high": float(np.percentile(diffs, 97.5)),
    }


def make_examples(scores: dict[str, object], candidate: dict[str, object], physical: np.ndarray) -> list[dict[str, object]]:
    left = candidate["left"]
    right = candidate["right"]
    agreement = scores["agreement"]
    combined = scores["combined_score"]
    rows: list[dict[str, object]] = []
    used: set[tuple[int, int]] = set()
    groups = (("high_agreement_near", physical < NEAR_M, agreement), ("high_agreement_far", physical > FAR_M, agreement), ("combined_near", physical < NEAR_M, combined), ("combined_far", physical > FAR_M, combined))
    for label, mask, score in groups:
        members = np.flatnonzero(mask)
        order = members[np.argsort(-score[members], kind="mergesort")]
        added = 0
        for idx in order:
            key = (int(left[idx]), int(right[idx]))
            if key in used:
                continue
            used.add(key)
            rows.append({"kind": label, "sample_i": key[0], "sample_j": key[1], "physical_distance_m": float(physical[idx]), "d_full": float(candidate["d_full"][idx]), "d_A": float(scores["d_a"][idx]), "d_B": float(scores["d_b"][idx]), "agreement": float(agreement[idx]), "time_gap_s": float(candidate["dt"][idx]), "stratum": int(candidate["strata"][idx])})
            added += 1
            if added == 5:
                break
    return rows


def write_report(summary: dict[str, object]) -> None:
    out = OUTPUT_DIR / "REPORT.md"
    e = summary["endpoints"]
    top_k = e["combined"]["top5_precision"]["k"]
    extra_near = round((e["combined"]["top5_precision"]["near_fraction"] - e["shuffled_combined"]["top5_precision"]["near_fraction"]) * top_k)
    lines = [
        "# Stage-1 两视角数值一致性诊断",
        "",
        "这是一项只检查已有关系是否可靠的诊断。没有改 Stage-1 图、没有重算 D_geo、没有训练 FCF，也没有运行 Stage 2。",
        "本轮实际计算的是两组天线给出的距离数值是否接近。它没有直接检验“两个视角是否都认为这对测量相近”，也没有实现 SC²-PCR 的第三方共同支持公式。后两项问题仍未回答。",
        "",
        "## 要解决的问题",
        "",
        "上一轮把共同邻居少的边拉长后效果变差。本轮取原始 k=20 图中 ADP 主导且相隔超过 60 秒的关系，试验“两组天线距离越接近，关系越可信”这一具体代理。但两个距离都大时也会得到高一致性分数，因此这个代理偏离了原先的“双方都认为相近”。",
        "",
        f"候选共 **{summary['candidate']['count']:,} 对**。其中真近邻 {summary['label_prevalence']['near_count']:,} 对（{summary['label_prevalence']['near_fraction']:.1%}），明显假近邻 {summary['label_prevalence']['far_count']:,} 对（{summary['label_prevalence']['far_fraction']:.1%}），其余在中间区间。两个视角固定为 A={ARRAY_A} 和 B={ARRAY_B}；full ADP 是两者之和。视角一致性分数是 `1-|d_A-d_B|/d_full`，并在固定 full ADP、时间差和端点局部密度后比较。",
        "",
        "## 结果怎么读",
        "",
        f"视角分解误差最大绝对值为 `{summary['score_check']['decomposition_max_abs']:.3g}`。这说明本轮确实把同一个 full ADP 拆成两个视角来检查，没有另造一个距离定义。",
        "",
        f"满足每个分层两端至少 30 对的分层共覆盖 {summary['stratified']['eligible_stratum_candidate_count']:,}/{summary['candidate']['count']:,} 对（{summary['stratified']['eligible_stratum_candidate_fraction']:.1%}）；实际用于最高/最低一致性比较的极端四分位关系共 {summary['stratified']['extreme_group_candidate_count']:,} 对。一致性最高组相对最低组的真近邻率差为 `{summary['stratified']['top_minus_bottom_near_fraction']:.4f}`；端点聚类 bootstrap 95% CI 为 `[{summary['stratified']['bootstrap']['ci95_low']:.4f}, {summary['stratified']['bootstrap']['ci95_high']:.4f}]`。",
        "",
        "ROC AUC 只在距离 `<0.25 m` 的真近邻和 `>0.5 m` 的明显假近邻上计算，中间距离不参与 AUC。Top-5% 精确率则以全部候选为分母，中间距离算作未命中。",
        "",
        "| 排序 | AUC（近/远明确样本） | Top-5% 真近邻率 | Top-5% 明显假近邻率 |",
        "|---|---:|---:|---:|",
    ]
    for name, label in (("full_adp", "原始 full ADP"), ("agreement", "两视角一致性"), ("combined", "原始 ADP + 一致性"), ("shuffled_combined", "打乱一致性对照")):
        auc = e[name]["auc"]
        precision = e[name]["top5_precision"]
        lines.append(f"| {label} | {auc:.4f} | {precision['near_fraction']:.4f} | {precision['far_fraction']:.4f} |" if auc is not None else f"| {label} | 不可计算 | {precision['near_fraction']:.4f} | {precision['far_fraction']:.4f} |")
    lines += [
        "",
        f"下面的五等分和相关系数是固定验收完成后的描述性分析，不改变验收标准。其结果是：一致性分数与真实距离的 Spearman 相关为 {summary['score_check']['agreement_spearman_physical']:.4f}；与 full ADP 的相关为 {summary['score_check']['agreement_spearman_full_adp']:.4f}。全局五等分如下，分数越高并没有对应更近的位置：",
        "",
        "| 一致性五等分（低→高） | 样本数 | full ADP 中位数 | 真实距离中位数(m) | 真近邻率 | 明显假近邻率 |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for q in summary["agreement_quintiles"]:
        lines.append(f"| {q['quintile']} | {q['count']} | {q['d_full_median']:.3f} | {q['physical_median_m']:.3f} | {q['near_fraction']:.4f} | {q['far_fraction']:.4f} |")
    lines += [
        "",
        f"组合 Top-5% 真近邻率为 {summary['endpoints']['combined']['top5_precision']['near_fraction']:.2%}，打乱对照为 {summary['endpoints']['shuffled_combined']['top5_precision']['near_fraction']:.2%}，只相差 {(summary['endpoints']['combined']['top5_precision']['near_fraction'] - summary['endpoints']['shuffled_combined']['top5_precision']['near_fraction']):.2%}；组合 AUC 为 {summary['endpoints']['combined']['auc']:.3f}，低于 full ADP 的 {summary['endpoints']['full_adp']['auc']:.3f}。",
        f"在排在最前面的 {top_k:,} 对里，加入一致性比打乱分数只多找对 {extra_near} 对；但看全部明确近/远的关系，组合排序反而明显弱于原始 ADP。因此不能把最前面一小段的改善解释为多视角评分有效。",
        "",
        f"预先固定的验收结果：一致性分层差值至少 0.05 且 CI 下界>0、组合 AUC 同时比 full ADP 和打乱对照高至少 0.02、组合 Top-5% 真近邻率不低于 full ADP。程序判定：**{'通过' if summary['decision']['supported'] else '未通过'}**。",
        "",
        "## 结论和下一步",
        "",
        f"{summary['decision']['plain_conclusion']}",
        "下一步先回到上一轮共同邻居边长实验：已知道哪些连接被加长、它们的真实近远及哪些 MDS 样本改善或恶化；尚不知道哪类连接的改动造成主要损害，也没有证据证明天线视角能解决该损害。应先用固定连接类别的反事实对照和最短路检查补上这一环，再决定是否需要多天线核验。真实位置只用于离线诊断。",
        "",
        "GT 只在候选、分数、分层和打乱对照固定后用于离线标签、统计和样本检查；本结果是同一数据集上的传导式关系诊断，不是未见样本或部署验证。",
        "",
        "机器文件：`config.json`、`candidate_scores.npz`、`evaluation.npz`、`summary.json`、`agreement_quintiles.csv`、`examples.csv`。",
    ]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    candidate = load_graph_and_candidates()
    scores = fixed_scores(candidate)
    decomposition = np.asarray(scores["decomposition_error"])
    if float(np.max(np.abs(decomposition))) > 2e-3:
        raise ValueError("two fixed views do not reproduce full ADP within tolerance")

    # Save all relation variables before reading UE positions. This is an explicit
    # audit boundary: no GT can influence candidate creation or score construction.
    np.savez_compressed(OUTPUT_DIR / "candidate_scores.npz", left=candidate["left"], right=candidate["right"], d_full=candidate["d_full"], d_A=scores["d_a"], d_B=scores["d_b"], dt=candidate["dt"], pair_density=candidate["pair_density"], full_bin=candidate["full_bin"], time_bin=candidate["time_bin"], density_bin=candidate["density_bin"], strata=candidate["strata"], agreement=scores["agreement"], agreement_shuffled=scores["agreement_shuffled"], combined_score=scores["combined_score"], shuffled_combined_score=scores["shuffled_combined_score"])
    write_json(OUTPUT_DIR / "config.json", {"seed": SEED, "bootstrap_seed": BOOTSTRAP_SEED, "time_gap_s": TIME_GAP_S, "near_m": NEAR_M, "far_m": FAR_M, "arrays_A": ARRAY_A, "arrays_B": ARRAY_B, "candidate_definition": "unique unordered original Stage-1 k20 relation; abs(dt)>60 s; saved ADP <= saved gamma*abs(dt)", "gamma": candidate["gamma"], "stratification": "full ADP decile x time-gap quartile x endpoint-density tertile", "minimum_group_count": 30, "auc_population": "physical <0.25 m versus physical >0.5 m; middle excluded", "top5_population": "all candidates; only physical <0.25 m is a hit", "gt_read_after_score_artifact": True})
    write_json(OUTPUT_DIR / "pre_gt_check.json", {"candidate_count": int(len(candidate["left"])), "directed_graph_edges": candidate["directed_count"], "directed_qualifying_count": candidate["directed_qualifying_count"], "unique_undirected_count": candidate["unique_undirected_count"], "gamma": candidate["gamma"], "decomposition_max_abs": float(np.max(np.abs(decomposition))), "decomposition_mean_abs": float(np.mean(np.abs(decomposition))), "agreement_quantiles": np.quantile(scores["agreement"], [0, .1, .5, .9, 1]).tolist()})

    # Fix the strata and top/bottom relation groups before reading GT.
    top, bottom, stratum_records = stratified_agreement_groups(scores["agreement"], candidate["strata"], 30)

    # GT enters only after the pre-GT files and all label-free decisions exist.
    positions = np.load(DATA_DIR / "UE_pos.npy").astype(np.float64)[:, :2]
    if positions.shape != (N_EXPECTED, 2):
        raise ValueError("unexpected UE position shape")
    physical = np.linalg.norm(positions[candidate["left"]] - positions[candidate["right"]], axis=1)
    near = physical < NEAR_M
    far = physical > FAR_M
    top_rate = float(np.mean(near[top]))
    bottom_rate = float(np.mean(near[bottom]))
    bootstrap = endpoint_bootstrap_difference(top, bottom, candidate["left"], candidate["right"], near, N_EXPECTED)


    agreement_edges = np.quantile(scores["agreement"], np.linspace(0.0, 1.0, 6))
    agreement_quintiles: list[dict[str, float | int]] = []
    for q in range(5):
        mask = (scores["agreement"] >= agreement_edges[q]) & ((scores["agreement"] <= agreement_edges[q + 1]) if q == 4 else (scores["agreement"] < agreement_edges[q + 1]))
        agreement_quintiles.append({"quintile": q + 1, "count": int(mask.sum()), "agreement_median": float(np.median(scores["agreement"][mask])), "d_full_median": float(np.median(candidate["d_full"][mask])), "physical_median_m": float(np.median(physical[mask])), "near_fraction": float(np.mean(near[mask])), "far_fraction": float(np.mean(far[mask]))})
    agreement_spearman_physical = float(spearmanr(scores["agreement"], physical).statistic)
    agreement_spearman_full = float(spearmanr(scores["agreement"], candidate["d_full"]).statistic)
    aucs: dict[str, float | None] = {}
    precisions: dict[str, dict[str, float | int]] = {}
    for name, score in (("full_adp", scores["full_score"]), ("agreement", scores["agreement"]), ("combined", scores["combined_score"]), ("shuffled_combined", scores["shuffled_combined_score"])):
        aucs[name], _ = auc_unambiguous(score, physical)
        precisions[name] = precision_at_top(score, physical)

    combined_auc = aucs["combined"]
    full_auc = aucs["full_adp"]
    shuffled_auc = aucs["shuffled_combined"]
    criteria = {"stratified_difference": top_rate - bottom_rate >= 0.05 and bootstrap["ci95_low"] > 0.0, "combined_auc_vs_full": combined_auc is not None and full_auc is not None and combined_auc - full_auc >= 0.02, "combined_auc_vs_shuffle": combined_auc is not None and shuffled_auc is not None and combined_auc - shuffled_auc >= 0.02, "top5_guard": precisions["combined"]["near_fraction"] >= precisions["full_adp"]["near_fraction"]}
    supported = bool(all(criteria.values()))
    if supported:
        conclusion = "在预先固定的关系诊断标准下，两视角距离数值接近这一评分提供了超出原始 full ADP 的关系判别信息；这仍未证明两个视角都认为关系相近或 SC² 第三方支持有效。"
    else:
        conclusion = "在预先固定的关系诊断标准下，两组距离数值接近这一评分失败，因此不把它接入 Stage 1 或 Stage 2。这不能被写成双方都认为相近或 SC² 第三方支持的失败；原先的问题仍未被有效检验。"
    summary = {
        "status": "diagnostic_only",
        "candidate": {"count": int(len(candidate["left"])), "directed_graph_edges": candidate["directed_count"], "directed_qualifying_count": candidate["directed_qualifying_count"], "unique_undirected_count": candidate["unique_undirected_count"], "dt_s": {"min": float(candidate["dt"].min()), "median": float(np.median(candidate["dt"])), "max": float(candidate["dt"].max())}, "d_full": {"min": float(candidate["d_full"].min()), "median": float(np.median(candidate["d_full"])), "max": float(candidate["d_full"].max())}},
        "score_check": {"decomposition_max_abs": float(np.max(np.abs(decomposition))), "decomposition_mean_abs": float(np.mean(np.abs(decomposition))), "agreement_quantiles": np.quantile(scores["agreement"], [0, .1, .5, .9, 1]).tolist(), "agreement_spearman_physical": agreement_spearman_physical, "agreement_spearman_full_adp": agreement_spearman_full},
        "agreement_quintiles": agreement_quintiles,
        "stratified": {"eligible_strata": int(sum(r["eligible"] for r in stratum_records)), "total_strata": int(len(stratum_records)), "eligible_stratum_candidate_count": int(sum(r["count"] for r in stratum_records if r["eligible"])), "eligible_stratum_candidate_fraction": float(sum(r["count"] for r in stratum_records if r["eligible"]) / len(physical)), "extreme_group_candidate_count": int(len(top) + len(bottom)), "top_count": int(len(top)), "bottom_count": int(len(bottom)), "top_near_fraction": top_rate, "bottom_near_fraction": bottom_rate, "top_minus_bottom_near_fraction": top_rate - bottom_rate, "bootstrap": bootstrap, "records": stratum_records},
        "endpoints": {name: {"auc": aucs[name], "auc_unambiguous_count": int((near | far).sum()), "top5_precision": precisions[name]} for name in aucs},
        "label_prevalence": {"near_count": int(near.sum()), "far_count": int(far.sum()), "middle_count": int((~near & ~far).sum()), "near_fraction": float(near.mean()), "far_fraction": float(far.mean()), "middle_fraction": float((~near & ~far).mean())},
        "decision": {"criteria": criteria, "supported": supported, "plain_conclusion": conclusion},
    }
    write_json(OUTPUT_DIR / "summary.json", summary)
    with (OUTPUT_DIR / "agreement_quintiles.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(agreement_quintiles[0]))
        writer.writeheader()
        writer.writerows(agreement_quintiles)
    np.savez_compressed(OUTPUT_DIR / "evaluation.npz", physical_distance_m=physical, near=near, far=far)
    examples = make_examples(scores, candidate, physical)
    with (OUTPUT_DIR / "examples.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(examples[0]))
        writer.writeheader()
        writer.writerows(examples)
    write_report(summary)
    print(json.dumps({"candidate_count": len(candidate["left"]), "decomposition_max_abs": float(np.max(np.abs(decomposition))), "eligible_stratum_fraction": summary["stratified"]["eligible_stratum_candidate_fraction"], "top_minus_bottom_near_fraction": summary["stratified"]["top_minus_bottom_near_fraction"], "aucs": aucs, "precisions": precisions, "criteria": criteria, "supported": supported}, indent=2))


if __name__ == "__main__":
    main()
