"""Presentation-only redraws from already saved G1 and SC2 evidence."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from run_adp_siamese import plot_chart  # noqa: E402

plt.rcParams.update({
    "font.family": "Microsoft YaHei",
    "axes.unicode_minus": False,
    "font.size": 12,
})


def box(ax, xy, wh, title, detail, color):
    patch = FancyBboxPatch(xy, *wh, boxstyle="round,pad=0.02",
                           linewidth=1.5, edgecolor=color, facecolor="white")
    ax.add_patch(patch)
    x, y = xy
    w, h = wh
    ax.text(x + w / 2, y + h * .62, title, ha="center", va="center",
            fontsize=15, weight="bold", color=color)
    ax.text(x + w / 2, y + h * .29, detail, ha="center", va="center", fontsize=11)


def arrow(ax, start, end):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=17,
                                 linewidth=1.4, color="#627489"))


def save_diagram_one(path):
    fig, ax = plt.subplots(figsize=(10.5, 3.1))
    ax.set(xlim=(0, 10), ylim=(0, 3))
    ax.axis("off")
    box(ax, (.2, 1.6), (2.4, 1.0), "节点表示", "多天线自相关特征", "#315773")
    box(ax, (3.0, 1.6), (2.4, 1.0), "传播范围", "融合距离的 20 近邻图", "#39775f")
    box(ax, (7.0, 1.6), (2.7, 1.0), "训练监督", "同图最短路测地距离", "#a06b2c")
    box(ax, (1.7, .35), (3.2, .82), "二维图谱网络", "输出初始图谱", "#315773")
    box(ax, (6.2, .35), (2.7, .82), "点对距离损失", "仅在训练时使用", "#a06b2c")
    arrow(ax, (1.4, 1.56), (2.7, 1.2))
    arrow(ax, (4.2, 1.56), (3.9, 1.2))
    arrow(ax, (8.35, 1.56), (7.65, 1.2))
    arrow(ax, (4.95, .76), (6.16, .76))
    fig.tight_layout(pad=.2)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def save_diagram_two(path):
    fig, ax = plt.subplots(figsize=(10.5, 3.1))
    ax.set(xlim=(0, 10), ylim=(0, 3))
    ax.axis("off")
    ax.text(5, 2.65, "相同节点特征、网络参数、训练点对和测地距离损失",
            ha="center", fontsize=14, weight="bold")
    arrow(ax, (5, 2.5), (2.5, 2.15))
    arrow(ax, (5, 2.5), (7.5, 2.15))
    box(ax, (.7, .85), (3.6, 1.25), "图邻居传播方法", "两层传播，使用原邻接", "#39775f")
    box(ax, (5.7, .85), (3.6, 1.25), "无邻居传播对照", "同构网络，使用单位邻接", "#315773")
    ax.text(5, .35, "唯一变化是传播邻接；比较两者才能判断邻居信息的独立作用",
            ha="center", fontsize=12)
    fig.tight_layout(pad=.2)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def save_g1_chart(path):
    comparison = ROOT / "results/g1_acf_graph_fcf/comparison"
    protocol = json.loads((comparison / "eval_protocol.json").read_text(encoding="utf-8"))
    with (comparison / "comparison.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = {row["method"]: row for row in csv.DictReader(stream)}
    methods = (("B0", "原始 Siamese"), ("G1-Identity", "无邻居传播"),
               ("G1-Graph", "图邻居传播"))
    saved = [np.load(Path(protocol["embeddings"][key])) for key, _ in methods]
    ids, positions = saved[0]["indices"], saved[0]["positions"]
    for data in saved[1:]:
        assert np.array_equal(ids, data["indices"])
        assert np.array_equal(positions, data["positions"])
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 7.8))
    for idx, ((key, label), data) in enumerate(zip(methods, saved)):
        row = rows[key]
        title = (f"{label}\nCT {float(row['ct']):.3f} · TW {float(row['tw']):.3f} · "
                 f"KS {float(row['ks']):.3f}")
        plot_chart(None, positions, data["embedding"], title,
                   axes=(axes[0, 0] if idx == 0 else None, axes.flat[idx + 1]),
                   align=True)
    axes[0, 0].set_title("真实位置")
    for ax in axes.flat:
        ax.tick_params(labelsize=10)
    fig.tight_layout(pad=1.0)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def save_g1_cdf(path):
    saved = np.load(ROOT / "results/g1_acf_graph_fcf/comparison/aligned_charts.npz")
    fig, ax = plt.subplots(figsize=(8.3, 4.4))
    for key, label in (("b0", "原始 Siamese"), ("identity", "无邻居传播"),
                       ("graph", "图邻居传播")):
        errors = np.linalg.norm(saved[key] - saved["positions"], axis=1)
        ordered = np.sort(errors)
        ax.plot(ordered, np.arange(1, len(ordered) + 1) / len(ordered), label=label, linewidth=1.8)
    ax.set(xlim=(0, 2), ylim=(0, 1), xlabel="仿射对齐误差（m）", ylabel="累计比例")
    ax.legend(fontsize=11)
    ax.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def save_sc2_bar(path):
    audit = json.loads((ROOT / "results/sc2_stage2_edge_consensus/posthoc_edge_and_sample_audit.json")
                       .read_text(encoding="utf-8"))
    rates = (audit["sc2_selected"]["GT_suspect_fraction"],
             audit["seed10_random_selected"]["GT_suspect_fraction"])
    fig, ax = plt.subplots(figsize=(8.6, 4.0))
    bars = ax.barh([1, 0], rates, color=["#315773", "#d18441"], height=.55)
    ax.set_yticks([1, 0], ["共同支持评分选边", "距离分层随机选边"])
    ax.set(xlim=(0, .35), xlabel="所选连接中物理跨度偏大者的比例（离线诊断）")
    ax.grid(axis="x", alpha=.2)
    ax.set_axisbelow(True)
    for bar, rate in zip(bars, rates):
        ax.text(rate + .008, bar.get_y() + bar.get_height() / 2, f"{rate:.1%}", va="center")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def save_sc2_map(path):
    saved = np.load(ROOT / "results/sc2_stage2_edge_consensus/posthoc_sample_errors.npz")
    positions = saved["positions"]
    differences = (saved["sc2_improvement"], saved["seed10_random_improvement"])
    vmax = max(float(np.percentile(np.abs(np.concatenate(differences)), 98)), 1e-6)
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 5.8), constrained_layout=True)
    for ax, delta, title in zip(axes, differences,
                                ("共同支持评分选边", "距离分层随机选边（一次示例）")):
        points = ax.scatter(positions[:, 0], positions[:, 1], c=delta, s=5,
                            cmap="RdBu", vmin=-vmax, vmax=vmax, linewidths=0,
                            rasterized=True)
        ax.set_title(title, fontsize=14)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("物理位置 x（m）", fontsize=13)
        ax.set_ylabel("物理位置 y（m）", fontsize=13)
        ax.grid(alpha=.2)
    cbar = fig.colorbar(points, ax=axes, shrink=.78)
    cbar.set_label("误差变化（m）", fontsize=13)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def main():
    g1 = OUT / "g1/figures"
    sc2 = OUT / "sc2/figures"
    g1.mkdir(parents=True, exist_ok=True)
    sc2.mkdir(parents=True, exist_ok=True)
    save_diagram_one(g1 / "method_roles.png")
    save_diagram_two(g1 / "propagation_control.png")
    save_g1_chart(g1 / "affine_chart_2x2.png")
    save_g1_cdf(g1 / "error_cdf_zh.png")
    save_sc2_bar(sc2 / "selected_fraction_zh.png")
    save_sc2_map(sc2 / "repair_harm_zh.png")
    print("presentation-only figures saved")


if __name__ == "__main__":
    main()
