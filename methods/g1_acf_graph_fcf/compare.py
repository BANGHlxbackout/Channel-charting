"""Compare saved B0, G1-Identity, and G1-Graph charts on one protocol."""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common_metrics import affine_transform, chart_metrics, localization_error_summary, write_json
from run_adp_siamese import plot_chart


def load_chart(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    saved = np.load(path)
    return saved["indices"], saved["positions"], saved["embedding"]


def compare(args: argparse.Namespace) -> dict:
    paths = {"B0": args.baseline_embedding, "G1-Identity": args.identity_embedding,
             "G1-Graph": args.graph_embedding}
    return compare_paths(paths, args.out_dir, args.seed)


def compare_paths(paths: dict[str, Path], out_dir: Path, seed: int = 10,
                  subset_path: Path | None = None) -> dict:
    charts = {name: load_chart(path) for name, path in paths.items()}
    ref_idx, ref_pos, _ = next(iter(charts.values()))
    for name, (indices, positions, embedding) in charts.items():
        if not np.array_equal(indices, ref_idx) or not np.allclose(positions, ref_pos, atol=1e-6):
            raise ValueError(f"{name} sample indices or ground truth positions differ from reference")
        if embedding.shape != (len(ref_idx), 2):
            raise ValueError(f"{name} embedding shape mismatch")
    n = len(ref_idx)
    subset = (np.load(subset_path) if subset_path is not None else
              np.array(random.Random(seed).sample(range(n), max(1, int(0.2 * n))), dtype=np.int64))
    if subset.ndim != 1 or len(np.unique(subset)) != len(subset) or np.any((subset < 0) | (subset >= n)):
        raise ValueError("invalid common evaluation subset")
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"metric_subset_seed{seed}.npy", subset)
    results = {}
    aligned = {}
    for name, (_, _, embedding) in charts.items():
        metrics = chart_metrics(ref_pos[subset], embedding[subset])
        errors = localization_error_summary(ref_pos, embedding)
        result_path = paths[name].parent / "result.json"
        saved_result = json.loads(result_path.read_text(encoding="utf-8")) if result_path.exists() else {}
        results[name] = {"ct": metrics["ct"], "tw": metrics["tw"], "ks": metrics["ks"],
                         **errors, "source_json": str(result_path.resolve()),
                         "train_time_s": saved_result.get("train_time_s"),
                         "peak_vram_bytes": saved_result.get("peak_vram_bytes")}
        aligned[name] = affine_transform(ref_pos, embedding)
    with (out_dir / "comparison.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["method", "ct", "tw", "ks", "mean_m", "median_m",
                                                  "p90_m", "p95_m", "max_m", "source_json",
                                                  "train_time_s", "peak_vram_bytes"])
        writer.writeheader()
        for name, result in results.items():
            writer.writerow({"method": name, **result})
    legacy_keys = {"B0": "b0", "G1-Identity": "identity", "G1-Graph": "graph"}
    aligned_keys = {name: legacy_keys.get(name, re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_"))
                    for name in charts}
    if len(set(aligned_keys.values())) != len(aligned_keys):
        raise ValueError("ambiguous aligned-chart names")
    np.savez(out_dir / "aligned_charts.npz", indices=ref_idx, positions=ref_pos,
             **{aligned_keys[name]: value for name, value in aligned.items()})
    for suffix, show_aligned in (("raw", False), ("affine", True)):
        columns = len(charts) + 1
        fig, axes = plt.subplots(1, columns, figsize=(4.5 * columns, 4))
        for column, name in enumerate(charts, start=1):
            metric = results[name]
            title = f"{name}\nCT={metric['ct']:.3f}, TW={metric['tw']:.3f}, KS={metric['ks']:.3f}"
            plot_chart(None, ref_pos, charts[name][2], title,
                       axes=(axes[0] if column == 1 else None, axes[column]),
                       align=show_aligned)
        fig.tight_layout()
        prefix = "four_column" if columns == 4 else f"{columns}_column"
        fig.savefig(out_dir / f"{prefix}_{suffix}_channel_chart.png", dpi=180)
        plt.close(fig)
    protocol = {"num_samples": n, "seed": seed, "metric_subset_size": len(subset),
                "metric_subset_path": str((out_dir / f"metric_subset_seed{seed}.npy").resolve()),
                "metric_subset_source": str(subset_path.resolve()) if subset_path else "seeded random sample",
                "alignment": "separate affine fit and evaluation on all identical N nodes",
                "colors": "same GT x/y mapping in all four panels",
                "plot_function": "run_adp_siamese.plot_chart (external axes; align=False for raw)",
                "chart_quality": "same-node transductive; no held-out claim",
                "embeddings": {name: str(path.resolve()) for name, path in paths.items()},
                "aligned_keys": aligned_keys}
    write_json(out_dir / "eval_protocol.json", protocol)
    lines = ["# Channel Chart comparison", "", "Same-node transductive chart quality; GT used only for evaluation.", "",
             "| method | ct | tw | ks | mean_m | median_m | p90_m | p95_m | max_m | source_json | Train time (s) | Peak VRAM (bytes) |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|"]
    for name, result in results.items():
        values = [f"{result[key]:.6f}" for key in ("ct", "tw", "ks", "mean_m", "median_m", "p90_m", "p95_m", "max_m")]
        time_text = str(result["train_time_s"]) if result["train_time_s"] is not None else "unavailable"
        vram_text = str(result["peak_vram_bytes"]) if result["peak_vram_bytes"] is not None else "unavailable"
        source_link = f"[result.json]({Path(result['source_json']).as_posix()})"
        lines.append("| " + " | ".join([name, *values, source_link, time_text, vram_text]) + " |")
    (out_dir / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-embedding", type=Path, required=True)
    parser.add_argument("--identity-embedding", type=Path, required=True)
    parser.add_argument("--graph-embedding", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--out-dir", type=Path, default=Path("results/g1_acf_graph_fcf/comparison"))
    args = parser.parse_args()
    print(json.dumps(compare(args), indent=2))


if __name__ == "__main__":
    main()
