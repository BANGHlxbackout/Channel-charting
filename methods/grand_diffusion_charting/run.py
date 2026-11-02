from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from common_dichasus import load_data1
from common_metrics import chart_metrics, localization_error_summary, write_json
from methods.g1_acf_graph_fcf.graph_utils import check_distance_run, feature_cache, train_pairs
from run_adp_fused import select_indices
from run_adp_siamese import plot_chart, siamese_loss

from .graph import (
    build_graph,
    estimate_geodesic_margin,
    raw_csi_features,
    sample_validation_pairs,
)
from .models import ChannelChartingModel


def _prepare_features(args: argparse.Namespace, h: np.ndarray,
                      indices: np.ndarray, device: torch.device
                      ) -> tuple[np.ndarray, dict[str, Any], int | None]:
    if args.features == "raw":
        feat, meta = raw_csi_features(h)
        bytes_needed = int(feat.nbytes)
        if args.feature_mode == "stream":
            return feat, meta, args.feature_chunk_size
        if args.feature_mode == "auto" and device.type == "cuda":
            total = torch.cuda.get_device_properties(device).total_memory
            if bytes_needed > 0.25 * total:
                return feat, meta, args.feature_chunk_size
        return feat, meta, None

    feat, meta = feature_cache(
        args.data_path, indices, h, args.out_dir / "_feature_cache",
        args.feature_chunk_size, device
    )
    bytes_needed = int(feat.size * feat.dtype.itemsize)
    if args.feature_mode == "stream":
        return feat, {**meta, "feature_type": "acf"}, args.feature_chunk_size
    if args.feature_mode == "auto":
        if device.type == "cuda":
            total = torch.cuda.get_device_properties(device).total_memory
            stream = bytes_needed > 0.25 * total
        else:
            stream = bytes_needed > 2 * 1024**3
        if stream:
            return feat, {**meta, "feature_type": "acf"}, args.feature_chunk_size
    return np.array(feat, copy=True), {**meta, "feature_type": "acf"}, None


def run(args: argparse.Namespace) -> dict[str, Any]:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    if args.diffusion_steps < 4 or args.diffusion_time <= 0:
        raise ValueError("diffusion_steps must be >= 4 and diffusion_time must be positive")
    started = time.perf_counter()

    data = load_data1(args.data_path, seed=args.seed)
    indices = select_indices(len(data.h_by_array), args.stride, args.max_samples)
    n = len(indices)
    h = data.h_by_array[indices]
    timestamps = data.timestamps[indices]
    fused, geo, baseline_neighbors, _ = check_distance_run(
        args.distance_run, args.data_path, indices, args.stride, args.seed
    )
    adp = np.load(args.distance_run / "adp.npy", mmap_mode="r")
    if adp.shape != (n, n):
        raise ValueError(f"ADP shape mismatch: {adp.shape} != {(n, n)}")
    if not np.all(np.isfinite(adp)) or not np.all(np.isfinite(fused)) or not np.all(np.isfinite(geo)):
        raise ValueError("distance matrices contain non-finite values")

    run_name = (f"{args.model}_{args.features}_{args.graph_source}_k{args.neighbors}_"
                f"stride{args.stride}_n{n}_seed{args.seed}")
    out_dir = args.out_dir / run_name
    if out_dir.exists() and any(out_dir.iterdir()) and not args.force:
        raise FileExistsError(
            f"run directory already exists: {out_dir}; use --force or a new output path"
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    graph, graph_meta = build_graph(
        adp, fused, timestamps, args.graph_source, args.neighbors, device
    )
    graph_meta.update({
        "source_distance_run": str(args.distance_run.resolve()),
        "baseline_geodesic_neighbors": int(baseline_neighbors),
    })
    write_json(out_dir / "graph_metadata.json", graph_meta)

    features, feature_meta, chunk_size = _prepare_features(
        args, h, indices, device
    )
    write_json(out_dir / "feature_metadata.json", feature_meta)
    input_dim = int(features.shape[1])
    if isinstance(features, np.ndarray) and chunk_size is None:
        features = torch.from_numpy(
            np.asarray(features, dtype=np.float32)
        ).to(device)

    model = ChannelChartingModel(
        input_dim=input_dim, hidden_dim=args.hidden_dim, model_type=args.model,
        dropout=args.dropout, diffusion_steps=args.diffusion_steps,
        diffusion_time=args.diffusion_time,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    margin = estimate_geodesic_margin(geo, n, args.seed + 90000)
    val_left, val_right, val_keys = sample_validation_pairs(
        n, args.validation_pairs, args.seed + 100000
    )
    val_pair_hash = hashlib.sha256(val_keys.tobytes()).hexdigest()
    val_l = torch.from_numpy(val_left).long().to(device)
    val_r = torch.from_numpy(val_right).long().to(device)
    val_target = torch.from_numpy(
        np.asarray(geo[val_left, val_right], dtype=np.float32)
    ).to(device)
    pair_rng = np.random.default_rng(args.seed + 200000)

    best_loss = float("inf")
    best_step = 0
    stale = 0
    history: list[dict[str, Any]] = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    train_started = time.perf_counter()

    for step in range(1, args.max_steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        if step <= args.warmup_steps:
            lr = args.learning_rate * step / max(1, args.warmup_steps)
        else:
            lr = args.min_learning_rate + 0.5 * (
                args.learning_rate - args.min_learning_rate
            ) * (
                1.0 + math.cos(
                    math.pi * (step - args.warmup_steps) /
                    max(1, args.max_steps - args.warmup_steps)
                )
            )
        for group in optimizer.param_groups:
            group["lr"] = lr

        left, right = train_pairs(n, args.pairs_per_step, pair_rng, val_keys)
        target = torch.from_numpy(
            np.asarray(geo[left, right], dtype=np.float32)
        ).to(device)
        y = model(features, graph, chunk_size)
        left_t = torch.from_numpy(left).long().to(device)
        right_t = torch.from_numpy(right).long().to(device)
        loss = siamese_loss(y[left_t], y[right_t], target, margin)
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite training loss at step {step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), args.gradient_clip_norm
        )
        optimizer.step()

        entry: dict[str, Any] = {
            "step": step, "train_loss": float(loss.item()), "lr": float(lr),
            "elapsed_s": float(time.perf_counter() - train_started),
        }
        if step % args.eval_every == 0 or step == args.max_steps:
            model.eval()
            with torch.no_grad():
                val_y = model(features, graph, chunk_size)
                val_loss = float(
                    siamese_loss(val_y[val_l], val_y[val_r], val_target, margin).item()
                )
            entry["validation_loss"] = val_loss
            entry["elapsed_s"] = float(time.perf_counter() - train_started)
            print(json.dumps(entry), flush=True)
            if val_loss < best_loss:
                best_loss, best_step, stale = val_loss, step, 0
                torch.save(model.state_dict(), out_dir / "model_best.pt")
            else:
                stale += 1
            history.append(entry)
            if stale >= args.patience_evals:
                break
        else:
            history.append(entry)

    train_time = time.perf_counter() - train_started
    torch.save(model.state_dict(), out_dir / "model_last.pt")
    if best_step == 0:
        raise RuntimeError(
            "no best checkpoint was produced; increase eval frequency or max steps"
        )
    model.load_state_dict(torch.load(
        out_dir / "model_best.pt", map_location=device, weights_only=True
    ))
    model.eval()
    with torch.no_grad():
        embedding = model(features, graph, chunk_size).cpu().numpy()
    if not np.all(np.isfinite(embedding)) or np.std(embedding, axis=0).max() < 1e-8:
        raise RuntimeError("embedding is non-finite or collapsed")

    positions = data.ue_pos[indices]  # Evaluation only.
    eval_rng = np.random.default_rng(args.seed + 300000)
    subset = np.sort(eval_rng.choice(
        n, size=max(1, int(0.2 * n)), replace=False
    ))
    metrics = chart_metrics(positions[subset], embedding[subset])
    errors = localization_error_summary(positions, embedding)
    np.savez(
        out_dir / "embedding.npz",
        indices=indices, positions=positions, embedding=embedding
    )
    np.save(out_dir / "metric_subset.npy", subset)
    plot_chart(
        out_dir / "chart.png", positions, embedding,
        f"{args.model.upper()} diffusion CC\nCT={metrics['ct']:.3f}, "
        f"TW={metrics['tw']:.3f}, KS={metrics['ks']:.3f}"
    )
    write_json(out_dir / "training_history.json", history)

    peak_vram = (
        int(torch.cuda.max_memory_allocated(device))
        if device.type == "cuda" else 0
    )
    result = {
        "status": "completed", "model": args.model, "features": args.features,
        "graph_source": args.graph_source, "dataset": str(args.data_path.resolve()),
        "distance_run": str(args.distance_run.resolve()), "num_samples": n,
        "stride": args.stride, "seed": args.seed, "input_dim": input_dim,
        "hidden_dim": args.hidden_dim, "neighbors": args.neighbors,
        "diffusion_steps": args.diffusion_steps, "diffusion_time": args.diffusion_time,
        "margin": margin, "optimizer": "AdamW", "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay, "pairs_per_step": args.pairs_per_step,
        "validation_pairs": args.validation_pairs, "validation_pair_hash": val_pair_hash,
        "best_step": best_step, "best_validation_loss": best_loss,
        "actual_steps": step, "train_time_s": train_time,
        "total_time_s": float(time.perf_counter() - started),
        "peak_vram_bytes": peak_vram,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "metrics": metrics, "localization_errors_after_affine": errors,
        "protocol": (
            "transductive; GT used only for final evaluation; "
            "geodesic targets are self-supervised"
        ),
        "graph": graph_meta,
    }
    write_json(out_dir / "result.json", result)
    (out_dir / "run_command.txt").write_text(
        " ".join(args.command), encoding="utf-8"
    )
    print("RESULT", json.dumps(result, indent=2), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="GRAND-inspired graph-diffusion channel charting"
    )
    parser.add_argument("--data-path", type=Path, default=Path("datasets/adp_old_shard4"))
    parser.add_argument("--distance-run", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("results/grand_diffusion_charting"))
    parser.add_argument("--model", choices=("gcn", "heat", "adaptive"), default="adaptive")
    parser.add_argument("--features", choices=("raw", "acf"), default="raw")
    parser.add_argument("--graph-source", choices=("adp", "fused"), default="adp")
    parser.add_argument("--neighbors", type=int, default=30)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--diffusion-steps", type=int, default=8)
    parser.add_argument("--diffusion-time", type=float, default=1.0)
    parser.add_argument(
        "--feature-mode", choices=("auto", "resident", "stream"), default="auto"
    )
    parser.add_argument("--feature-chunk-size", type=int, default=128)
    parser.add_argument("--pairs-per-step", type=int, default=65536)
    parser.add_argument("--validation-pairs", type=int, default=200000)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--eval-every", type=int, default=20)
    parser.add_argument("--patience-evals", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--min-learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--gradient-clip-norm", type=float, default=5.0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    args.command = [sys.executable, *sys.argv]
    run(args)


if __name__ == "__main__":
    main()
