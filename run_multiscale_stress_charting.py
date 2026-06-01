from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from common_dichasus import DEFAULT_DATA1, load_data1
from common_metrics import chart_metrics, localization_error_summary, write_json
from run_adp_fused import select_indices
from run_adp_siamese import ForwardChartingFunction, plot_chart, predict, train_schedule


class PairArrayDataset(Dataset):
    def __init__(self, h: np.ndarray, left: np.ndarray, right: np.ndarray, target: np.ndarray) -> None:
        self.h = torch.from_numpy(h)
        self.left = torch.from_numpy(left.astype(np.int64, copy=False))
        self.right = torch.from_numpy(right.astype(np.int64, copy=False))
        self.target = torch.from_numpy(target.astype(np.float32, copy=False))

    def __len__(self) -> int:
        return len(self.left)

    def __getitem__(self, idx: int):
        i = self.left[idx]
        j = self.right[idx]
        return self.h[i], self.h[j], self.target[idx]


def parse_bands(spec: str) -> list[tuple[int, int, float]]:
    bands = []
    for item in spec.split(","):
        lo_s, hi_s, weight_s = item.split(":")
        lo = int(lo_s)
        hi = int(hi_s)
        weight = float(weight_s)
        if lo < 0 or hi <= lo or weight <= 0:
            raise ValueError(f"invalid band {item!r}")
        bands.append((lo, hi, weight))
    total = sum(weight for _, _, weight in bands)
    return [(lo, hi, weight / total) for lo, hi, weight in bands]


def compute_neighbor_indices(distances: np.ndarray, top_k: int, device: torch.device, cache_path: Path | None) -> np.ndarray:
    if cache_path is not None and cache_path.exists():
        neighbors = np.load(cache_path)
        if neighbors.shape == (distances.shape[0], top_k):
            return neighbors

    print(f"computing geodesic top-{top_k} neighbors on {device}", flush=True)
    with torch.no_grad():
        d = torch.from_numpy(distances).to(device)
        _, idx = torch.topk(d, k=top_k + 1, largest=False, dim=1)
        idx = idx[:, 1:].to(torch.int32).cpu().numpy()
        del d
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, idx)
    return idx


def make_multiscale_pairs(
    neighbors: np.ndarray,
    distances: np.ndarray,
    samples: int,
    bands: list[tuple[int, int, float]],
    uniform_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    if not 0 <= uniform_fraction < 1:
        raise ValueError("--uniform-fraction must satisfy 0 <= value < 1")
    n = distances.shape[0]
    rng = np.random.default_rng(seed)

    band_samples = int(round(samples * (1.0 - uniform_fraction)))
    counts = [int(round(band_samples * weight)) for _, _, weight in bands]
    counts[-1] += band_samples - sum(counts)
    uniform_count = samples - sum(counts)

    left_chunks = []
    right_chunks = []
    for (lo, hi, _), count in zip(bands, counts):
        if hi > neighbors.shape[1]:
            raise ValueError(f"band [{lo}, {hi}) exceeds cached neighbor width {neighbors.shape[1]}")
        anchors = rng.integers(0, n, size=count, endpoint=False, dtype=np.int64)
        rank = rng.integers(lo, hi, size=count, endpoint=False, dtype=np.int64)
        left_chunks.append(anchors)
        right_chunks.append(neighbors[anchors, rank].astype(np.int64, copy=False))

    if uniform_count > 0:
        left_chunks.append(rng.integers(0, n, size=uniform_count, endpoint=False, dtype=np.int64))
        right_chunks.append(rng.integers(0, n, size=uniform_count, endpoint=False, dtype=np.int64))

    left = np.concatenate(left_chunks)
    right = np.concatenate(right_chunks)
    order = rng.permutation(len(left))
    left = left[order]
    right = right[order]
    target = distances[left, right].astype(np.float32, copy=False)

    meta = {
        "samples": int(samples),
        "uniform_count": int(uniform_count),
        "bands": [
            {"lo_rank": int(lo), "hi_rank": int(hi), "weight": float(weight), "samples": int(count)}
            for (lo, hi, weight), count in zip(bands, counts)
        ],
    }
    return left, right, target, meta


def multiscale_stress_loss(left: torch.Tensor, right: torch.Tensor, target: torch.Tensor, margin: float) -> torch.Tensor:
    pred = torch.linalg.norm(left - right, dim=1)
    return torch.mean((pred - target) ** 2 / (target + margin))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA1)
    parser.add_argument("--distance-run", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("results/multiscale_stress_charting"))
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--top-k", type=int, default=512)
    parser.add_argument("--bands", default="0:32:0.35,32:128:0.30,128:512:0.20")
    parser.add_argument("--uniform-fraction", type=float, default=0.15)
    parser.add_argument("--session-limit", type=int, default=None)
    parser.add_argument("--samples-per-session", type=int, default=None)
    parser.add_argument("--lr-scale", type=float, default=1.0)
    parser.add_argument("--metric-subset-fraction", type=float, default=0.2)
    parser.add_argument("--init-model", type=Path, default=None)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    t0 = time.time()
    device = torch.device(args.device)
    data = load_data1(args.data_path, seed=args.seed)
    idx = select_indices(len(data.h_by_array), args.stride, args.max_samples)
    h = data.h_by_array[idx]
    pos = data.ue_pos[idx]

    saved = np.load(args.distance_run / "distance_data.npz")
    distance_indices = saved["indices"]
    if len(distance_indices) != len(idx) or not np.array_equal(distance_indices, idx):
        raise ValueError("distance-run indices do not match requested data selection")
    distances = np.load(args.distance_run / "geodesic.npy").astype(np.float32)
    if not np.all(np.isfinite(distances)):
        raise ValueError("distance matrix contains non-finite values")
    margin = float(np.quantile(distances, 0.01))

    bands = parse_bands(args.bands)
    if max(hi for _, hi, _ in bands) > args.top_k:
        raise ValueError("--top-k must cover the largest band hi-rank")
    neighbor_cache = args.distance_run / f"geodesic_top{args.top_k}.npy"
    neighbors = compute_neighbor_indices(distances, args.top_k, device, neighbor_cache)

    init_tag = "scratch" if args.init_model is None else "init"
    run_name = f"stride{args.stride}_n{len(idx)}_top{args.top_k}_{init_tag}_lr{args.lr_scale:g}_seed{args.seed}"
    out_dir = args.out_dir / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    model = ForwardChartingFunction(input_shape=h.shape[1:]).to(device)
    if args.init_model is not None:
        model.load_state_dict(torch.load(args.init_model, map_location=device))
    optimizer = torch.optim.Adam(model.parameters(), eps=1e-7)

    schedule = train_schedule()
    if args.session_limit is not None:
        schedule = schedule[: args.session_limit]

    history = []
    for session_idx, (lr, batch_size, samples) in enumerate(schedule, start=1):
        lr *= args.lr_scale
        if args.samples_per_session is not None:
            samples = args.samples_per_session
        effective_samples = (samples // batch_size) * batch_size
        for group in optimizer.param_groups:
            group["lr"] = lr
        left, right, target, sampling_meta = make_multiscale_pairs(
            neighbors,
            distances,
            samples=effective_samples,
            bands=bands,
            uniform_fraction=args.uniform_fraction,
            seed=args.seed + 1000 * session_idx,
        )
        dataset = PairArrayDataset(h, left, right, target)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False)
        model.train()
        running_loss = 0.0
        for step, (left_h, right_h, target_d) in enumerate(loader, start=1):
            left_h = left_h.to(device)
            right_h = right_h.to(device)
            target_d = target_d.to(device)
            optimizer.zero_grad(set_to_none=True)
            left_y = model(left_h)
            right_y = model(right_h)
            loss = multiscale_stress_loss(left_y, right_y, target_d, margin)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item())
            if step % 20 == 0 or step == len(loader):
                print(
                    f"session {session_idx}/{len(schedule)} step {step}/{len(loader)} "
                    f"lr={lr:g} bs={batch_size} loss={running_loss / step:.6f}",
                    flush=True,
                )
        history.append(
            {
                "session": session_idx,
                "lr": lr,
                "batch_size": batch_size,
                "samples": effective_samples,
                "loss": running_loss / max(1, len(loader)),
                "sampling": sampling_meta,
            }
        )
        torch.save(model.state_dict(), out_dir / "model.pt")

    embedding = predict(model, h, device)
    subset_size = max(1, int(len(pos) * args.metric_subset_fraction))
    subset_indices = random.sample(range(len(pos)), subset_size)
    metrics = chart_metrics(pos[subset_indices], embedding[subset_indices])
    errors = localization_error_summary(pos, embedding)
    result = {
        "paper_context": "new method on the ADP-native DICHASUS channel charting dataset",
        "method": "Multi-scale Riemannian geodesic stress distillation",
        "dataset": str(args.data_path),
        "distance_run": str(args.distance_run),
        "run_name": run_name,
        "num_samples": int(len(idx)),
        "stride": int(args.stride),
        "seed": int(args.seed),
        "margin": margin,
        "top_k": int(args.top_k),
        "bands": [
            {"lo_rank": int(lo), "hi_rank": int(hi), "weight": float(weight)}
            for lo, hi, weight in bands
        ],
        "uniform_fraction": float(args.uniform_fraction),
        "lr_scale": float(args.lr_scale),
        "init_model": str(args.init_model) if args.init_model else None,
        "metric_subset_fraction": float(args.metric_subset_fraction),
        "metric_subset_size": int(subset_size),
        "history": history,
        "metrics": metrics,
        "localization_errors_after_affine": errors,
        "elapsed_s": float(time.time() - t0),
    }
    np.savez(out_dir / "embedding.npz", indices=idx, positions=pos, embedding=embedding)
    write_json(out_dir / "result.json", result)
    plot_chart(out_dir / "chart.png", pos, embedding, f"Multi-scale Stress\nCT={metrics['ct']:.3f}, TW={metrics['tw']:.3f}, KS={metrics['ks']:.3f}")
    print("RESULT", json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
