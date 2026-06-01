from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from common_dichasus import DEFAULT_DATA1, load_data1
from common_metrics import affine_transform, chart_metrics, localization_error_summary, write_json
from run_adp_fused import select_indices


class PairDataset(Dataset):
    def __init__(self, h: np.ndarray, distances: np.ndarray, samples: int, seed: int) -> None:
        self.h = torch.from_numpy(h)
        self.distances = distances.astype(np.float32, copy=False)
        rng = np.random.default_rng(seed)
        self.left = rng.integers(0, len(h), size=samples, endpoint=False, dtype=np.int64)
        self.right = rng.integers(0, len(h), size=samples, endpoint=False, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.left)

    def __getitem__(self, idx: int):
        i = int(self.left[idx])
        j = int(self.right[idx])
        return self.h[i], self.h[j], self.distances[i, j]


class AdpFeatureLayer(nn.Module):
    """CSI autocorrelation feature layer from the DICHASUS ADP notebook."""

    def forward(self, csi: torch.Tensor) -> torch.Tensor:
        # csi: (batch, arrays, antennas, taps), complex64.
        corr = torch.einsum("bamt,bcnt->btacmn", csi, torch.conj(csi))
        feat = torch.stack((corr.real, corr.imag), dim=-1)
        return feat.flatten(start_dim=1)


class ForwardChartingFunction(nn.Module):
    def __init__(self, input_shape: tuple[int, int, int]) -> None:
        super().__init__()
        arrays, antennas, taps = input_shape
        feature_dim = taps * arrays * arrays * antennas * antennas * 2
        self.features = AdpFeatureLayer()
        layers: list[nn.Module] = []
        prev = feature_dim
        for width in (1024, 512, 256, 128, 64):
            layers.append(nn.Linear(prev, width))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(width, eps=1e-3, momentum=0.01))
            prev = width
        layers.append(nn.Linear(prev, 2))
        self.net = nn.Sequential(*layers)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, csi: torch.Tensor) -> torch.Tensor:
        return self.net(self.features(csi))


def siamese_loss(left: torch.Tensor, right: torch.Tensor, target: torch.Tensor, margin: float) -> torch.Tensor:
    pred = torch.linalg.norm(left - right, dim=1)
    return torch.mean((pred - target) ** 2 / (target + margin))


def train_schedule() -> list[tuple[float, int, int]]:
    learning_rates = [1e-2, 1e-2, 8e-3, 4e-3, 1e-3, 5e-4, 2e-4, 1e-4]
    batch_sizes = [500, 1000, 1500, 2000, 3000, 4000, 5000, 6000]
    samples_per_session = 150_000
    return [(lr, bs, samples_per_session) for lr, bs in zip(learning_rates, batch_sizes)]


@torch.no_grad()
def predict(model: nn.Module, h: np.ndarray, device: torch.device, batch_size: int = 512) -> np.ndarray:
    model.eval()
    loader = DataLoader(torch.from_numpy(h), batch_size=batch_size, shuffle=False)
    out = []
    for batch in loader:
        out.append(model(batch.to(device)).cpu().numpy())
    return np.concatenate(out, axis=0)


def plot_chart(path: Path, reference: np.ndarray, embedding: np.ndarray, title: str) -> None:
    aligned = affine_transform(reference, embedding)
    colors = np.zeros((len(reference), 3), dtype=np.float32)
    colors[:, 0] = (reference[:, 0] - reference[:, 0].min()) / np.ptp(reference[:, 0])
    colors[:, 1] = (reference[:, 1] - reference[:, 1].min()) / np.ptp(reference[:, 1])
    colors[:, 2] = 0.25

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].scatter(reference[:, 0], reference[:, 1], c=colors, s=5, linewidths=0)
    axes[0].set_title("Ground Truth")
    axes[0].axis("equal")
    axes[0].grid(True, alpha=0.25)
    axes[1].scatter(aligned[:, 0], aligned[:, 1], c=colors, s=5, linewidths=0)
    axes[1].set_title(title)
    axes[1].axis("equal")
    axes[1].grid(True, alpha=0.25)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA1)
    parser.add_argument("--distance-run", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("results/adp_siamese"))
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--session-limit", type=int, default=None, help="Run only the first N schedule sessions for debugging.")
    parser.add_argument("--samples-per-session", type=int, default=None)
    parser.add_argument("--metric-subset-fraction", type=float, default=0.2)
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

    distance_data_path = args.distance_run / "distance_data.npz"
    saved = np.load(distance_data_path if distance_data_path.exists() else args.distance_run / "embedding.npz")
    distance_indices = saved["indices"]
    if len(distance_indices) != len(idx) or not np.array_equal(distance_indices, idx):
        raise ValueError("distance-run indices do not match requested data selection")
    distances = np.load(args.distance_run / "geodesic.npy").astype(np.float32)
    if not np.all(np.isfinite(distances)):
        raise ValueError("distance matrix contains non-finite values")
    margin = float(np.quantile(distances, 0.01))

    run_name = f"stride{args.stride}_n{len(idx)}_seed{args.seed}"
    out_dir = args.out_dir / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    model = ForwardChartingFunction(input_shape=h.shape[1:]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), eps=1e-7)

    schedule = train_schedule()
    if args.session_limit is not None:
        schedule = schedule[: args.session_limit]

    history = []
    for session_idx, (lr, batch_size, samples) in enumerate(schedule, start=1):
        if args.samples_per_session is not None:
            samples = args.samples_per_session
        effective_samples = (samples // batch_size) * batch_size
        for group in optimizer.param_groups:
            group["lr"] = lr
        dataset = PairDataset(h, distances, samples=effective_samples, seed=args.seed + session_idx)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False)
        model.train()
        running_loss = 0.0
        for step, (left_h, right_h, target) in enumerate(loader, start=1):
            left_h = left_h.to(device)
            right_h = right_h.to(device)
            target = target.to(device)
            optimizer.zero_grad(set_to_none=True)
            left_y = model(left_h)
            right_y = model(right_h)
            loss = siamese_loss(left_y, right_y, target, margin)
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
                "requested_samples": samples,
                "loss": running_loss / max(1, len(loader)),
            }
        )
        torch.save(model.state_dict(), out_dir / "model.pt")

    embedding = predict(model, h, device)
    subset_size = max(1, int(len(pos) * args.metric_subset_fraction))
    subset_indices = random.sample(range(len(pos)), subset_size)
    metrics = chart_metrics(pos[subset_indices], embedding[subset_indices])
    errors = localization_error_summary(pos, embedding)
    result = {
        "paper": "Angle-Delay Profile-Based and Timestamp-Aided Dissimilarity Metrics for Channel Charting",
        "method": "ADP + timestamp fused + geodesic + Siamese FCF",
        "dataset": str(args.data_path),
        "distance_run": str(args.distance_run),
        "run_name": run_name,
        "num_samples": int(len(idx)),
        "stride": int(args.stride),
        "seed": int(args.seed),
        "margin": margin,
        "metric_subset_fraction": float(args.metric_subset_fraction),
        "metric_subset_size": int(subset_size),
        "history": history,
        "metrics": metrics,
        "localization_errors_after_affine": errors,
        "elapsed_s": float(time.time() - t0),
    }
    np.savez(out_dir / "embedding.npz", indices=idx, positions=pos, embedding=embedding)
    write_json(out_dir / "result.json", result)
    plot_chart(out_dir / "chart.png", pos, embedding, f"ADP Siamese\nCT={metrics['ct']:.3f}, TW={metrics['tw']:.3f}, KS={metrics['ks']:.3f}")
    print("RESULT", json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
