"""Train the transductive G1-ACF chart with graph or identity propagation."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from common_dichasus import load_data1
from common_metrics import chart_metrics, localization_error_summary, write_json
from methods.g1_acf_graph_fcf.graph_utils import (audit_shortest_paths, check_distance_run, feature_cache,
                                identity_adjacency, propagation_adjacency,
                                rebuild_directed_graph, train_pairs, validation_pairs)
from run_adp_fused import select_indices
from run_adp_siamese import plot_chart, siamese_loss


class ResidualSparseGCN(nn.Module):
    def __init__(self, hidden: int, dropout: float):
        super().__init__()
        self.lin = nn.Linear(hidden, hidden)
        self.norm = nn.LayerNorm(hidden)
        self.drop = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        message = torch.sparse.mm(adj, h)
        delta = F.relu(self.norm(self.lin(message)))
        return h + self.drop(delta)


class GraphFCFWithExistingACF(nn.Module):
    def __init__(self, input_dim: int = 26624, hidden: int = 512,
                 layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden)
        self.input_bn = nn.BatchNorm1d(hidden, eps=1e-3, momentum=0.01)
        self.convs = nn.ModuleList([ResidualSparseGCN(hidden, dropout) for _ in range(layers)])
        self.head_norm = nn.LayerNorm(hidden)
        self.out = nn.Linear(hidden, 2)
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, features: torch.Tensor | np.ndarray, adj: torch.Tensor,
                chunk_size: int | None = None) -> torch.Tensor:
        if chunk_size is None:
            h = self.input_proj(features)
        else:
            device = self.input_proj.weight.device

            def project(block: torch.Tensor) -> torch.Tensor:
                return self.input_proj(block.to(device))

            blocks = []
            for start in range(0, len(features), chunk_size):
                stop = min(start + chunk_size, len(features))
                block = torch.from_numpy(np.array(features[start:stop], copy=True))
                if self.training and torch.is_grad_enabled():
                    blocks.append(checkpoint(project, block, use_reentrant=False))
                else:
                    blocks.append(project(block))
            h = torch.cat(blocks, dim=0)
        h = self.input_bn(F.relu(h))
        for conv in self.convs:
            h = conv(h, adj)
        return self.out(self.head_norm(h))


def scheduler_factor(step: int, max_steps: int, warmup: int, min_ratio: float) -> float:
    if step <= warmup:
        return max(step / warmup, 1e-8)
    span = max(1, max_steps - warmup)
    progress = min(1.0, (step - warmup) / span)
    return min_ratio + (1 - min_ratio) * (1 + np.cos(np.pi * progress)) / 2


def run(args: argparse.Namespace) -> dict:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    started = time.perf_counter()
    device = torch.device(args.device)
    data = load_data1(args.data_path, seed=args.seed)
    indices = select_indices(len(data.h_by_array), args.stride, args.max_samples)
    run_name = f"stride{args.stride}_n{len(indices)}_seed{args.seed}_{args.mode}_{args.phase}"
    out_dir = args.out_dir / run_name
    if out_dir.exists() and any(out_dir.iterdir()) and not args.force:
        raise FileExistsError(f"run directory already exists: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    h = data.h_by_array[indices]
    n = len(indices)
    fused, geo, neighbors, source_metadata = check_distance_run(
        args.distance_run, args.data_path, indices, args.stride, args.seed)
    if not np.all(np.isfinite(fused)) or not np.all(np.isfinite(geo)):
        raise ValueError("distance matrices have nonfinite entries")
    directed = rebuild_directed_graph(fused, neighbors)
    path_audit = audit_shortest_paths(directed, geo)
    graph_adj, affinity_info = propagation_adjacency(directed, fused)
    adj = graph_adj if args.mode == "graph" else identity_adjacency(n)
    adj = adj.to(device)
    graph_info = {"mode": args.mode, "neighbors": neighbors, "path_audit": path_audit,
                  "affinity": affinity_info, "source_distance_run": str(args.distance_run.resolve()),
                  "source_result": source_metadata}
    write_json(out_dir / "graph_metadata.json", graph_info)

    feat, feat_info = feature_cache(args.data_path, indices, h,
                                    args.out_dir / "_feature_cache", args.feature_chunk_size,
                                    device)
    write_json(out_dir / "feature_metadata.json", feat_info)
    bytes_needed = feat.size * feat.dtype.itemsize
    if args.feature_mode == "auto":
        stream = device.type == "cuda" and bytes_needed > 0.25 * torch.cuda.get_device_properties(device).total_memory
    else:
        stream = args.feature_mode == "stream"
    if stream:
        features = feat
        chunk_size = args.feature_chunk_size
    else:
        features = torch.from_numpy(np.array(feat, copy=True)).to(device)
        chunk_size = None
    model = GraphFCFWithExistingACF(hidden=args.hidden_dim, layers=args.num_layers,
                                     dropout=args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                  weight_decay=args.weight_decay)
    margin = float(np.quantile(geo, 0.01))
    if margin <= 0 or not np.isfinite(margin):
        raise ValueError("nonpositive/nonfinite geodesic loss margin")
    val_left, val_right, val_keys = validation_pairs(n, args.validation_pairs,
                                                      args.seed + 100000)
    val_pair_hash = hashlib.sha256(val_keys.tobytes()).hexdigest()
    pair_rng = np.random.default_rng(args.seed + 200000)
    val_l = torch.from_numpy(val_left).to(device)
    val_r = torch.from_numpy(val_right).to(device)
    val_target = torch.from_numpy(np.asarray(geo[val_left, val_right]).astype(np.float32)).to(device)
    best_loss = float("inf")
    best_step = 0
    stale = 0
    history = []
    first_train_pair_hash = None
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    train_started = time.perf_counter()
    for step in range(1, args.max_steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        lr = args.learning_rate * scheduler_factor(step, args.max_steps, args.warmup_steps,
                                                    args.cosine_min_lr / args.learning_rate)
        for group in optimizer.param_groups:
            group["lr"] = lr
        left, right = train_pairs(n, args.pairs_per_step, pair_rng, val_keys)
        if first_train_pair_hash is None:
            first_keys = np.minimum(left, right) * n + np.maximum(left, right)
            first_train_pair_hash = hashlib.sha256(first_keys.tobytes()).hexdigest()
        target = torch.from_numpy(np.asarray(geo[left, right]).astype(np.float32)).to(device)
        y = model(features, adj, chunk_size)
        loss = siamese_loss(y[torch.from_numpy(left).to(device)],
                            y[torch.from_numpy(right).to(device)], target, margin)
        if not torch.isfinite(loss):
            raise RuntimeError(f"nonfinite training loss at step {step}")
        loss.backward()
        if step == 1:
            bad_grad = [name for name, parameter in model.named_parameters()
                        if parameter.grad is None or not torch.all(torch.isfinite(parameter.grad))]
            if bad_grad:
                raise RuntimeError(f"missing/nonfinite parameter gradients: {bad_grad}")
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip_norm)
        optimizer.step()
        entry = {"step": step, "train_loss": float(loss.item()), "lr": lr,
                 "elapsed_s": time.perf_counter() - train_started}
        history.append(entry)
        if step % args.eval_every == 0 or step == args.max_steps:
            model.eval()
            with torch.no_grad():
                val_y = model(features, adj, chunk_size)
                val_loss = float(siamese_loss(val_y[val_l], val_y[val_r], val_target, margin).item())
            if not np.isfinite(val_loss):
                raise RuntimeError(f"nonfinite validation loss at step {step}")
            entry["validation_loss"] = val_loss
            entry["elapsed_s"] = time.perf_counter() - train_started
            print(json.dumps(entry), flush=True)
            if val_loss < best_loss:
                best_loss, best_step, stale = val_loss, step, 0
                torch.save(model.state_dict(), out_dir / "model_best.pt")
            else:
                stale += 1
            if stale >= args.patience_evals:
                break
    train_time = time.perf_counter() - train_started
    torch.save(model.state_dict(), out_dir / "model_last.pt")
    model.load_state_dict(torch.load(out_dir / "model_best.pt", map_location=device, weights_only=True))
    model.eval()
    with torch.no_grad():
        embedding = model(features, adj, chunk_size).cpu().numpy()
    if not np.all(np.isfinite(embedding)) or np.std(embedding, axis=0).max() < 1e-8:
        raise RuntimeError("embedding is nonfinite or collapsed")
    peak_vram = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    positions = data.ue_pos[indices]  # Evaluation only, after checkpoint selection.
    subset = np.array(random.Random(args.seed).sample(range(n), max(1, int(0.2 * n))),
                      dtype=np.int64)
    metrics = chart_metrics(positions[subset], embedding[subset])
    errors = localization_error_summary(positions, embedding)
    np.savez(out_dir / "embedding.npz", indices=indices, positions=positions,
             embedding=embedding)
    np.save(out_dir / "metric_subset_seed10.npy", subset)
    plot_chart(out_dir / "chart.png", positions, embedding,
               f"G1-ACF {args.mode} ({args.phase})\n"
               f"CT={metrics['ct']:.3f}, TW={metrics['tw']:.3f}, KS={metrics['ks']:.3f}")
    write_json(out_dir / "training_history.json", history)
    result = {"status": "completed", "phase": args.phase, "mode": args.mode,
              "dataset": str(args.data_path.resolve()), "distance_run": str(args.distance_run.resolve()),
              "num_samples": n, "stride": args.stride, "seed": args.seed,
              "input_dim": 26624, "hidden_dim": args.hidden_dim, "num_layers": args.num_layers,
              "dropout": args.dropout, "neighbors": neighbors, "margin": margin,
              "feature_mode": "stream" if stream else "resident", "feature_chunk_size": args.feature_chunk_size,
              "optimizer": "AdamW", "learning_rate": args.learning_rate,
              "weight_decay": args.weight_decay, "pairs_per_step": args.pairs_per_step,
              "validation_pairs": args.validation_pairs,
              "validation_pair_hash": val_pair_hash,
              "first_train_pair_hash": first_train_pair_hash,
              "max_steps": args.max_steps,
              "best_step": best_step, "best_validation_loss": best_loss,
              "actual_steps": step, "train_time_s": train_time,
              "total_time_s": time.perf_counter() - started, "peak_vram_bytes": peak_vram,
              "parameter_count": sum(p.numel() for p in model.parameters()),
              "metrics": metrics, "localization_errors_after_affine": errors,
              "protocol": "same-node transductive chart quality; GT used only after training"}
    write_json(out_dir / "result.json", result)
    (out_dir / "run_command.txt").write_text(" ".join(args.command), encoding="utf-8")
    print("RESULT", json.dumps(result, indent=2), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=Path, default=Path("datasets/adp_old_shard4"))
    parser.add_argument("--distance-run", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("results/g1_acf_graph_fcf"))
    parser.add_argument("--mode", choices=("graph", "identity"), required=True)
    parser.add_argument("--phase", choices=("sanity", "pilot", "full"), required=True)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--feature-mode", choices=("auto", "resident", "stream"), default="auto")
    parser.add_argument("--feature-chunk-size", type=int, default=128)
    parser.add_argument("--pairs-per-step", type=int, default=65536)
    parser.add_argument("--validation-pairs", type=int, default=200000)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--eval-every", type=int, default=20)
    parser.add_argument("--patience-evals", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--cosine-min-lr", type=float, default=1e-5)
    parser.add_argument("--gradient-clip-norm", type=float, default=5.0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    import sys
    args.command = [sys.executable, *sys.argv]
    run(args)


if __name__ == "__main__":
    main()
