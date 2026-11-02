from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class ChannelChartingModel(nn.Module):
    """GCN, fixed heat diffusion, and adaptive anisotropic diffusion encoders."""

    def __init__(self, input_dim: int, hidden_dim: int, model_type: str,
                 dropout: float, diffusion_steps: int, diffusion_time: float):
        super().__init__()
        if model_type not in {"gcn", "heat", "adaptive"}:
            raise ValueError(f"unsupported model type: {model_type}")
        self.model_type = model_type
        self.diffusion_steps = max(4, int(diffusion_steps))
        self.diffusion_time = float(diffusion_time)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.dropout = nn.Dropout(dropout)

        if model_type == "gcn":
            # Three GCN layers: hidden -> hidden -> 2D chart.
            self.gcn_layers = nn.ModuleList([
                nn.Linear(hidden_dim, hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Linear(hidden_dim, 2),
            ])
            self.gcn_norms = nn.ModuleList([
                nn.LayerNorm(hidden_dim), nn.LayerNorm(hidden_dim)
            ])
        else:
            if model_type == "adaptive":
                self.edge_projection = nn.Linear(hidden_dim, 16, bias=False)
                self.edge_gate = nn.Sequential(
                    nn.Linear(19, 32), nn.GELU(), nn.Linear(32, 1)
                )
                self.raw_alpha = nn.Parameter(torch.tensor(0.0))
                self.raw_beta = nn.Parameter(torch.tensor(-2.1972246))
            self.chart_head = nn.Sequential(
                nn.Linear(hidden_dim * 4, hidden_dim * 2),
                nn.GELU(), nn.LayerNorm(hidden_dim * 2), nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, hidden_dim), nn.GELU(),
                nn.Linear(hidden_dim, 2),
            )

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _encode(self, features: torch.Tensor | np.ndarray,
                chunk_size: int | None) -> torch.Tensor:
        if isinstance(features, torch.Tensor):
            if chunk_size is None:
                return self.encoder(features)
            blocks = []
            for start in range(0, len(features), chunk_size):
                block = features[start:start + chunk_size]
                if self.training and torch.is_grad_enabled():
                    blocks.append(checkpoint(self.encoder, block, use_reentrant=False))
                else:
                    blocks.append(self.encoder(block))
            return torch.cat(blocks, dim=0)

        if chunk_size is None:
            block = torch.from_numpy(np.array(features, copy=True)).to(
                next(self.parameters()).device
            )
            return self.encoder(block)
        blocks = []
        device = next(self.parameters()).device
        for start in range(0, len(features), chunk_size):
            stop = min(start + chunk_size, len(features))
            block = torch.from_numpy(
                np.array(features[start:stop], copy=True)
            ).to(device)
            if self.training and torch.is_grad_enabled():
                blocks.append(checkpoint(self.encoder, block, use_reentrant=False))
            else:
                blocks.append(self.encoder(block))
        return torch.cat(blocks, dim=0)

    @staticmethod
    def _fixed_vector_field(h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        return torch.sparse.mm(adj, h) - h

    def _integrate_heat(self, h0: torch.Tensor,
                        adj: torch.Tensor) -> list[torch.Tensor]:
        """Approximate exp(-t L_sym) h0 with a differentiable RK4 solver."""
        h = h0
        steps = self.diffusion_steps
        dt = self.diffusion_time / steps
        checkpoints = {max(1, steps // 4), max(1, steps // 2), steps}
        states: list[torch.Tensor] = []
        for step in range(1, steps + 1):
            k1 = self._fixed_vector_field(h, adj)
            k2 = self._fixed_vector_field(h + 0.5 * dt * k1, adj)
            k3 = self._fixed_vector_field(h + 0.5 * dt * k2, adj)
            k4 = self._fixed_vector_field(h + dt * k3, adj)
            h = h + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            if step in checkpoints:
                states.append(h)
        return states

    def _adaptive_vector_field(self, h: torch.Tensor, h0: torch.Tensor,
                               graph: dict[str, torch.Tensor]) -> torch.Tensor:
        src, dst = graph["edge_index"]
        projected = self.edge_projection(h)
        edge_delta = torch.abs(projected[src] - projected[dst])
        edge_input = torch.cat((edge_delta, graph["edge_attr"]), dim=1)
        logits = self.edge_gate(edge_input).squeeze(1)
        gate = 0.05 + 0.95 * torch.sigmoid(logits)
        conductance = graph["base_weight"] * gate

        n = h.shape[0]
        degree = torch.zeros(n, device=h.device, dtype=h.dtype)
        degree.index_add_(0, src, conductance)
        norm = conductance / torch.sqrt(
            torch.clamp(degree[src] * degree[dst], min=1e-12)
        )
        aggregate = torch.zeros_like(h)
        aggregate.index_add_(0, src, norm.unsqueeze(1) * h[dst])

        # Bounded positive rates keep the explicit RK4 integration stable.
        alpha = 2.0 * torch.sigmoid(self.raw_alpha)
        beta = torch.sigmoid(self.raw_beta)
        # Nonlinear anisotropic graph diffusion + fidelity to the initial state.
        return alpha * (aggregate - h) + beta * (h0 - h)

    def _integrate_adaptive(self, h0: torch.Tensor,
                            graph: dict[str, torch.Tensor]) -> list[torch.Tensor]:
        h = h0
        steps = self.diffusion_steps
        dt = self.diffusion_time / steps
        checkpoints = {max(1, steps // 4), max(1, steps // 2), steps}
        states: list[torch.Tensor] = []
        for step in range(1, steps + 1):
            k1 = self._adaptive_vector_field(h, h0, graph)
            k2 = self._adaptive_vector_field(h + 0.5 * dt * k1, h0, graph)
            k3 = self._adaptive_vector_field(h + 0.5 * dt * k2, h0, graph)
            k4 = self._adaptive_vector_field(h + dt * k3, h0, graph)
            h = h + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            if step in checkpoints:
                states.append(h)
        return states

    def forward(self, features: torch.Tensor | np.ndarray,
                graph: dict[str, torch.Tensor],
                chunk_size: int | None = None) -> torch.Tensor:
        h0 = self._encode(features, chunk_size)
        if self.model_type == "gcn":
            h = h0
            for layer_index, layer in enumerate(self.gcn_layers):
                message = torch.sparse.mm(graph["adj"], h)
                h = layer(h + message)
                if layer_index < 2:
                    h = F.gelu(self.gcn_norms[layer_index](h))
                    h = self.dropout(h)
            return h

        if self.model_type == "heat":
            states = self._integrate_heat(h0, graph["adj"])
        else:
            states = self._integrate_adaptive(h0, graph)
        # Multi-scale readout preserves the initial signal and three diffusion scales.
        return self.chart_head(torch.cat([h0, *states], dim=1))
