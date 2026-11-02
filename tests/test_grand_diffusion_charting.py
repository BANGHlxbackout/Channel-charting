import numpy as np
import torch

from methods.grand_diffusion_charting.graph import (
    build_graph,
    estimate_geodesic_margin,
    raw_csi_features,
    sample_validation_pairs,
)
from methods.grand_diffusion_charting.models import ChannelChartingModel


def _synthetic_distances(n: int = 40):
    rng = np.random.default_rng(7)
    xy = rng.normal(size=(n, 2)).astype(np.float32)
    adp = np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=-1).astype(np.float32)
    fused = (0.8 * adp).astype(np.float32)
    timestamps = np.arange(n, dtype=np.float32)
    return adp, fused, timestamps


def test_graph_is_symmetric_and_edge_modes_are_controlled():
    adp, fused, timestamps = _synthetic_distances()
    graph_adp, meta = build_graph(
        adp, fused, timestamps, "adp", 6, "adp", torch.device("cpu")
    )
    dense = graph_adp["adj"].to_dense()
    assert torch.allclose(dense, dense.T, atol=1e-6)
    assert graph_adp["edge_attr"].shape[1] == 3
    assert torch.count_nonzero(graph_adp["edge_attr"][:, 1:]) == 0
    assert meta["num_undirected_edges"] > 0

    graph_multi, _ = build_graph(
        adp, fused, timestamps, "adp", 6, "multiview", torch.device("cpu")
    )
    assert torch.count_nonzero(graph_multi["edge_attr"][:, 1:]) > 0


def test_all_diffusion_encoders_have_finite_gradients():
    adp, fused, timestamps = _synthetic_distances()
    graph, _ = build_graph(
        adp, fused, timestamps, "adp", 6, "adp", torch.device("cpu")
    )
    features = torch.randn(len(adp), 24)

    for model_type in ("gcn", "heat", "adaptive"):
        torch.manual_seed(11)
        model = ChannelChartingModel(
            input_dim=24, hidden_dim=16, model_type=model_type,
            dropout=0.1, diffusion_steps=8, diffusion_time=1.0,
        )
        embedding = model(features, graph)
        assert embedding.shape == (len(adp), 2)
        loss = embedding.square().mean()
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.requires_grad]
        assert grads
        assert all(g is not None and torch.isfinite(g).all() for g in grads)


def test_feature_and_pair_helpers():
    rng = np.random.default_rng(5)
    h = (
        rng.normal(size=(20, 4, 8, 13))
        + 1j * rng.normal(size=(20, 4, 8, 13))
    ).astype(np.complex64)
    features, meta = raw_csi_features(h)
    assert features.shape == (20, 4 * 8 * 13 * 2)
    assert meta["feature_type"] == "raw_csi_real_imag"

    xy = rng.normal(size=(20, 2))
    geo = np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=-1)
    assert estimate_geodesic_margin(geo, 20, seed=9) > 0

    left, right, keys = sample_validation_pairs(20, 200, seed=13)
    assert len(left) == len(right) == 200
    assert np.all(left != right)
    assert np.all(keys >= 0)
