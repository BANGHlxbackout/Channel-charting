"""One frozen edge-level SC2-inspired Stage-2 experiment.

The SC2 score is defined on candidate graph edges and never becomes node fidelity.
Original D_geo/Y0 are read-only inputs. Ground truth is used only after masks are fixed.
"""
from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from common_metrics import affine_transform, chart_metrics, localization_error_summary  # noqa: E402
from run_diffusion_refined_charting import (  # noqa: E402
    build_local_scaled_graph,
    solve_graph_harmonic_refinement,
)

SOURCE = Path(r"D:\diffusion_refined_charting")
DIST = SOURCE / "results/adp_old_fused/stride1_n20997_seed10"
Y0_FILE = SOURCE / "results/adp_old_siamese/stride1_n20997_seed10/embedding.npz"
Z_FILE = SOURCE / "results/diffusion_refined_charting/stride1_n20997_k128_lambda0.15_seed10/embedding.npz"
OUT = ROOT / "results/sc2_stage2_edge_consensus"
SEED = 10
K = 128
LAM = 0.15
ITER = 200
FRACTION = 0.10
ATTENUATION = 0.50


def jsonable(value):
    if isinstance(value, dict):
        return {k: jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    return value


def get_overlap(binary: sp.csr_matrix, upper: sp.csr_matrix) -> np.ndarray:
    """Return (B @ B)[i,j] for every upper-triangle W edge, in CSR data order."""
    n = binary.shape[0]
    result = np.empty(upper.nnz, dtype=np.int32)
    for start in range(0, n, 32):
        stop = min(start + 32, n)
        block = (binary[start:stop] @ binary).tocsr()
        block.sort_indices()
        for i in range(start, stop):
            a, b = upper.indptr[i:i + 2]
            targets = upper.indices[a:b]
            pa, pb = block.indptr[i - start:i - start + 2]
            columns = block.indices[pa:pb]
            data = block.data[pa:pb]
            at = np.searchsorted(columns, targets)
            valid = at < len(columns)
            values = np.zeros(len(targets), dtype=np.int32)
            hit = np.zeros(len(targets), dtype=bool)
            hit[valid] = columns[at[valid]] == targets[valid]
            values[hit] = data[at[hit]]
            result[a:b] = values
        if stop % 2048 == 0 or stop == n:
            print(f"overlap rows {stop}/{n}", flush=True)
    return result


def graph_from_upper(upper: sp.csr_matrix, factors: np.ndarray) -> sp.coo_matrix:
    u = upper.tocoo()
    data = (u.data * factors).astype(np.float32)
    half = sp.coo_matrix((data, (u.row, u.col)), shape=u.shape).tocsr()
    return (half + half.T).tocoo()


def make_diagnostic(
    score: np.ndarray,
    residual: np.ndarray,
    geo: np.ndarray,
    eligible: np.ndarray,
    positions: np.ndarray,
    row: np.ndarray,
    col: np.ndarray,
    tie: np.ndarray,
) -> dict:
    physical = np.linalg.norm(
        positions[row].astype(np.float64) - positions[col].astype(np.float64), axis=1
    )
    cuts = np.quantile(geo, np.linspace(0, 1, 11)[1:-1])
    bins = np.searchsorted(cuts, geo, side="right")
    bad = np.zeros(len(geo), dtype=bool)
    rows = []
    for b in range(10):
        ix = np.flatnonzero(eligible & (bins == b))
        if len(ix) == 0:
            continue
        threshold = np.quantile(physical[ix], 0.9)
        bad[ix] = physical[ix] >= threshold
        order = ix[np.lexsort((tie[ix], score[ix]))]
        group_n = max(1, len(order) // 5)
        low, high = order[:group_n], order[-group_n:]
        rows.append({
            "geo_bin": b,
            "eligible_edges": len(ix),
            "low_score_bad_rate": bad[low].mean(),
            "high_score_bad_rate": bad[high].mean(),
            "low_score_median_physical_m": np.median(physical[low]),
            "high_score_median_physical_m": np.median(physical[high]),
        })
    ix = np.flatnonzero(eligible)
    return {
        "definition": "Within each D_geo decile and B=1, physical top-decile edges are offline suspect labels.",
        "rows": rows,
        "pooled_low_rate": np.average([x["low_score_bad_rate"] for x in rows],
                                          weights=[x["eligible_edges"] for x in rows]),
        "pooled_high_rate": np.average([x["high_score_bad_rate"] for x in rows],
                                           weights=[x["eligible_edges"] for x in rows]),
        "rho_score_vs_physical": spearmanr(score[ix], physical[ix]).statistic,
        "rho_first_order_residual_vs_physical": spearmanr(residual[ix], physical[ix]).statistic,
        "suspect_count": int(bad.sum()),
        "eligible_count": int(len(ix)),
        "geo_cuts": cuts.tolist(),
    }


def main() -> None:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    yfile = np.load(Y0_FILE)
    zfile = np.load(Z_FILE)
    dfile = np.load(DIST / "distance_data.npz")
    idx = dfile["indices"]
    pos = dfile["positions"]
    if not (np.array_equal(idx, yfile["indices"]) and np.array_equal(idx, zfile["indices"])):
        raise ValueError("Input index order mismatch")
    y0 = yfile["embedding"].astype(np.float32)
    z_saved = zfile["embedding"].astype(np.float32)
    n = len(idx)
    distances = np.load(DIST / "geodesic.npy", mmap_mode="r")
    neighbors = np.load(DIST / "geodesic_top512.npy", mmap_mode="r")
    if n != 20997 or distances.shape != (n, n) or neighbors.shape != (n, 512):
        raise ValueError("Original input shape mismatch")
    if not (np.all(np.isfinite(y0)) and np.all(np.isfinite(z_saved))):
        raise ValueError("Non-finite embedding")
    print("build original W", flush=True)
    graph = build_local_scaled_graph(distances, neighbors, K).tocsr()
    if graph.nnz != 3088334 or (graph != graph.T).nnz:
        raise ValueError("Original Stage-2 graph mismatch")
    upper = sp.triu(graph, k=1).tocsr()
    u = upper.tocoo()
    wr, wc = u.row.astype(np.int32), u.col.astype(np.int32)
    wgeo = np.asarray(distances[wr, wc], dtype=np.float64)
    wy = np.linalg.norm(y0[wr].astype(np.float64) - y0[wc].astype(np.float64), axis=1)
    if np.any(wgeo <= 0) or np.any(wy <= 0):
        raise ValueError("Nonpositive edge distances")

    print("build original top128 compatibility", flush=True)
    rows = np.repeat(np.arange(n, dtype=np.int32), K)
    cols = np.asarray(neighbors[:, :K].reshape(-1), dtype=np.int32)
    dgeo = np.asarray(distances[rows, cols], dtype=np.float64)
    dy = np.linalg.norm(y0[rows].astype(np.float64) - y0[cols].astype(np.float64), axis=1)
    if np.any(dgeo <= 0) or np.any(dy <= 0):
        raise ValueError("Nonpositive local distances")
    log_scale = np.median(np.log(dy) - np.log(dgeo))
    residual = np.abs(np.log(dy) - np.log(dgeo) - log_scale)
    med = np.median(residual)
    mad = np.median(np.abs(residual - med))
    threshold = med + mad
    directed_good = residual <= threshold
    binary = sp.coo_matrix(
        (np.ones(int(directed_good.sum()), dtype=np.int32),
         (rows[directed_good], cols[directed_good])),
        shape=(n, n),
    ).tocsr()
    binary = binary.maximum(binary.T).tocsr()
    binary.data[:] = 1
    binary.setdiag(0)
    binary.eliminate_zeros()
    eligible = np.asarray(binary[wr, wc]).ravel().astype(bool)
    print("compute second-order edge support", flush=True)
    overlap = get_overlap(binary, upper)
    score = np.where(eligible, overlap, 0).astype(np.int32)
    # Verify sparse extraction against direct row intersection on fixed edges.
    check_at = np.random.default_rng(SEED).choice(len(score), 100, replace=False)
    for edge_idx in check_at:
        i, j = int(wr[edge_idx]), int(wc[edge_idx])
        direct = int(binary[i].multiply(binary[j]).sum())
        if score[edge_idx] != (direct if eligible[edge_idx] else 0):
            raise ValueError("Sparse common-witness score mismatch")

    wr_residual = np.abs(np.log(wy) - np.log(wgeo) - log_scale)
    tie = np.random.default_rng(SEED).random(len(score))
    pool = np.flatnonzero(eligible)
    take = max(1, int(FRACTION * len(pool)))
    sc2_order = pool[np.lexsort((tie[pool], score[pool]))]
    first_order_order = pool[np.lexsort((tie[pool], -wr_residual[pool]))]
    sc2_selected = sc2_order[:take]
    fo_selected = first_order_order[:take]
    geo_cuts = np.quantile(wgeo, np.linspace(0, 1, 11)[1:-1])
    geo_bin = np.searchsorted(geo_cuts, wgeo, side="right")
    shuffled = []
    rng = np.random.default_rng(SEED)
    for b in range(10):
        count = int(np.sum(geo_bin[sc2_selected] == b))
        candidates = pool[geo_bin[pool] == b]
        shuffled.append(rng.choice(candidates, count, replace=False))
    shuffled_selected = np.concatenate(shuffled)
    diagnostic = make_diagnostic(score, wr_residual, wgeo, eligible, pos, wr, wc, tie)
    print("edge diagnostic", diagnostic["pooled_low_rate"], diagnostic["pooled_high_rate"], flush=True)

    # Masks are fixed before the GT diagnostic affects any method choice.
    masks = {
        "uniform": np.array([], dtype=np.int64),
        "sc2": sc2_selected,
        "first_order": fo_selected,
        "shuffled": shuffled_selected,
    }
    factors = {}
    for name, selected in masks.items():
        value = np.ones(len(score), dtype=np.float32)
        value[selected] = ATTENUATION
        factors[name] = value
    mass_ratio = float(np.sum(u.data.astype(np.float64) * factors["sc2"]) /
                       np.sum(u.data.astype(np.float64)))
    factors["uniform_mass"] = np.full(len(score), mass_ratio, dtype=np.float32)
    np.savez_compressed(
        OUT / "edge_scores_and_masks.npz",
        row=wr, col=wc, w=u.data, score=score, eligible=eligible,
        first_order_residual=wr_residual,
        sc2_selected=sc2_selected, first_order_selected=fo_selected,
        shuffled_selected=shuffled_selected,
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    subset = np.asarray(random.sample(range(n), int(n * 0.2)), dtype=np.int64)
    metrics = {}
    embeddings = {}
    for name in ("uniform", "sc2", "first_order", "shuffled", "uniform_mass"):
        print("solve", name, flush=True)
        candidate_graph = graph.tocoo() if name == "uniform" else graph_from_upper(upper, factors[name])
        z = solve_graph_harmonic_refinement(y0, candidate_graph, LAM, ITER, device)
        embeddings[name] = z
        metrics[name] = {
            **chart_metrics(pos[subset], z[subset]),
            **localization_error_summary(pos, z),
            "edge_weight_sum": float(candidate_graph.data.astype(np.float64).sum()),
            "graph_edges": int(candidate_graph.nnz),
        }
    np.savez_compressed(OUT / "embeddings.npz", indices=idx, positions=pos, **embeddings)
    uniform_max_abs = float(np.max(np.abs(embeddings["uniform"] - z_saved)))
    if uniform_max_abs > 5e-4:
        raise ValueError(f"Uniform failed saved-baseline reproduction: {uniform_max_abs}")
    result = {
        "question": "Does second-order common support among compatible Stage-2 edges identify harmful smoothing relations?",
        "source_paper": "https://arxiv.org/html/2203.14453",
        "source_code": "https://github.com/ZhiChen902/SC2-PCR/blob/main/SC2_PCR.py",
        "seed": SEED, "n": n, "graph_k": K, "lambda": LAM, "iterations": ITER,
        "first_order_threshold": float(threshold), "log_scale": float(log_scale),
        "median_residual": float(med), "mad_residual": float(mad),
        "eligible_edges": int(eligible.sum()), "candidate_edges": int(len(score)),
        "selected_edges": int(take), "attenuation": ATTENUATION,
        "mass_ratio": mass_ratio, "uniform_max_abs_vs_saved": uniform_max_abs,
        "diagnostic": diagnostic, "metrics": metrics,
        "elapsed_s": time.time() - t0,
    }
    (OUT / "result.json").write_text(json.dumps(jsonable(result), indent=2), encoding="utf-8")
    lines = [
        "# SC²-inspired Stage-2 edge-consensus test", "",
        "This experiment scores candidate Stage-2 graph edges, not nodes. Node fidelity remains 1 in every variant.",
        "", "## Fixed setup", "",
        f"N={n}; seed={SEED}; original W k={K}; lambda={LAM}; Jacobi iterations={ITER}.",
        f"First-order compatibility threshold=median+MAD={threshold:.6f}; B=1 on {eligible.sum():,} of {len(score):,} undirected W edges.",
        f"Bottom {take:,} eligible edges were attenuated by {ATTENUATION}. Uniform solver max coordinate gap to saved Z={uniform_max_abs:.9f}.",
        "", "## Does common support flag suspicious edges?", "",
        "Offline suspect means a GT endpoint separation in the top 10% within its D_geo-distance bin. GT did not enter edge scores or masks.",
        f"Suspect rate, low-score versus high-score edge quintile, pooled across D_geo bins: {diagnostic['pooled_low_rate']:.4f} versus {diagnostic['pooled_high_rate']:.4f}.",
        f"Spearman(score, physical endpoint distance)={diagnostic['rho_score_vs_physical']:.4f}; first-order residual comparison={diagnostic['rho_first_order_residual_vs_physical']:.4f}.",
        "", "## Fixed five-way refinement comparison", "",
        "| Variant | CT | TW | KS | Mean m | Median m | P90 m | P95 m | Max m | Total W |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ("uniform", "sc2", "first_order", "shuffled", "uniform_mass"):
        m = metrics[name]
        lines.append(f"| {name} | {m['ct']:.6f} | {m['tw']:.6f} | {m['ks']:.6f} | {m['mean_m']:.6f} | {m['median_m']:.6f} | {m['p90_m']:.6f} | {m['p95_m']:.6f} | {m['max_m']:.6f} | {m['edge_weight_sum']:.1f} |")
    delta_mean = metrics["sc2"]["mean_m"] - metrics["uniform"]["mean_m"]
    delta_p95 = metrics["sc2"]["p95_m"] - metrics["uniform"]["p95_m"]
    support = (
        all(metrics["sc2"]["mean_m"] < metrics[c]["mean_m"] for c in
            ("uniform", "first_order", "shuffled", "uniform_mass"))
        and delta_p95 <= 0.005
    )
    lines += [
        "", f"SC² mean change versus original baseline: {delta_mean:+.6f} m; P95 change: {delta_p95:+.6f} m.",
        f"Frozen support criterion met: {support}.",
        "", "## Interpretation boundary", "",
        "This tests a specific edge-consensus transfer, not the full SC²-PCR registration pipeline. The relationship between D_geo and Y0 is not a rigid-transform invariant, and Y0 was trained using D_geo. The offline GT suspect label is an imperfect proxy for a graph edge's causal effect on refinement.",
        "No Stage-1 artifact was changed. No node reliability or adaptive fidelity was used. All inputs on D: were read-only.",
        "", "Machine outputs: result.json, edge_scores_and_masks.npz, embeddings.npz.",
    ]
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("RESULT", json.dumps({"diagnostic": jsonable(diagnostic),
                                 "metrics": jsonable(metrics),
                                 "support": support}, indent=2), flush=True)


if __name__ == "__main__":
    main()
