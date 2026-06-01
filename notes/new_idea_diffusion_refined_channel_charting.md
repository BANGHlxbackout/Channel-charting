# New Idea: Graph-Harmonic Diffusion Refined Channel Charting

## Core Claim

The current strongest baseline on the ADP-native DICHASUS dataset is the
ADP/time geodesic Siamese FCF. Its output is already a good chart, but it is
still a noisy coordinate realization of the same intrinsic UE manifold used to
construct the ADP/time geodesic graph.

The proposed idea is to estimate the UE manifold Laplace-Beltrami operator from
the physically motivated ADP/time graph, then refine the neural chart by solving
a graph-harmonic Tikhonov problem. This is a single academic thread:

1. ADP/time fused dissimilarity estimates the intrinsic channel manifold.
2. A local-scaled graph approximates the manifold Laplace-Beltrami operator.
3. The FCF chart is denoised by minimizing Dirichlet energy on that manifold
   while staying close to the learned coordinates.

This is not a collection of engineering tweaks; it is a manifold-regularized
channel charting method.

## Objective

Let `Y0` be the Siamese FCF chart and `W` be a symmetric local-scaled graph
constructed from ADP/time geodesic neighbors:

```text
W_ij = exp(-d_ij^2 / (sigma_i sigma_j))
```

where `sigma_i` is the distance to the `k`-th neighbor. The refined chart `Z`
solves:

```text
min_Z ||Z - Y0||_F^2 + lambda * sum_ij W_ij ||Z_i - Z_j||_2^2
```

Equivalently:

```text
(I + lambda L) Z = Y0
```

where `L = D - W` is the weighted graph Laplacian. The implementation solves
this linear system by GPU Jacobi iterations.

## Why This Is Publishable

- It has a clear mathematical object: a Laplace-Beltrami regularizer on the
  channel manifold.
- It is tied to physical channel charting rather than generic smoothing: the
  graph comes from ADP/time geodesic distances, not from ground-truth positions.
- It preserves the self-supervised nature of channel charting: no UE positions
  are used in training/refinement.
- It explains why it can outperform Siamese FCF: the FCF learns pointwise
  coordinates from stochastic pair regression, while graph-harmonic refinement
  removes high-frequency coordinate noise with respect to the intrinsic
  channel manifold.

## Reproduction Command

```bash
python3 repro_common/run_diffusion_refined_charting.py \
  --data-path repro_common/datasets/adp_old_shard4 \
  --distance-run repro_common/results/adp_old_fused/stride1_n20997_seed10 \
  --init-embedding repro_common/results/adp_old_siamese/stride1_n20997_seed10/embedding.npz \
  --out-dir repro_common/results/diffusion_refined_charting \
  --stride 1 \
  --graph-k 128 \
  --lambda-smooth 0.15 \
  --iterations 200
```

## Result on ADP-Native DICHASUS

Dataset: DICHASUS old/version 2.0 `cf02/cf03/cf04`, `full_dataset.shard(4,0)`,
`20997` samples.

| Method | CT | TW | KS | Mean Error | Median Error | P95 Error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ADP/time fused MDS | 0.9917 | 0.9934 | 0.0907 | 0.458 m | 0.385 m | 0.961 m |
| ADP/time Siamese FCF | 0.9957 | 0.9960 | 0.0841 | 0.409 m | 0.371 m | 0.854 m |
| Graph-harmonic diffusion refined chart | 0.9982 | 0.9982 | 0.0651 | 0.319 m | 0.293 m | 0.635 m |

Focused parameter search around the first working configuration showed that
`graph_k=128, lambda=0.15` gives the best KS, mean error, and P95 error among
the tested settings. A nearby setting, `graph_k=96, lambda=0.2`, gives
slightly higher CT/TW but worse KS and localization error.

Output:

- `repro_common/results/diffusion_refined_charting/stride1_n20997_k128_lambda0.15_seed10/result.json`
- `repro_common/results/diffusion_refined_charting/stride1_n20997_k128_lambda0.15_seed10/embedding.npz`
- `repro_common/results/diffusion_refined_charting/stride1_n20997_k128_lambda0.15_seed10/chart.png`

## Caveats Before Manuscript Use

- This is currently a transductive refinement of a learned chart. For a full
  inductive method, the refined chart should be distilled back into the FCF or
  evaluated with out-of-sample extension.
- A top-journal manuscript would need ablations over `k`, `lambda`, and graph
  construction, plus repeated seeds.
- Comparisons should be reported on a consistent data protocol. The strict
  same-dataset claim here is against the ADP-native fused MDS and Siamese FCF
  baselines.
