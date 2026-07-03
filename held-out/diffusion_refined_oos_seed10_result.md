# Diffusion-Refined OOS Held-Out Result, Seed 10

## Command

```powershell
D:\anaconda3\envs\diffusion_charting\python.exe held-out\run_diffusion_refined_oos.py --seed 10
```

## Protocol

This run implements the diffusion-refined out-of-sample baseline described in
`held-out/diffusion_refined_oos_plan.md`.

It reuses:

- `results/heldout/seed10/diffusion_train/embedding_train.npz` as `Z_train`.
- `results/heldout/seed10/adp_fused_train/geodesic_train.npy`.
- `results/heldout/seed10/adp_fused_train/geodesic_top512.npy`.
- `results/heldout/seed10/adp_siamese/embedding_train.npz`.
- `results/heldout/seed10/adp_siamese/embedding_test.npz`.

It produces:

- `results/heldout/seed10/diffusion_oos/embedding_test.npz`
- `results/heldout/seed10/diffusion_oos/result.json`

## Parameters

| Parameter | Value |
|---|---:|
| seed | 10 |
| entry_m | 32 |
| k | 128 |
| graph_k | 128 |
| top_k | 512 |
| lambda_oos | 0.15 |
| gamma_train | 13.316666603088379 |
| num_train | 16798 |
| num_test | 4199 |

## Held-Out Test Results

| Method | Mean m | Median m | P90 m | P95 m | Max m | CT | TW | KS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FCF held-out | 0.5766 | 0.5007 | 1.0126 | 1.2026 | 7.5254 | 0.9911 | 0.9884 | 0.1083 |
| Diffusion-refined OOS | 0.4250 | 0.3811 | 0.7274 | 0.8578 | 2.0741 | 0.9961 | 0.9934 | 0.0814 |
| CF-HRD y0_q | 0.4871 | 0.4261 | 0.8299 | 1.0211 | 7.2932 | 0.9917 | 0.9893 | 0.0991 |

## Self-Check

`embedding_test.npz` contains:

| Array | Shape |
|---|---:|
| indices | `(4199,)` |
| positions | `(4199, 2)` |
| init_embedding | `(4199, 2)` |
| embedding | `(4199, 2)` |
| neighbor_indices | `(4199, 128)` |
| neighbor_local_indices | `(4199, 128)` |
| neighbor_weights | `(4199, 128)` |
| neighbor_distances | `(4199, 128)` |
| entry_indices | `(4199, 32)` |
| entry_local_indices | `(4199, 32)` |
| entry_distances | `(4199, 32)` |

Checks:

- `protocol = diffusion_refined_oos_heldout`
- `uses_test_test_graph = false`
- `uses_test_positions_for_extension = false`
- `uses_test_positions_for_evaluation_only = true`
- `embedding` is finite.
- `neighbor_weights` are finite.
- `neighbor_weights` min/max = `0.12600159645080566 / 0.9999989867210388`.

## Interpretation

For seed 10, diffusion-refined OOS is stronger than held-out FCF and also
stronger than the current CF-HRD `y0_q` residual-student result. Therefore the
current evidence does not support promoting CF-HRD as a better method than the
mentor diffusion-refined baseline.

The stronger conclusion is narrower but cleaner: the mentor baseline can be
extended to a held-out / OOS setting, and the physical graph refinement signal
still improves held-out localization error without using test positions during
extension.
