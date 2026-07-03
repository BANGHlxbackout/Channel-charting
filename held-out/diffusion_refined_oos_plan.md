# Diffusion-Refined OOS Held-Out Baseline Plan

## Purpose

`diffusion-refined OOS` is the held-out / out-of-sample extension of the mentor
baseline. It is not a CF-HRD method.

The existing train-only diffusion refinement solves:

```math
(I+\lambda L_{\mathrm{phys,train}})Z_{\mathrm{train}}
=
Y_{0,\mathrm{train}}.
```

This gives refined coordinates only for training graph nodes. It does not
produce `Z_test`, so it cannot by itself report held-out test performance.

The OOS baseline extends `Z_train` to held-out test samples using physical
test-to-train neighbors:

```math
(Y_{0,\mathrm{test}}, d_{\mathrm{phys}}(\mathrm{test},\mathrm{train}),
Z_{\mathrm{train}})
\rightarrow
Z_{\mathrm{test}}^{\mathrm{OOS}}.
```

This baseline is required before claiming that CF-HRD is better than the mentor
diffusion-refined baseline.

## Current Progress To Reuse

The train side has already been completed and should be reused directly. Do not
rerun train diffusion unless the train inputs or hyperparameters change.

Existing train diffusion output:

```text
results/heldout/seed10/diffusion_train/embedding_train.npz
```

This file contains:

| Array | Meaning |
|---|---|
| `indices` | Global train indices. |
| `positions` | Train positions, used only for affine fitting/evaluation. |
| `embedding` | `Z_train`, the train-only diffusion-refined chart. |

Existing train graph artifacts:

```text
results/heldout/seed10/adp_fused_train/geodesic_train.npy
results/heldout/seed10/adp_fused_train/geodesic_top512.npy
results/heldout/seed10/adp_fused_train/result.json
```

Known train diffusion settings:

```text
graph_k = 128
top_k = 512
lambda_smooth = 0.15
gamma_train = 13.316666603088379
```

Therefore the missing piece is only:

```text
Z_test_oos
```

## Method

The main OOS baseline should match the train diffusion graph mechanism. The
train diffusion graph is built from ADP/time geodesic distances, not direct
fused distances alone. Therefore, direct fused distance is used only to attach
each test point to the train graph. The final OOS neighbor search and weights
should use a test-to-train geodesic extension.

For a held-out test sample `i`, first compute direct fused physical distance to
each training sample `a`:

```math
d_{ia}^{\mathrm{fused}}
=
\min\left(
d_{\mathrm{ADP}}(i,a),
\gamma_{\mathrm{train}}|\Delta t_{ia}|
\right).
```

Choose `m` train entry nodes by direct fused distance:

```math
\mathcal A_m(i)
=
\operatorname{topm}_{a\in\mathrm{train}}
d_{ia}^{\mathrm{fused}}.
```

Then extend the train geodesic distance to the test point:

```math
d_{ij}^{\mathrm{oos\_geo}}
=
\min_{a\in\mathcal A_m(i)}
\left[
d_{ia}^{\mathrm{fused}}
+
d_{\mathrm{geo,train}}(a,j)
\right].
```

This means the test sample enters the existing train graph through a small set
of direct fused edges, then distances to other train samples are measured along
the train geodesic graph. This uses no test-test graph and no test positions.

Select the OOS train neighbors using the extended geodesic distance:

```math
\mathcal N_k(i)
=
\operatorname{topk}_{j\in\mathrm{train}}
d_{ij}^{\mathrm{oos\_geo}}.
```

Use bipartite local-scaled OOS weights consistent with the train diffusion
local scaling:

```math
w_{ij}^{\mathrm{oos}}
=
\exp\left(
-\frac{
\left(d_{ij}^{\mathrm{oos\_geo}}\right)^2
}{
\sigma_i^{\mathrm{test}}\sigma_j^{\mathrm{train}}
}
\right),
\qquad
\sigma_i^{\mathrm{test}}=d_{i,k}^{\mathrm{oos\_geo}},
\qquad
\sigma_j^{\mathrm{train}}=d_{\mathrm{geo,train}}(j,j_k).
```

Here `j_k` is the `graph_k`-th train neighbor of train node `j`, using the
existing `geodesic_top512.npy` cache.

Then solve the one-point local Tikhonov problem:

```math
z_i^\star
=
\arg\min_z
\|z-Y_{0,\mathrm{test},i}\|_2^2
+
\lambda_{\mathrm{oos}}
\sum_{j\in\mathcal N_k(i)}
w_{ij}^{\mathrm{oos}}
\|z-Z_{\mathrm{train},j}\|_2^2.
```

The closed-form solution is:

```math
Z_{\mathrm{test},i}^{\mathrm{OOS}}
=
\frac{
Y_{0,\mathrm{test},i}
+
\lambda_{\mathrm{oos}}
\sum_{j\in\mathcal N_k(i)}
w_{ij}^{\mathrm{oos}}Z_{\mathrm{train},j}
}{
1+
\lambda_{\mathrm{oos}}
\sum_{j\in\mathcal N_k(i)}
w_{ij}^{\mathrm{oos}}
}.
```

This keeps the mentor baseline structure:

```text
FCF fidelity + physical graph smoothness.
```

## Default Parameters

Use fixed defaults first; do not tune on test labels.

```text
entry_m = 32
k = 128
lambda_oos = 0.15
gamma_train = value from results/heldout/seed10/adp_fused_train/result.json
```

These match the current train diffusion diagnostic setting:

```text
graph_k = 128
top_k = 512
lambda_smooth = 0.15
```

`entry_m=32` controls how many direct fused train entry nodes attach a test
sample to the train graph. It is fixed before evaluation and must not be tuned
on test labels.

## Code Location

The implementation should live under `held-out`, not `CF_HRD`, because this is a
held-out extension of the diffusion-refined baseline.

Planned script:

```text
held-out/run_diffusion_refined_oos.py
```

Default output:

```text
results/heldout/seed10/diffusion_oos/
  embedding_test.npz
  result.json
```

## Inputs

### Train diffusion result

```text
results/heldout/seed10/diffusion_train/embedding_train.npz
```

Use:

| Array | Meaning |
|---|---|
| `indices` | Global train indices. |
| `positions` | Train positions, used only for affine fitting/evaluation. |
| `embedding` | `Z_train`, train-only diffusion-refined chart. |

### Held-out FCF train/test embeddings

```text
results/heldout/seed10/adp_siamese/embedding_train.npz
results/heldout/seed10/adp_siamese/embedding_test.npz
```

Use:

| Array | Meaning |
|---|---|
| train `embedding` | `Y0_train`, for FCF baseline evaluation. |
| test `embedding` | `Y0_test`, fidelity anchor for OOS extension. |
| train/test `positions` | Used only for Stage 5-style evaluation. |

### Held-out train graph metadata

```text
results/heldout/seed10/adp_fused_train/result.json
results/heldout/seed10/adp_fused_train/geodesic_train.npy
results/heldout/seed10/adp_fused_train/geodesic_top512.npy
```

Use:

```text
gamma_train
geodesic_train
train geodesic neighbors
sigma_train from each train node's graph_k-th geodesic neighbor
```

### Full ADP cache

```text
results/adp_old_fused/stride1_n20997_seed10/adp.npy
results/adp_old_fused/stride1_n20997_seed10/distance_data.npz
```

Use only test-to-train ADP distances:

```math
ADP[\mathrm{test},\mathrm{train}].
```

Do not use test-test ADP distances.

### Timestamps

```text
datasets/adp_old_shard4/timestamps.npy
```

Use to compute:

```math
|\Delta t_{\mathrm{test},\mathrm{train}}|.
```

## Outputs

### `embedding_test.npz`

Save:

| Array | Shape | Meaning |
|---|---:|---|
| `indices` | `(num_test,)` | Global test indices. |
| `positions` | `(num_test, 2)` | Test positions, copied for evaluation. |
| `init_embedding` | `(num_test, 2)` | `Y0_test`. |
| `embedding` | `(num_test, 2)` | `Z_test_oos`. |
| `neighbor_indices` | `(num_test, k)` | Global train neighbor indices. |
| `neighbor_weights` | `(num_test, k)` | Test-to-train OOS weights. |
| `neighbor_distances` | `(num_test, k)` | OOS geodesic distances to selected train neighbors. |
| `entry_indices` | `(num_test, entry_m)` | Global train entry-node indices. |
| `entry_distances` | `(num_test, entry_m)` | Direct fused distances to entry nodes. |

### `result.json`

Save:

| Field | Meaning |
|---|---|
| `protocol` | `diffusion_refined_oos_heldout`. |
| `entry_m` | Number of direct fused train entry nodes. |
| `k` | Number of train neighbors. |
| `lambda_oos` | OOS fidelity/smoothness weight. |
| `gamma_train` | Train-estimated ADP/time scale. |
| `num_train`, `num_test` | Split sizes. |
| `uses_test_test_graph` | Must be `false`. |
| `uses_test_positions_for_extension` | Must be `false`. |
| `uses_test_positions_for_evaluation_only` | Must be `true`. |
| `fcf_test_localization_errors` | Held-out FCF baseline. |
| `diffusion_oos_test_localization_errors` | OOS baseline result. |
| `diffusion_oos_test_metrics` | CT/TW/KS on test. |

## Evaluation

Use the same held-out rule as CF-HRD Stage 5:

1. Fit affine alignment on train predictions only.
2. Apply the affine transform to test predictions.
3. Report held-out test localization errors.
4. Report test CT/TW/KS.

For FCF:

```math
A_{\mathrm{FCF}}
=
\arg\min_A
\|A(Y_{0,\mathrm{train}})-P_{\mathrm{train}}\|^2.
```

For diffusion OOS:

```math
A_{\mathrm{diff}}
=
\arg\min_A
\|A(Z_{\mathrm{train}})-P_{\mathrm{train}}\|^2.
```

Then apply `A_diff` to `Z_test_oos`.

## Comparisons

After implementation, compare:

| Method | Held-out test? | OOS/inductive? | Meaning |
|---|---|---|---|
| FCF | yes | yes | Original held-out FCF baseline. |
| Diffusion-refined train | no | no | Train-only diagnostic; cannot report `Z_test`. |
| Diffusion-refined OOS | yes | yes | Direct OOS extension of mentor baseline. |
| CF-HRD `y0_q` | yes | yes | Distilled residual student. |

Decision logic:

1. If diffusion-refined OOS is better than CF-HRD `y0_q`, CF-HRD should not be
   promoted as the main method.
2. If CF-HRD `y0_q` is better than diffusion-refined OOS, CF-HRD has a stronger
   research case.
3. If they are similar, CF-HRD may be framed as a learnable approximation or
   deployment-friendly variant, not necessarily a stronger refinement method.

## Leakage Rules

The OOS extension may use:

```text
Y0_test
test CSI/timestamps
test-to-train ADP/time distances
train geodesic distances
Z_train
train positions for affine fitting
test positions for final evaluation only
```

The OOS extension must not use:

```text
test positions during extension
test-test graph
test-test geodesic distances
full train+test diffusion solve
test labels for parameter tuning
```
