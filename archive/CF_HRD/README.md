# CF-HRD validation

This folder contains the staged validation code for Cross-Fitted Harmonic
Residual Distillation.

See [closed_loop_validation.md](closed_loop_validation.md) for the full validation
loop and pass/fail criteria.

## Current implemented scope

1. Load the existing held-out training artifacts:
   - ADP/time train graph from `results/heldout/seed10/adp_fused_train`.
   - FCF train embedding from `results/heldout/seed10/adp_siamese/embedding_train.npz`.
2. Generate cross-fitted no-self-anchor harmonic teacher coordinates:

```math
(S^{(-k)}+\lambda L_{\mathrm{phys}})Z^{(-k)}=S^{(-k)}Y_0.
```

The saved residual target is:

```math
\Delta_i^\star=Z_i^{(-k(i))}-Y_{0,i}.
```

Run:

```powershell
D:\anaconda3\envs\diffusion_charting\python.exe CF_HRD\run_cross_fitted_teacher.py
```

Default output:

```text
results/CF_HRD/seed10/cross_fitted_teacher/
```

This stage does not train the residual student and does not report held-out test
refinement performance.

## Seed 10 result: cross-fitted teacher

Run completed with the default configuration:

```text
seed=10
num_train=16798
num_folds=5
fold_sizes=3360 / 3360 / 3360 / 3359 / 3359
graph_k=128
top_k=512
lambda_teacher=0.15
teacher_iterations=200
graph_edges=2459366
```

Output files:

```text
results/CF_HRD/seed10/cross_fitted_teacher/teacher_train.npz
results/CF_HRD/seed10/cross_fitted_teacher/result.json
```

Saved arrays in `teacher_train.npz`:

| Array | Shape | Meaning |
|---|---:|---|
| `indices` | `(16798,)` | Global train sample indices. |
| `positions` | `(16798, 2)` | Train positions, used only for diagnostics. |
| `init_embedding` | `(16798, 2)` | FCF train chart `Y0`. |
| `teacher_embedding` | `(16798, 2)` | Out-of-fold no-self-anchor harmonic teacher. |
| `teacher_residual` | `(16798, 2)` | `teacher_embedding - init_embedding`. |
| `fold_id` | `(16798,)` | Fold assignment used for cross-fitting. |

Train diagnostic comparison:

| Method | mean | P95 | max | CT | TW | KS |
|---|---:|---:|---:|---:|---:|---:|
| FCF train `Y0` | 0.5629 | 1.1875 | 7.0105 | 0.9926 | 0.9896 | 0.1034 |
| Diffusion-refined train, full anchor | 0.4122 | 0.8502 | 2.0319 | 0.9966 | 0.9939 | 0.0795 |
| CF-HRD cross-fitted teacher | 0.3977 | 0.8616 | 1.9614 | 0.9966 | 0.9938 | 0.0802 |

Convergence check:

The original teacher run used 100 Jacobi iterations. A 200-iteration rerun
changed mean/P95/max only from `0.3977028 / 0.8615996 / 1.9614326` to
`0.3976993 / 0.8615556 / 1.9614319`. The default is now 200 iterations to match
the existing `diffusion_train` diagnostic setting.

Interpretation:

The cross-fitted teacher is not meant to replace the full train diffusion
solver. Its role is to generate leakage-reduced residual supervision. The useful
signal here is that the teacher remains close to full diffusion refinement even
when each target fold has its own `Y0_i` fidelity anchor removed. This supports
the next validation stage: learning whether a held-out usable student can
predict this residual from FCF output and physical landmark context.

Important limitation:

The current result is still train-only diagnostic evidence. It does not prove
that CF-HRD improves held-out test refinement. That requires the later student
distillation and train-fitted affine evaluation described in
[closed_loop_validation.md](closed_loop_validation.md).

## Stage 3 result: landmark context

Implemented script:

```powershell
D:\anaconda3\envs\diffusion_charting\python.exe CF_HRD\run_landmark_context.py
```

Implementation:

1. Load the held-out train/test split from
   `results/heldout/seed10/adp_fused_train/distance_data.npz`.
2. Load `gamma_train=13.316666603088379` from
   `results/heldout/seed10/adp_fused_train/result.json`.
3. Select `B=128` landmarks from the train split only using farthest-point
   sampling on `geodesic_train.npy`, initialized by the train node with minimum
   mean train-geodesic distance.
4. Compute direct sample-to-train-landmark fused distances:

```math
d_{\mathrm{fused}}(i,a_b)
=
\min(d_{\mathrm{ADP}}(i,a_b),\gamma_{\mathrm{train}}|\Delta t_{i,a_b}|).
```

5. Fit per-landmark RBF scales on train only:

```math
\sigma_b=\operatorname{median}_{i\in\mathrm{train}}d_{\mathrm{fused}}(i,a_b).
```

6. Apply the same scales to train and test:

```math
q_{i,b}=\exp\left(-(d_{\mathrm{fused}}(i,a_b)/\sigma_b)^2\right).
```

Output files:

```text
results/CF_HRD/seed10/landmark_context/context.npz
results/CF_HRD/seed10/landmark_context/result.json
```

Saved arrays in `context.npz`:

| Array | Shape | Meaning |
|---|---:|---|
| `train_indices` | `(16798,)` | Global train sample indices. |
| `test_indices` | `(4199,)` | Global test sample indices. |
| `landmark_indices` | `(128,)` | Global train-sample landmark indices. |
| `landmark_local_indices` | `(128,)` | Local indices into the train split. |
| `q_train` | `(16798, 128)` | Train landmark context. |
| `q_test` | `(4199, 128)` | Test landmark context. |
| `sigma` | `(128,)` | Train-fitted per-landmark RBF scales. |
| `gamma_train` | scalar | ADP/time scaling factor from train only. |
| `fused_train` | `(16798, 128)` | Auditable train fused distances to landmarks. |
| `fused_test` | `(4199, 128)` | Auditable test fused distances to train landmarks. |

Validation summary:

| Check | Result |
|---|---|
| Landmarks are train subset | pass |
| Landmark indices unique | pass |
| `q_train` finite | pass |
| `q_test` finite | pass |
| `sigma` finite and positive | pass |
| Uses test positions for features | false |
| Uses test-test graph | false |
| Uses train+test diffusion solve | false |

Numerical summary:

| Quantity | min | mean/median | max |
|---|---:|---:|---:|
| `q_train` | 0.2231 | 0.3780 mean | 1.0000 |
| `q_test` | 0.2208 | 0.3777 mean | 0.9999 |
| `sigma` | 38.8447 | 41.3807 median | 44.7706 |

Hashes recorded in `result.json`:

```text
train_indices_sha256 = 9e61398817cd49ee4f4792937220b6c2a1cf09343317a6ae08f5d4042a237c38
test_indices_sha256 = 6baabd3884c23c695f308bade37cefe13ac288a9e47c6a37590e2092f311fa20
landmark_indices_sha256 = b65926bb492826afcee65b96a3f2e7763f95a1e1f9523b4afc6a742a5d9f66f6
```

Stage 3 is complete for seed 10. The next stage can use `q_train/q_test` and
`teacher_residual` to train the three required residual-student variants:
`y0_only`, `q_only`, and `y0_q`.

## Stage 4 result: residual student

Implemented script:

```powershell
D:\anaconda3\envs\diffusion_charting\python.exe CF_HRD\run_residual_student.py
```

Implementation:

1. Load the cross-fitted teacher residual from
   `results/CF_HRD/seed10/cross_fitted_teacher/teacher_train.npz`.
2. Load landmark context from
   `results/CF_HRD/seed10/landmark_context/context.npz`.
3. Load held-out FCF train/test embeddings from
   `results/heldout/seed10/adp_siamese/`.
4. Train three fixed ridge residual students with `alpha=1.0`:
   - `y0_only`: input `Y0_train`.
   - `q_only`: input `q_train`.
   - `y0_q`: input `[Y0_train, q_train]`.
5. Fit feature mean/std on train only and apply unchanged to test.
6. Export train/test predictions for Stage 5. No test labels, test positions,
   or test statistics are used for training, scaling, or model selection.

Model:

```math
\widehat W
=
\arg\min_W
\|XW-\Delta^\star\|_F^2+\alpha\|W\|_F^2,
\qquad \alpha=1.0.
```

Final exported chart:

```math
\widehat Z=Y_0+\widehat\Delta.
```

Output files:

```text
results/CF_HRD/seed10/residual_student/result.json
results/CF_HRD/seed10/residual_student/y0_only_embedding_train.npz
results/CF_HRD/seed10/residual_student/y0_only_embedding_test.npz
results/CF_HRD/seed10/residual_student/q_only_embedding_train.npz
results/CF_HRD/seed10/residual_student/q_only_embedding_test.npz
results/CF_HRD/seed10/residual_student/y0_q_embedding_train.npz
results/CF_HRD/seed10/residual_student/y0_q_embedding_test.npz
```

Saved arrays in each embedding file:

| Array | Train shape | Test shape | Meaning |
|---|---:|---:|---|
| `indices` | `(16798,)` | `(4199,)` | Global sample indices. |
| `init_embedding` | `(16798, 2)` | `(4199, 2)` | Original FCF chart. |
| `predicted_residual` | `(16798, 2)` | `(4199, 2)` | Student-predicted residual. |
| `embedding` | `(16798, 2)` | `(4199, 2)` | `init_embedding + predicted_residual`. |

Train residual fit to the cross-fitted teacher:

| Variant | feature dim | MSE | MAE | RMSE | mean error norm | relative mean norm |
|---|---:|---:|---:|---:|---:|---:|
| `y0_only` | 2 | 179.3092 | 9.9258 | 13.3906 | 15.4680 | 0.7796 |
| `q_only` | 128 | 127.2429 | 8.1075 | 11.2802 | 12.6601 | 0.6381 |
| `y0_q` | 130 | 83.6696 | 6.4440 | 9.1471 | 10.0952 | 0.5088 |

Predicted residual norm summary:

| Variant | train mean / P95 / max | test mean / P95 / max |
|---|---:|---:|
| `y0_only` | 13.8273 / 24.9015 / 30.9937 | 13.8216 / 24.5067 / 29.9089 |
| `q_only` | 15.8293 / 31.7827 / 62.9939 | 15.8545 / 31.1130 / 55.5704 |
| `y0_q` | 17.5645 / 36.2740 / 59.9860 | 17.5580 / 35.8599 / 61.6709 |
| teacher target | 19.8406 / 44.0637 / 284.7039 | n/a |

Validation summary:

| Check | Result |
|---|---|
| Required variants exported | pass |
| Train/test indices aligned | pass |
| Train/test embeddings finite | pass |
| Predicted residuals finite | pass |
| Uses test labels for training | false |
| Uses test positions | false |
| Uses test statistics for scaling | false |

Interpretation:

Stage 4 is complete and ready for Stage 5. The strongest train residual
imitation is `y0_q`, followed by `q_only`, then `y0_only`, so the physical
landmark context carries additional information beyond the FCF coordinates in
the train teacher-fitting task. This is not yet evidence of held-out localization
improvement; Stage 5 must fit affine alignment on train outputs and evaluate on
test positions.

## Stage 5 result: held-out evaluation

Implemented script:

```powershell
D:\anaconda3\envs\diffusion_charting\python.exe CF_HRD\run_stage5_evaluation.py
```

Evaluation rule:

1. Fit affine alignment on train embedding and train positions.
2. Apply the same affine transform to the held-out test embedding.
3. Report held-out test localization errors and test CT/TW/KS.

Output:

```text
results/CF_HRD/seed10/stage5_evaluation/result.json
```

Held-out test localization:

| Method | mean | median | P90 | P95 | max |
|---|---:|---:|---:|---:|---:|
| FCF | 0.5766 | 0.5007 | 1.0126 | 1.2026 | 7.5254 |
| `y0_only` | 0.5766 | 0.5007 | 1.0126 | 1.2026 | 7.5254 |
| `q_only` | 0.5136 | 0.4474 | 0.8850 | 1.1012 | 7.6570 |
| `y0_q` | 0.4871 | 0.4261 | 0.8299 | 1.0211 | 7.2932 |

Held-out test chart metrics:

| Method | CT | TW | KS |
|---|---:|---:|---:|
| FCF | 0.9911 | 0.9884 | 0.1083 |
| `y0_only` | 0.9913 | 0.9884 | 0.1084 |
| `q_only` | 0.9900 | 0.9881 | 0.1013 |
| `y0_q` | 0.9917 | 0.9893 | 0.0991 |

`y0_q` vs FCF deltas:

| Metric | Delta |
|---|---:|
| mean | -0.0895 |
| median | -0.0746 |
| P90 | -0.1828 |
| P95 | -0.1815 |
| max | -0.2323 |
| CT | +0.0006 |
| TW | +0.0009 |
| KS | -0.0093 |

Interpretation:

Stage 5 gives a positive seed-10 held-out signal. The main `y0_q` student
improves all reported held-out localization errors over FCF and also improves
CT/TW/KS. The `q_only` student improves mean/P90/P95 but worsens max and slightly
reduces CT/TW, so the combined `Y0 + q` variant is the cleaner candidate.

The `y0_only` result is effectively identical to FCF after affine alignment.
This is expected because a ridge residual using only `Y0` mostly produces an
affine-like coordinate correction, which train-fitted affine alignment can
absorb. Its lack of localization improvement is a useful control: the gain in
`y0_q` is not explained by a simple coordinate-only linear correction.

Limitation:

This is still one held-out seed. It is enough to continue the CF-HRD line, but
not enough for a paper claim. The next verification should repeat Stage 1-5 over
multiple seeds and compare against direct full-diffusion residual distillation.

## Current problems and risks

### 1. Student is much weaker than the teacher

The cross-fitted teacher is a strong train-side repair target, but the current
ridge student does not fully imitate it.

| Method | train mean | train P95 | train max |
|---|---:|---:|---:|
| CF-HRD cross-fitted teacher | 0.3977 | 0.8616 | 1.9614 |
| CF-HRD `y0_q` student | 0.4623 | 0.9920 | 6.8877 |

This gap is important. It means the current student only learns part of the
graph-harmonic repair signal. Stage 4 confirms the same issue at the residual
level:

```text
teacher residual norm mean = 19.8406
y0_q residual prediction error norm mean = 10.0952
relative mean norm = 0.5088
```

Interpretation:

The `y0_q` student has a positive held-out signal, but it is still a weak
inductive approximation of the teacher. The current ridge model should not be
presented as a mature replacement for graph diffusion refinement.

### 2. Missing diffusion-refined OOS baseline

The current held-out comparison is:

```text
FCF held-out vs CF-HRD y0_q held-out
```

This is not enough to claim that CF-HRD improves the mentor baseline. The mentor
baseline is diffusion refinement; its most direct held-out version should be a
diffusion-refined train graph plus out-of-sample extension:

```math
Z_{\mathrm{train}}
=(I+\lambda L_{\mathrm{phys,train}})^{-1}Y_{0,\mathrm{train}},
```

then estimate `Z_test` from train physical neighbors or a local harmonic
extension.

Until this OOS baseline is implemented, the correct claim is limited:

```text
CF-HRD y0_q improves held-out FCF on seed 10.
```

It does not yet show:

```text
CF-HRD is better than diffusion-refined charting.
```

### 3. Current claim boundary

Supported by current evidence:

1. The no-self-anchor teacher is close to train diffusion refinement.
2. Landmark context `q` helps imitate the teacher residual on train.
3. `y0_q` improves held-out FCF on seed 10.

Not yet supported:

1. CF-HRD beats a diffusion-refined out-of-sample extension.
2. CF-HRD is better than the mentor diffusion-refined baseline.
3. The result is stable across seeds.
4. The ridge student is strong enough to preserve the teacher's repair quality.
