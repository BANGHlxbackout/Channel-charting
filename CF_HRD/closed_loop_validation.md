# CF-HRD closed-loop validation

This document defines the validation loop for Cross-Fitted Harmonic Residual
Distillation. The goal is to test whether the current transductive
`diffusion_refined` solver can be converted into an inductive residual repair
module for held-out samples.

## Baseline to preserve

The mentor baseline is:

```math
Y_0=f_\theta(H),
```

```math
(I+\lambda L_{\mathrm{phys}})Z=Y_0.
```

Here `L_phys` is built from the ADP/time physical graph. This solver is strong
on the graph nodes it sees, but it is transductive: it does not directly produce
`Z_test` for held-out samples unless those samples are inserted into a graph and
the graph problem is solved again.

CF-HRD must not be evaluated only as another full-data smoother. Its target is:

```math
\widehat Z_{\mathrm{test}}
=Y_{0,\mathrm{test}}+g_\phi(Y_{0,\mathrm{test}},q_{\mathrm{test}}),
```

where `q_test` is physical landmark context computed without test position
labels.

## Validation stages

### Stage 1: held-out FCF inputs

Existing artifacts:

```text
results/heldout/seed10/adp_fused_train/
results/heldout/seed10/adp_siamese/embedding_train.npz
results/heldout/seed10/adp_siamese/embedding_test.npz
```

Required checks:

1. `distance_data.npz` train indices must match `embedding_train.npz`.
2. Train and test indices must match the deterministic seed split.
3. Test positions are used only for final evaluation, not for feature
   construction or training.

### Stage 2: cross-fitted no-self-anchor teacher

For each fold `M_k`, define a diagonal fidelity mask:

```math
S^{(-k)}_{ii}=
\begin{cases}
0, & i\in M_k,\\
1, & i\notin M_k.
\end{cases}
```

Solve:

```math
(S^{(-k)}+\lambda_T L_{\mathrm{phys}})Z^{(-k)}
=
S^{(-k)}Y_0.
```

For each node `i`, save the out-of-fold teacher residual:

```math
\Delta_i^\star=Z_i^{(-k(i))}-Y_{0,i}.
```

Purpose:

This removes the direct self-anchor from the teacher target for node `i`. If
we distilled the full diffusion result directly, `Z_i` would still depend on
`Y_{0,i}` through the fidelity term, making the target easier to learn as an
identity-like correction. The cross-fitted teacher instead asks whether the
physical graph and other train nodes can infer the repair direction for the
masked node.

Implemented script:

```powershell
D:\anaconda3\envs\diffusion_charting\python.exe CF_HRD\run_cross_fitted_teacher.py
```

Current output:

```text
results/CF_HRD/seed10/cross_fitted_teacher/teacher_train.npz
results/CF_HRD/seed10/cross_fitted_teacher/result.json
```

### Stage 3: physical landmark context

Select train landmarks `A={a_1,...,a_B}` using only training samples and
physical distances. For each train or test sample, compute:

```math
q_i=\Gamma(i;A,d_{\mathrm{phys}}).
```

First implementation can use ADP/time fused distances to train landmarks:

```math
d_{\mathrm{fused}}(i,a_b)
=
\min(d_{\mathrm{ADP}}(i,a_b),\gamma_{\mathrm{train}}|\Delta t_{i,a_b}|).
```

Then convert distances to stable features, for example RBF landmark weights:

```math
q_{i,b}=\exp\left(-d_{\mathrm{fused}}(i,a_b)^2/\sigma_b^2\right).
```

All scaling parameters such as `sigma_b` must be fitted on training samples
only, then reused unchanged for `q_test`.

Constraint:

For test samples, `q_test` may use test CSI/timestamps and train landmarks, but
must not use test positions or test-test graph information.

Completion standard for Stage 3:

Stage 3 is complete only when the code produces a reusable landmark-context
artifact with all of the following properties:

1. Landmarks are selected from the training split only.
2. Saved landmark indices are global sample indices and are a subset of
   `train_indices`.
3. `q_train` has shape `(num_train, B)` and `q_test` has shape `(num_test, B)`.
4. `q_train` and `q_test` are finite, deterministic for the same seed, and use
   the same landmark ordering.
5. Test features are computed only from test CSI/timestamps and train landmark
   CSI/timestamps. Test positions are not loaded or used during feature
   construction.
6. No test-test graph, no test-test geodesic matrix, and no full train+test
   diffusion solve is used.
7. Feature normalization is fitted on train only and applied unchanged to test.
8. The artifact records `B`, landmark selection method, `gamma_train`, feature
   transform, normalization parameters, train/test index hashes or exact index
   arrays, and source files.

Minimum expected output:

```text
results/CF_HRD/seed10/landmark_context/context.npz
results/CF_HRD/seed10/landmark_context/result.json
```

The `.npz` file must contain at least:

| Array | Shape | Meaning |
|---|---:|---|
| `train_indices` | `(num_train,)` | Global train sample indices. |
| `test_indices` | `(num_test,)` | Global test sample indices. |
| `landmark_indices` | `(B,)` | Global train-sample landmark indices. |
| `q_train` | `(num_train, B)` | Train landmark context. |
| `q_test` | `(num_test, B)` | Test landmark context. |

Stage 3 should not be considered complete if it only computes `q_train`. The
point of CF-HRD is held-out use, so `q_test` is required before moving to the
student stage.

### Stage 4: residual student

Train a small residual predictor on train samples only:

```math
\widehat\Delta_i=g_\phi(Y_{0,i},q_i).
```

The first version should be deliberately minimal, such as ridge regression or a
small MLP. The point is to test whether the teacher residual is learnable from
inductive features. If a simple student has no signal, a more complex network
should not be used as a way to hide the failure.

Final prediction:

```math
\widehat Z_i=Y_{0,i}+\widehat\Delta_i.
```

Required ablations:

| Variant | Purpose |
|---|---|
| `Y0` only | Tests whether the student just learns a coordinate-dependent bias. |
| `q` only | Tests whether physical landmark context alone carries repair signal. |
| `Y0 + q` | Main CF-HRD student. |
| Full-diffusion residual teacher | Later comparison to test whether cross-fitting matters. |

Completion standard for Stage 4:

Stage 4 is complete only when the code trains and exports the minimum residual
student family without using test labels during training, hyperparameter
selection, feature scaling, or model selection.

Required inputs:

```text
results/CF_HRD/seed10/cross_fitted_teacher/teacher_train.npz
results/CF_HRD/seed10/landmark_context/context.npz
results/heldout/seed10/adp_siamese/embedding_train.npz
results/heldout/seed10/adp_siamese/embedding_test.npz
```

Required training target:

```math
\Delta_i^\star
=
Z_i^{(-k(i))}-Y_{0,i},
\qquad i\in \mathrm{train}.
```

Required model variants:

| Variant | Input | Target | Required output |
|---|---|---|---|
| `y0_only` | `Y0_train` | `teacher_residual` | `delta_train_pred`, `delta_test_pred` |
| `q_only` | `q_train` | `teacher_residual` | `delta_train_pred`, `delta_test_pred` |
| `y0_q` | `[Y0_train, q_train]` | `teacher_residual` | `delta_train_pred`, `delta_test_pred` |

Model settings such as regression type, regularization strength, landmark count,
and feature transform must be fixed before looking at test localization errors.
If a validation split is needed, it must be carved out of the training split.

For every variant, final chart outputs are:

```math
\widehat Z_{\mathrm{train}}=Y_{0,\mathrm{train}}+\widehat\Delta_{\mathrm{train}},
```

```math
\widehat Z_{\mathrm{test}}=Y_{0,\mathrm{test}}+\widehat\Delta_{\mathrm{test}}.
```

Required Stage 4 diagnostic rule:

1. Report train residual fit to the teacher target.
2. Report correction magnitude summaries on train and test.
3. Save `Zhat_train` and `Zhat_test` for every variant.
4. Do not decide method success from Stage 4 alone; held-out success is judged
   by Stage 5.

Minimum expected output:

```text
results/CF_HRD/seed10/residual_student/result.json
results/CF_HRD/seed10/residual_student/<variant>_embedding_train.npz
results/CF_HRD/seed10/residual_student/<variant>_embedding_test.npz
```

The `result.json` file must include, for each variant:

| Field | Meaning |
|---|---|
| `train_residual_fit` | MSE/MAE between predicted and teacher residual on train. |
| `delta_norm_summary` | Magnitude of predicted correction on train/test. |
| `model_config` | Student model type, regularization, feature set, and fixed seed. |

Stage 4 readiness condition:

Stage 4 is ready for Stage 5 only if all three required variants export finite
train/test predictions with matching indices and no missing samples. A low train
residual error is not enough to claim success; it only shows that the student
can imitate the teacher on the training split.

Failure condition:

If `y0_q` cannot fit the teacher residual even on train, or if exported
corrections contain non-finite values or extreme correction magnitudes, stop
before Stage 5 and debug the student/features.

### Stage 5: held-out evaluation

For localization errors, fit affine alignment on train predictions only:

```math
A^\star=\arg\min_A \|A(\widehat Z_{\mathrm{train}})-P_{\mathrm{train}}\|^2.
```

Apply the same transform to test:

```math
\widehat P_{\mathrm{test}}=A^\star(\widehat Z_{\mathrm{test}}).
```

Report test metrics:

```text
mean / median / P90 / P95 / max localization error
CT / TW / KS
```

Stage 5 expected inputs:

```text
results/CF_HRD/seed10/residual_student/<variant>_embedding_train.npz
results/CF_HRD/seed10/residual_student/<variant>_embedding_test.npz
results/heldout/seed10/adp_siamese/embedding_train.npz
results/heldout/seed10/adp_siamese/embedding_test.npz
```

Compare against:

1. Held-out FCF test result.
2. Train-only diffusion diagnostic, explicitly marked as not a test result.
3. Full-data transductive diffusion, only as an upper-bound reference if used.
4. Direct full-diffusion residual distillation, if that comparison has been
   implemented.

## Pass/fail criteria

CF-HRD has a real validation signal only if:

1. The student improves held-out FCF test localization error, especially P90/P95
   or max, without degrading CT/TW/KS in a way that breaks chart quality.
2. `Y0 + q` is better than `Y0` only, showing that ADP/time physical landmark
   context contributes useful information.
3. The cross-fitted teacher is at least competitive with direct full-diffusion
   residual distillation on held-out test, supporting the no-self-anchor design.

CF-HRD should be downgraded or rejected if:

1. The student only improves train diagnostics but not held-out test.
2. Adding `q` gives no improvement over `Y0` only.
3. The correction reduces mean error while worsening the tail severely.
4. The method requires test positions, test-test graph edges, or full test graph
   optimization to work.

## Current status

Completed:

1. Stage 1 artifact loading and split checks.
2. Stage 2 cross-fitted no-self-anchor teacher generation.
3. Stage 3 physical landmark context generation for seed 10.
4. Stage 4 residual student training/export for seed 10.
5. Stage 5 held-out evaluation for seed 10.

Current seed 10 train diagnostic:

| Method | mean | P95 | max | CT | TW | KS |
|---|---:|---:|---:|---:|---:|---:|
| FCF train `Y0` | 0.5629 | 1.1875 | 7.0105 | 0.9926 | 0.9896 | 0.1034 |
| Diffusion-refined train, full anchor | 0.4122 | 0.8502 | 2.0319 | 0.9966 | 0.9939 | 0.0795 |
| CF-HRD cross-fitted teacher | 0.3977 | 0.8616 | 1.9614 | 0.9966 | 0.9938 | 0.0802 |

Interpretation:

The cross-fitted teacher is close to the full-anchor train diffusion result
despite removing each target node's self-anchor. This is sufficient to justify
the next validation stage, but it is not yet evidence that CF-HRD improves
held-out test performance.

The teacher script defaults to 200 Jacobi iterations, matching the existing
train diffusion diagnostic. A 100-to-200 iteration sensitivity check changed
the teacher train localization metrics only in the fourth decimal place.

Current Stage 3 seed 10 landmark context:

| Artifact | Value |
|---|---|
| Output | `results/CF_HRD/seed10/landmark_context/context.npz` |
| Landmarks | 128 train-only farthest-point landmarks |
| `q_train` | `(16798, 128)` |
| `q_test` | `(4199, 128)` |
| `gamma_train` | 13.316666603088379 |
| `sigma` median | 41.3807 |
| Test-position feature use | false |
| Test-test graph use | false |

Current Stage 4 seed 10 residual student:

| Variant | feature dim | train residual MSE | train residual MAE | relative mean norm |
|---|---:|---:|---:|---:|
| `y0_only` | 2 | 179.3092 | 9.9258 | 0.7796 |
| `q_only` | 128 | 127.2429 | 8.1075 | 0.6381 |
| `y0_q` | 130 | 83.6696 | 6.4440 | 0.5088 |

Stage 4 exported finite train/test predictions for all three required variants
under `results/CF_HRD/seed10/residual_student/`. These outputs are ready for
Stage 5 held-out evaluation. The Stage 4 numbers only measure imitation of the
cross-fitted teacher residual on train; they are not held-out localization
evidence.

Current Stage 5 seed 10 held-out evaluation:

| Method | mean | P90 | P95 | max | CT | TW | KS |
|---|---:|---:|---:|---:|---:|---:|---:|
| FCF | 0.5766 | 1.0126 | 1.2026 | 7.5254 | 0.9911 | 0.9884 | 0.1083 |
| `y0_only` | 0.5766 | 1.0126 | 1.2026 | 7.5254 | 0.9913 | 0.9884 | 0.1084 |
| `q_only` | 0.5136 | 0.8850 | 1.1012 | 7.6570 | 0.9900 | 0.9881 | 0.1013 |
| `y0_q` | 0.4871 | 0.8299 | 1.0211 | 7.2932 | 0.9917 | 0.9893 | 0.0991 |

Stage 5 verdict for seed 10:

`y0_q` improves FCF on mean, P90, P95, and max localization error, while CT and
TW increase and KS decreases. This is a positive held-out signal for CF-HRD on
the current split. It is not yet a final method claim because it has only been
verified on one seed.
