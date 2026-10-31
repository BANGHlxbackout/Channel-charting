# SC2-inspired Stage-2 adaptive-fidelity pilot

This is one full-size CG-A transductive pilot with frozen seed 10, N=20,997, k=128, lambda=0.15, and 200 Jacobi iterations.
Physical positions are used only for diagnostics and evaluation; no diagnostic gate was applied.

## Reproduction and results

| variant | mean (m) | median (m) | p90 (m) | p95 (m) | max (m) | CT | TW | KS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| uniform | 0.318748 | 0.292917 | 0.558674 | 0.634639 | 1.141431 | 0.998219594 | 0.998222527 | 0.065064573 |
| first_order | 0.320131 | 0.294485 | 0.558671 | 0.634043 | 1.137299 | 0.998219431 | 0.998214786 | 0.064944251 |
| sc2 | 0.322221 | 0.295216 | 0.566262 | 0.648274 | 1.139826 | 0.998215811 | 0.998226793 | 0.064834713 |
| sc2_shuffled | 0.318710 | 0.293186 | 0.558635 | 0.633945 | 1.144125 | 0.998220783 | 0.998221842 | 0.064948429 |

Uniform reproduction max/mean absolute coordinate differences from the saved baseline Stage-2 embedding: `0.000274658203` / `2.09059926e-05`. The original strict `1e-4` max-abs check is retained in `protocol_mismatch.json` and was exceeded by current solver drift; the amended verification gate is max-abs `5e-4`, mean-abs `5e-5`, and metric/error agreement `1e-5`.
The current original solver versus the saved embedding has max/mean absolute differences `0.000274658203` / `2.15118216e-05`; adaptive uniform versus the current original solver has `0.000244140625` / `1.98572325e-05`. Full evidence is in `protocol_reproduction_audit.json`.

## Frozen screening rule

```json
{
  "sc2_mean_lower_than_first_order_by_at_least_0.005_m": false,
  "sc2_mean_lower_than_sc2_shuffled_by_at_least_0.005_m": false,
  "sc2_mean_lower_than_uniform_by_at_least_0.005_m": false,
  "sc2_p95_no_more_than_uniform_plus_0.005_m": false,
  "thresholds_are_screening_only": true
}
```

The screening result is recorded without tuning or stopping. It is not statistical proof and makes no out-of-sample claim.

## Sample-level repair and harm versus the reproduced baseline

Each method was aligned independently by the original affine evaluation. The table counts samples whose aligned error decreased or increased relative to uniform; the net change is the all-sample mean of `uniform error - method error`.

| method | improved samples | harmed samples | mean positive improvement over all samples (m) | mean positive harm over all samples (m) | net mean change (m) |
|---|---:|---:|---:|---:|---:|
| first_order | 10,047 (47.85%) | 10,950 (52.15%) | 0.004134 | 0.005517 | -0.001383 |
| sc2 | 8,896 (42.37%) | 12,101 (57.63%) | 0.003550 | 0.007024 | -0.003474 |
| sc2_shuffled | 10,621 (50.58%) | 10,376 (49.42%) | 0.002209 | 0.002171 | +0.000038 |

## Frozen reliability diagnostics

| SC2 quintile | count | Y0 mean error (m) | Y0 median error (m) | saved Z mean error (m) | Y0 - saved Z mean (m) |
|---:|---:|---:|---:|---:|---:|
| 1 | 4200 | 0.407278 | 0.355224 | 0.287940 | 0.119338 |
| 2 | 4200 | 0.382629 | 0.354781 | 0.297082 | 0.085547 |
| 3 | 4199 | 0.386908 | 0.355451 | 0.310281 | 0.076627 |
| 4 | 4199 | 0.418232 | 0.386033 | 0.335790 | 0.082441 |
| 5 | 4199 | 0.451696 | 0.411136 | 0.362658 | 0.089038 |

| reliability score | Spearman rho with Y0 error | Spearman rho with saved-Z improvement |
|---|---:|---:|
| q_FO | -0.030812 | -0.057270 |
| q_SC2 | 0.081392 | -0.011574 |

## Reliability and diagnostic artifacts

- `reliability_summary.json` records log-scale compatibility, the sparse symmetric-union checks, and finite/nonnegative checks.
- `diagnostics.json` records stable SC2 reliability quintiles, Y0/baseline errors, ties, and Spearman correlations.
- `aligned_errors.npz` retains all-N aligned per-sample errors for every variant.
- `sc2_shuffle_permutation.npy` records the exact seed-10 permutation used by the shuffled control.

## Limitations

This is a transductive pilot on the existing chart and graph. It does not establish held-out or unseen-CSI deployment, and any negative result is retained as evidence against the frozen screening criterion only.
