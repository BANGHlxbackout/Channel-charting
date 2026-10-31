# Local Consistency Proxy Diagnostics

This report is diagnostic only. UE positions are used only to create post-hoc labels.

## Hard Gate

- All-score diagnostic gate: `True`
- Method-eligible proxy gate: `True`
- Protection proxy gate: `True`
- Repair proxy gate: `True`
- All-score best AUC evidence: `d_move_diagnostic_only:auc_outlier_initial_best_direction` = 0.9188
- Method-eligible best AUC evidence: `r_smoothness_residual:auc_outlier_initial_best_direction` = 0.7544
- Method-eligible best |Spearman(error_delta)| evidence: `r_smoothness_residual` = 0.2320

Gate rule:

```text
AUC >= 0.60 or |Spearman(error_delta)| >= 0.10 with interpretable direction.
```

## Score Summary

| score | rho(init_error) | rho(error_delta) | AUC strong worsened | AUC good initial | AUC outlier initial |
| --- | ---: | ---: | ---: | ---: | ---: |
| A_nbr_agreement | 0.0119 | 0.0320 | 0.5041 | 0.5270 | 0.5881 |
| B_scale_stress | 0.0894 | -0.1861 | 0.4335 | 0.4680 | 0.6418 |
| B_pos_scale_stress | 0.0954 | -0.1853 | 0.4423 | 0.4665 | 0.6390 |
| r_smoothness_residual | 0.2228 | -0.2320 | 0.4909 | 0.3972 | 0.7544 |
| d_move_diagnostic_only | 0.3787 | -0.2532 | 0.5147 | 0.3750 | 0.9188 |
| protect_A_z | 0.0119 | 0.0320 | 0.5041 | 0.5270 | 0.5881 |
| protect_A_minus_Bpos_z | -0.0484 | 0.1254 | 0.5374 | 0.5368 | 0.4848 |
| repair_need_minusA_plus_Bpos_z | 0.0484 | -0.1254 | 0.4626 | 0.4632 | 0.5152 |
| repair_need_minusA_plus_Bpos_plus_0p25r_z | 0.0748 | -0.1475 | 0.4642 | 0.4519 | 0.5472 |

## Interpretation

- At least one score has diagnostic signal, but the all-score gate can be dominated by baseline movement.
- Baseline movement remains diagnostic-only and should not be used as a first-version method input.
- Protection proxies have some signal, but inspect AUC values before using them for adaptive fidelity.
- Repair proxies pass the gate, meaning the current scores are better at finding repair-needed/outlier regions than protecting already-good regions.

## Output Files

- `json`: `results\local_consistency_proxy_diagnostics\stride1_n20997_seed10_k32_phys128\local_consistency_proxy_diagnostics.json`
- `markdown`: `results\local_consistency_proxy_diagnostics\stride1_n20997_seed10_k32_phys128\local_consistency_proxy_diagnostics.md`
- `node_proxy_scores_csv`: `results\local_consistency_proxy_diagnostics\stride1_n20997_seed10_k32_phys128\node_proxy_scores.csv`
- `plots_dir`: `results\local_consistency_proxy_diagnostics\stride1_n20997_seed10_k32_phys128\plots`
