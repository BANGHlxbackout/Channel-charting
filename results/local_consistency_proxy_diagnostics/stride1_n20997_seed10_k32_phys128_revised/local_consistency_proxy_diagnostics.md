# Local Consistency Proxy Diagnostics

This is a proxy diagnostic only. UE positions are used only for post-hoc labels and metrics.

## Strict Gate Result

- Best protection candidate: `C_prot_minus_repair_z`
- Best protection gate pass: `False`
- Repair gate pass: `True`
- Repair best outlier AUC: `r_smoothness_residual` = 0.7544

## Protection Gate Details

| candidate | pass | AUC good | AUC strong worsened | rho(delta) | good AUC gain | strong AUC gain | top20 good enrich | top20 strong enrich | top20 outlier enrich |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| C_mutual | False | 0.5298 | 0.5008 | 0.0405 | 0.0028 | -0.0033 | 1.0736 | 0.9876 | 1.5475 |
| C_rank_weighted | False | 0.5263 | 0.5062 | 0.0378 | -0.0007 | 0.0021 | 1.0130 | 1.0021 | 2.0949 |
| C_multi_k_stability | False | 0.5178 | 0.4996 | 0.0197 | -0.0092 | -0.0045 | 1.0457 | 0.9842 | 1.9160 |
| C_prot | False | 0.5243 | 0.5019 | 0.0320 | -0.0027 | -0.0023 | 1.0093 | 0.9696 | 1.8528 |
| C_prot_anti | False | 0.5375 | 0.5254 | 0.1070 | 0.0105 | 0.0212 | 1.0979 | 1.0581 | 1.2001 |
| C_prot_minus_repair_z | False | 0.5470 | 0.5335 | 0.1377 | 0.0200 | 0.0294 | 1.0797 | 1.0044 | 1.5370 |

## Score Summary

| score | rho(init_error) | rho(error_delta) | AUC good | AUC strong worsened | AUC outlier |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw_A_nbr | 0.0119 | 0.0320 | 0.5270 | 0.5041 | 0.5881 |
| C_mutual | -0.0078 | 0.0405 | 0.5298 | 0.5008 | 0.5237 |
| C_rank_weighted | 0.0158 | 0.0378 | 0.5263 | 0.5062 | 0.5947 |
| C_multi_k_stability | 0.0134 | 0.0197 | 0.5178 | 0.4996 | 0.5767 |
| C_prot | 0.0080 | 0.0320 | 0.5243 | 0.5019 | 0.5652 |
| C_prot_anti | -0.0474 | 0.1070 | 0.5375 | 0.5254 | 0.4753 |
| C_prot_minus_repair_z | -0.0664 | 0.1377 | 0.5470 | 0.5335 | 0.4817 |
| B_scale_stress | 0.0894 | -0.1861 | 0.4680 | 0.4335 | 0.6418 |
| B_pos_scale_stress | 0.0954 | -0.1853 | 0.4665 | 0.4423 | 0.6390 |
| r_smoothness_residual | 0.2228 | -0.2320 | 0.3972 | 0.4909 | 0.7544 |
| d_move_diagnostic_only | 0.3787 | -0.2532 | 0.3750 | 0.5147 | 0.9188 |

## Interpretation

- No revised protection proxy passed the strict gate. Do not implement adaptive fidelity from these protection scores yet.
- Repair-needed / outlier-like regions still have usable signal from stress/residual proxies.
- Baseline displacement is kept diagnostic-only and is not included in the protection gate.

## Output Files

- `json`: `results\local_consistency_proxy_diagnostics\stride1_n20997_seed10_k32_phys128_revised\local_consistency_proxy_diagnostics.json`
- `markdown`: `results\local_consistency_proxy_diagnostics\stride1_n20997_seed10_k32_phys128_revised\local_consistency_proxy_diagnostics.md`
- `node_proxy_scores_csv`: `results\local_consistency_proxy_diagnostics\stride1_n20997_seed10_k32_phys128_revised\node_proxy_scores.csv`
- `plots_dir`: `results\local_consistency_proxy_diagnostics\stride1_n20997_seed10_k32_phys128_revised\plots`
