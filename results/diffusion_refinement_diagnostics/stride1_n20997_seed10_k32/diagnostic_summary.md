# Diffusion Refinement Diagnostics

This is a post-hoc diagnostic report. UE positions are used only for diagnosis, not for refinement.

## Key Metrics

| item | value |
| --- | ---: |
| improved fraction | 0.6825 |
| worsened fraction | 0.3175 |
| strongly worsened fraction (>0.05 m) | 0.2127 |
| mean error delta (refined - init) | -0.0906 m |
| median error delta | -0.0777 m |
| aligned displacement median | 0.2375 m |
| corr(displacement, error delta) | -0.6340 |
| true-neighbor overlap init -> refined | 0.3973 -> 0.6750 |
| physical-neighbor overlap init -> refined | 0.3448 -> 0.5189 |
| local compression median refined/init | 0.5940 |
| compressed fraction (<0.90) | 0.8896 |
| heavily compressed fraction (<0.75) | 0.7383 |

## Initial Error Slices

| slice | count | fraction | worsened frac | strong worsened frac | mean delta m | median compression |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| init_error_lt_0p2m | 4122 | 0.1963 | 0.7363 | 0.5437 | 0.0765 | 0.6087 |
| init_error_0p2m_to_0p5m | 10776 | 0.5132 | 0.3022 | 0.1874 | -0.0625 | 0.6249 |
| init_error_0p5m_to_1p0m | 5624 | 0.2678 | 0.0663 | 0.0363 | -0.2222 | 0.5353 |
| init_error_ge_1p0m | 475 | 0.0226 | 0.0063 | 0.0042 | -0.6191 | 0.3972 |

## Diagnostic Interpretation

- Diffusion refinement is strongly beneficial overall: most samples improve and the worst initial outliers are repaired.
- It is not a free improvement: a nontrivial fraction of samples worsens, especially points that were already accurate before refinement.
- Local physical-neighbor lengths are compressed after refinement, which is direct evidence of a smoothing/compression effect.
- Neighbor overlap with true local neighborhoods improves substantially, so the method is not simply harmful over-smoothing; it repairs topology for many points while damaging some already-good local geometry.
- A useful next module should therefore be selective: preserve already-good local regions and target outlier/low-consistency regions, rather than adding more global attraction.

## Output Files

- `node_csv`: `results\diffusion_refinement_diagnostics\stride1_n20997_seed10_k32\node_diagnostics.csv`
- `error_delta_hist`: `results\diffusion_refinement_diagnostics\stride1_n20997_seed10_k32\error_delta_hist.png`
- `displacement_vs_error_delta`: `results\diffusion_refinement_diagnostics\stride1_n20997_seed10_k32\displacement_vs_error_delta.png`
- `compression_vs_error_delta`: `results\diffusion_refinement_diagnostics\stride1_n20997_seed10_k32\compression_vs_error_delta.png`
- `ue_position_error_delta`: `results\diffusion_refinement_diagnostics\stride1_n20997_seed10_k32\ue_position_error_delta.png`
- `markdown_report`: `results\diffusion_refinement_diagnostics\stride1_n20997_seed10_k32\diagnostic_summary.md`
