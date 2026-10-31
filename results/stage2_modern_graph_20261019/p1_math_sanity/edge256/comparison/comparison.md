# Channel Chart comparison

Same-node transductive chart quality; GT used only for evaluation.

| method | ct | tw | ks | mean_m | median_m | p90_m | p95_m | max_m | source_json | Train time (s) | Peak VRAM (bytes) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| Observation | 0.976883 | 0.971311 | 0.094736 | 0.118798 | 0.108596 | 0.214585 | 0.239875 | 0.366090 | [result.json](D:/diffusion_refined_charting/results/stage2_modern_graph_20261019/p1_math_sanity/edge256/observation/result.json) | unavailable | unavailable |
| FNode-small | 0.986584 | 0.985965 | 0.068931 | 0.083867 | 0.073251 | 0.155839 | 0.181721 | 0.287430 | [result.json](D:/diffusion_refined_charting/results/stage2_modern_graph_20261019/p1_math_sanity/edge256/fnode/result.json) | unavailable | unavailable |
| FE-P-small-author [NONCONVERGED] | 0.977915 | 0.973787 | 0.092467 | 0.119030 | 0.108519 | 0.219350 | 0.239582 | 0.371124 | [result.json](D:/diffusion_refined_charting/results/stage2_modern_graph_20261019/p1_math_sanity/edge256/fe-p-small-author/result.json) | unavailable | unavailable |
| FE-P-small-prior [NONCONVERGED] | 0.976264 | 0.973581 | 0.095163 | 0.119003 | 0.108537 | 0.219182 | 0.240968 | 0.365585 | [result.json](D:/diffusion_refined_charting/results/stage2_modern_graph_20261019/p1_math_sanity/edge256/fe-p-small-prior/result.json) | unavailable | unavailable |
| DoT-P-small-author [NONCONVERGED] | 0.986171 | 0.985965 | 0.070523 | 0.083375 | 0.071315 | 0.150522 | 0.180639 | 0.304355 | [result.json](D:/diffusion_refined_charting/results/stage2_modern_graph_20261019/p1_math_sanity/edge256/dot-p-small-author/result.json) | unavailable | unavailable |
| DoT-P-small-prior [NONCONVERGED] | 0.985965 | 0.985759 | 0.070690 | 0.083422 | 0.071093 | 0.149938 | 0.181088 | 0.304827 | [result.json](D:/diffusion_refined_charting/results/stage2_modern_graph_20261019/p1_math_sanity/edge256/dot-p-small-prior/result.json) | unavailable | unavailable |
