# Channel Chart comparison

Same-node transductive chart quality; GT used only for evaluation.

| method | ct | tw | ks | mean_m | median_m | p90_m | p95_m | max_m | source_json | Train time (s) | Peak VRAM (bytes) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| B0 | 0.995712 | 0.995987 | 0.084027 | 0.409347 | 0.372228 | 0.714152 | 0.855793 | 7.893630 | [result.json](D:/diffusion_refined_charting/results/adp_old_siamese/stride1_n20997_seed10/result.json) | unavailable | unavailable |
| B0+S2-J | 0.998220 | 0.998223 | 0.065065 | 0.318748 | 0.292917 | 0.558675 | 0.634640 | 1.141431 | [result.json](D:/diffusion_refined_charting/results/stage2_modern_graph_20261019/p0_preflight/s2_j/result.json) | unavailable | unavailable |
| B0+S2-CG | 0.998220 | 0.998223 | 0.065065 | 0.318748 | 0.292917 | 0.558674 | 0.634639 | 1.141432 | [result.json](D:/diffusion_refined_charting/results/stage2_modern_graph_20261019/p0_preflight/s2_cg/result.json) | unavailable | unavailable |
