# P2A Channel Chart comparison

Same fixed 4199-node metric subset; all-N affine errors; GT evaluation only.

| method | ct | tw | ks | mean_m | median_m | p90_m | p95_m | max_m | source_json | train_time_s | peak_vram_bytes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| B0 | 0.995712 | 0.995987 | 0.084027 | 0.409347 | 0.372228 | 0.714152 | 0.855793 | 7.893630 | [JSON](D:/diffusion_refined_charting/results/stage2_modern_graph_20261019/p2a_b0_ic_fulln/references/b0/result.json) | unavailable | unavailable |
| B0+S2-J | 0.998220 | 0.998223 | 0.065065 | 0.318748 | 0.292917 | 0.558675 | 0.634640 | 1.141431 | [JSON](D:/diffusion_refined_charting/results/stage2_modern_graph_20261019/p2a_b0_ic_fulln/references/s2_j/result.json) | unavailable | unavailable |
| B0+S2-CG | 0.998220 | 0.998223 | 0.065065 | 0.318748 | 0.292917 | 0.558674 | 0.634639 | 1.141432 | [JSON](D:/diffusion_refined_charting/results/stage2_modern_graph_20261019/p2a_b0_ic_fulln/references/s2_cg/result.json) | unavailable | unavailable |
| IC8-P a=0.5 eta=0.5d0 | 0.997927 | 0.997955 | 0.070276 | 0.347487 | 0.319950 | 0.615430 | 0.734243 | 2.816398 | [JSON](D:/diffusion_refined_charting/results/stage2_modern_graph_20261019/p2a_b0_ic_fulln/ic8-p_a50_eta050/result.json) | unavailable | 468262912 |
| IC9-P a=0.5 eta=0.5d0 | 0.997849 | 0.997864 | 0.073810 | 0.367350 | 0.338916 | 0.658345 | 0.794416 | 2.174598 | [JSON](D:/diffusion_refined_charting/results/stage2_modern_graph_20261019/p2a_b0_ic_fulln/ic9-p_a50_eta050/result.json) | unavailable | 418604032 |
| IC9-G a=0.5 eta=0.5d0 | 0.997849 | 0.997863 | 0.073812 | 0.367363 | 0.338964 | 0.658190 | 0.794415 | 2.170456 | [JSON](D:/diffusion_refined_charting/results/stage2_modern_graph_20261019/p2a_b0_ic_fulln/ic9-g_a50_eta050/result.json) | unavailable | 393438208 |
