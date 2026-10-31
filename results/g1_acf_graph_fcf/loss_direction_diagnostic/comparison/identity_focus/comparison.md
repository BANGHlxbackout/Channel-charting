# Channel Chart comparison

Same-node transductive chart quality; GT used only for evaluation.

| method | ct | tw | ks | mean_m | median_m | p90_m | p95_m | max_m | source_json | Train time (s) | Peak VRAM (bytes) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| G1-Identity | 0.996966 | 0.996868 | 0.079401 | 0.398343 | 0.361716 | 0.713916 | 0.843197 | 1.429263 | [result.json](D:/diffusion_refined_charting/results/g1_acf_graph_fcf/stride1_n20997_seed10_identity_full/result.json) | 5292.062650499996 | 698310656 |
| G1-Identity/geo-step | 0.996976 | 0.996881 | 0.079327 | 0.398324 | 0.361987 | 0.712919 | 0.842530 | 1.428611 | [result.json](D:/diffusion_refined_charting/results/g1_acf_graph_fcf/loss_direction_diagnostic/identity_geo_step/result.json) | unavailable | unavailable |
| G1-Identity/new-step | 0.997000 | 0.996910 | 0.079202 | 0.397428 | 0.360852 | 0.712296 | 0.840959 | 1.407999 | [result.json](D:/diffusion_refined_charting/results/g1_acf_graph_fcf/loss_direction_diagnostic/identity_new_step/result.json) | unavailable | unavailable |
