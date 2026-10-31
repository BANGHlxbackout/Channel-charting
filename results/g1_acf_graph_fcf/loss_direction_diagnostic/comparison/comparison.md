# Channel Chart comparison

Same-node transductive chart quality; GT used only for evaluation.

| method | ct | tw | ks | mean_m | median_m | p90_m | p95_m | max_m | source_json | Train time (s) | Peak VRAM (bytes) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| B0 | 0.995712 | 0.995987 | 0.084027 | 0.409347 | 0.372228 | 0.714152 | 0.855793 | 7.893630 | [result.json](D:/diffusion_refined_charting/results/adp_old_siamese/stride1_n20997_seed10/result.json) | unavailable | unavailable |
| G1-Identity | 0.996966 | 0.996868 | 0.079401 | 0.398343 | 0.361716 | 0.713916 | 0.843197 | 1.429263 | [result.json](D:/diffusion_refined_charting/results/g1_acf_graph_fcf/stride1_n20997_seed10_identity_full/result.json) | 5292.062650499996 | 698310656 |
| G1-Graph | 0.996947 | 0.996849 | 0.079340 | 0.399771 | 0.364578 | 0.714408 | 0.846337 | 1.463179 | [result.json](D:/diffusion_refined_charting/results/g1_acf_graph_fcf/stride1_n20997_seed10_graph_full/result.json) | 6224.643790000002 | 707461120 |
| G1-Identity/geo-step | 0.996976 | 0.996881 | 0.079327 | 0.398324 | 0.361987 | 0.712919 | 0.842530 | 1.428611 | [result.json](D:/diffusion_refined_charting/results/g1_acf_graph_fcf/loss_direction_diagnostic/identity_geo_step/result.json) | unavailable | unavailable |
| G1-Graph/geo-step | 0.996959 | 0.996863 | 0.079259 | 0.399704 | 0.365140 | 0.713207 | 0.845223 | 1.460867 | [result.json](D:/diffusion_refined_charting/results/g1_acf_graph_fcf/loss_direction_diagnostic/graph_geo_step/result.json) | unavailable | unavailable |
| G1-Identity/diff-step | 0.997001 | 0.996912 | 0.079211 | 0.397239 | 0.361124 | 0.712484 | 0.840558 | 1.402385 | [result.json](D:/diffusion_refined_charting/results/g1_acf_graph_fcf/loss_direction_diagnostic/identity_diff_step/result.json) | unavailable | unavailable |
| G1-Graph/diff-step | 0.996982 | 0.996894 | 0.079159 | 0.398712 | 0.363619 | 0.713783 | 0.843205 | 1.432677 | [result.json](D:/diffusion_refined_charting/results/g1_acf_graph_fcf/loss_direction_diagnostic/graph_diff_step/result.json) | unavailable | unavailable |
| G1-Identity/new-step | 0.997000 | 0.996910 | 0.079202 | 0.397428 | 0.360852 | 0.712296 | 0.840959 | 1.407999 | [result.json](D:/diffusion_refined_charting/results/g1_acf_graph_fcf/loss_direction_diagnostic/identity_new_step/result.json) | unavailable | unavailable |
| G1-Graph/new-step | 0.996982 | 0.996895 | 0.079146 | 0.398884 | 0.364134 | 0.713734 | 0.843249 | 1.438673 | [result.json](D:/diffusion_refined_charting/results/g1_acf_graph_fcf/loss_direction_diagnostic/graph_new_step/result.json) | unavailable | unavailable |
