# Channel Chart comparison

Same-node transductive chart quality; GT used only for evaluation.

| method | ct | tw | ks | mean_m | median_m | p90_m | p95_m | max_m | source_json | Train time (s) | Peak VRAM (bytes) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| G1-Graph | 0.996947 | 0.996849 | 0.079340 | 0.399771 | 0.364578 | 0.714408 | 0.846337 | 1.463179 | [result.json](D:/diffusion_refined_charting/results/g1_acf_graph_fcf/stride1_n20997_seed10_graph_full/result.json) | 6224.643790000002 | 707461120 |
| G1-Graph/geo-step | 0.996959 | 0.996863 | 0.079259 | 0.399704 | 0.365140 | 0.713207 | 0.845223 | 1.460867 | [result.json](D:/diffusion_refined_charting/results/g1_acf_graph_fcf/loss_direction_diagnostic/graph_geo_step/result.json) | unavailable | unavailable |
| G1-Graph/new-step | 0.996982 | 0.996895 | 0.079146 | 0.398884 | 0.364134 | 0.713734 | 0.843249 | 1.438673 | [result.json](D:/diffusion_refined_charting/results/g1_acf_graph_fcf/loss_direction_diagnostic/graph_new_step/result.json) | unavailable | unavailable |
