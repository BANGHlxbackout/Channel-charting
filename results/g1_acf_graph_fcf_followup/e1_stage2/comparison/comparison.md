# Channel Chart comparison

Same-node transductive chart quality; GT used only for evaluation.

| method | ct | tw | ks | mean_m | median_m | p90_m | p95_m | max_m | source_json | Train time (s) | Peak VRAM (bytes) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| B0 | 0.995712 | 0.995987 | 0.084027 | 0.409347 | 0.372228 | 0.714152 | 0.855793 | 7.893630 | [result.json](D:/diffusion_refined_charting/results/adp_old_siamese/stride1_n20997_seed10/result.json) | unavailable | unavailable |
| G1-Identity | 0.996966 | 0.996868 | 0.079401 | 0.398343 | 0.361716 | 0.713916 | 0.843197 | 1.429263 | [result.json](D:/diffusion_refined_charting/results/g1_acf_graph_fcf/stride1_n20997_seed10_identity_full/result.json) | 5292.062650499996 | 698310656 |
| G1-Graph | 0.996947 | 0.996849 | 0.079340 | 0.399771 | 0.364578 | 0.714408 | 0.846337 | 1.463179 | [result.json](D:/diffusion_refined_charting/results/g1_acf_graph_fcf/stride1_n20997_seed10_graph_full/result.json) | 6224.643790000002 | 707461120 |
| B0+S2 | 0.998220 | 0.998223 | 0.065065 | 0.318748 | 0.292917 | 0.558675 | 0.634640 | 1.141432 | [result.json](D:/diffusion_refined_charting/results/g1_acf_graph_fcf_followup/e1_stage2/b0/result.json) | unavailable | 111771648 |
| G1-Identity+S2 | 0.998164 | 0.998159 | 0.066559 | 0.323004 | 0.296902 | 0.565467 | 0.648439 | 1.146562 | [result.json](D:/diffusion_refined_charting/results/g1_acf_graph_fcf_followup/e1_stage2/g1_identity/result.json) | unavailable | 111771648 |
| G1-Graph+S2 | 0.998152 | 0.998142 | 0.066692 | 0.324695 | 0.299163 | 0.565541 | 0.649174 | 1.132579 | [result.json](D:/diffusion_refined_charting/results/g1_acf_graph_fcf_followup/e1_stage2/g1_graph/result.json) | unavailable | 111771648 |
