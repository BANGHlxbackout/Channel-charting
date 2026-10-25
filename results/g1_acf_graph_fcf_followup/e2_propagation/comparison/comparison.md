# Channel Chart comparison

Same-node transductive chart quality; GT used only for evaluation.

| method | ct | tw | ks | mean_m | median_m | p90_m | p95_m | max_m | source_json | Train time (s) | Peak VRAM (bytes) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| Graph ckpt @ Graph | 0.996947 | 0.996849 | 0.079340 | 0.399771 | 0.364578 | 0.714408 | 0.846337 | 1.463179 | [result.json](D:/diffusion_refined_charting/results/g1_acf_graph_fcf_followup/e2_propagation/graph_at_graph/result.json) | unavailable | 400580096 |
| Graph ckpt @ I | 0.991064 | 0.990633 | 0.099622 | 0.491343 | 0.438520 | 0.872250 | 1.025123 | 8.185935 | [result.json](D:/diffusion_refined_charting/results/g1_acf_graph_fcf_followup/e2_propagation/graph_at_identity/result.json) | unavailable | 400580096 |
| Identity ckpt @ I | 0.996966 | 0.996868 | 0.079401 | 0.398343 | 0.361716 | 0.713916 | 0.843197 | 1.429263 | [result.json](D:/diffusion_refined_charting/results/g1_acf_graph_fcf_followup/e2_propagation/identity_at_identity/result.json) | unavailable | 400580096 |
