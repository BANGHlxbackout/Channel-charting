# Channel Chart comparison

Same-node transductive chart quality; GT used only for evaluation.

| method | ct | tw | ks | mean_m | median_m | p90_m | p95_m | max_m | source_json | Train time (s) | Peak VRAM (bytes) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| Observation | 0.991667 | 0.983333 | 0.094605 | 0.137966 | 0.131623 | 0.247844 | 0.265496 | 0.297015 | [result.json](D:/diffusion_refined_charting/results/stage2_modern_graph_20261019/p1_math_sanity/framelet/n64/observation/result.json) | unavailable | unavailable |
| FN-P | 0.975000 | 0.975000 | 0.068143 | 0.093414 | 0.082894 | 0.173940 | 0.191803 | 0.214841 | [result.json](D:/diffusion_refined_charting/results/stage2_modern_graph_20261019/p1_math_sanity/framelet/n64/fn-p/result.json) | unavailable | unavailable |
| FN-G-nu | 0.991667 | 0.983333 | 0.068406 | 0.100241 | 0.090515 | 0.185780 | 0.205269 | 0.237373 | [result.json](D:/diffusion_refined_charting/results/stage2_modern_graph_20261019/p1_math_sanity/framelet/n64/fn-g-nu/result.json) | unavailable | unavailable |
| FN-G-delta | 0.991667 | 0.983333 | 0.062860 | 0.092184 | 0.078476 | 0.168769 | 0.191302 | 0.222053 | [result.json](D:/diffusion_refined_charting/results/stage2_modern_graph_20261019/p1_math_sanity/framelet/n64/fn-g-delta/result.json) | unavailable | unavailable |
