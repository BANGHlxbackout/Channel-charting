# Method Summary
> LEGACY RESULT SNAPSHOT
>
> This file mixes results from different evaluation protocols
> (full/transductive, test split, stride settings, etc.).
> Do not rank methods across rows unless they share the same
> comparability group and evaluation protocol.
>
> Use `results/experiment_registry.md` for the current result index.
Localization errors are in meters. P2 uses the saved test-split metrics.

| method | CT | TW | KS | mean_m | median_m | p90_m | p95_m | max_m | source_json |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ADP-fused-MDS | 0.9845 | 0.9899 | 0.1063 | 0.5275 | 0.4003 | 0.8837 | 1.1303 | 9.1438 | results/adp_old_fused/stride1_n20997_seed10/result.json |
| ADP-Siamese| 0.9957 | 0.9960 | 0.0840 | 0.4093 | 0.3722 | 0.7142 | 0.8558 | 7.8936 | results/adp_old_siamese/stride1_n20997_seed10/result.json |
| | 0.9982 | 0.9982 | 0.0651 | 0.3187 | 0.2929 | 0.5587 | 0.6346 | 1.1414 | results/diffusion_refined_charting/stride1_n20997_k128_lambda0.15_seed10/result.json |
| Triplet | 0.9755 | 0.9596 | 0.2647 | 1.6477 | 1.3055 | 3.5279 | 4.1381 | 6.5759 | results/triplet_baseline_notebook/stride4_n20997_seed10/result.json |
| | 0.9955 | 0.9956 | 0.0804 | 0.3944 | 0.3672 | 0.6607 | 0.7645 | 8.4133 | results/euclidean_knn_refined_charting/stride1_n20997_euclidean_k128_lambda0.15_seed10/result.json |


| method | ct | tw | ks | mean_m | median_m | p90_m | p95_m | max_m | source_json |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ADP + timestamp fused + geodesic + metric MDS | 0.9845 | 0.9899 | 0.1063 | 0.5275 | 0.4003 | 0.8837 | 1.1303 | 9.1438 | results/adp_old_fused/stride1_n20997_seed10/result.json |
| ADP + timestamp fused + geodesic + Siamese FCF | 0.9957 | 0.9960 | 0.0840 | 0.4093 | 0.3722 | 0.7142 | 0.8558 | 7.8936 | results/adp_old_siamese/stride1_n20997_seed10/result.json |
| Graph-harmonic diffusion refinement of ADP/time geodesic channel charts | 0.9982 | 0.9982 | 0.0651 | 0.3187 | 0.2929 | 0.5587 | 0.6346 | 1.1414 | results/diffusion_refined_charting/stride1_n20997_k128_lambda0.15_seed10/result.json |
| Graph-harmonic diffusion refinement with Euclidean kNN graph on the initial chart | 0.9955 | 0.9956 | 0.0804 | 0.3944 | 0.3672 | 0.6607 | 0.7645 | 8.4133 | results/euclidean_knn_refined_charting/stride1_n20997_euclidean_k128_lambda0.15_seed10/result.json |
| P2 baseline (test split) | 0.9777 | 0.9612 | 0.2292 | 1.4700 | 1.3000 | 2.6200 | 3.1100 | 6.0500 | results/p2_baseline_shard4/p2_baseline_shard4_seed10-lambda_b1.0-lambda_box1.0-powtype0-Pthresh16.2-Mp3-Tc5-Mt0.8-1tperanch-bs100-lr0.001-100epochs-hidden832-fereim-schpar50_result.json |
| Triplet baseline from ChannelCharting.ipynb | 0.9755 | 0.9596 | 0.2647 | 1.6477 | 1.3055 | 3.5279 | 4.1381 | 6.5759 | results/triplet_baseline_notebook/stride4_n20997_seed10/result.json |
