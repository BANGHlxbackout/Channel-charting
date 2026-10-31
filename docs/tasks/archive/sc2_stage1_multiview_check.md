# SC²-inspired Stage-1 multi-view relation check (frozen before execution)

**执行后范围纠正（2026-10-08）：** 下文保留原冻结方案，供追溯；其中 `1-|d_A-d_B|/d_full` 实际测的是两个数是否接近，不能代表“两组视角都认为这对测量相近”，也不是 SC²-PCR 的第三方共同支持。该诊断只能对这个具体数值一致性评分给出负结果，原先的问题仍待设计。不得事后把失败扩大为多视角或 SC² 思路失败。

## Question

The previous Stage-1 pilot penalized edges with few common neighbors and failed. This task tests a narrower relation-level hypothesis: among existing Stage-1 k=20 relations that are dominated by ADP and span more than 60 seconds, does agreement between two fixed, disjoint antenna-array views identify physically close pairs more reliably than the original full-array ADP distance? This is a diagnostic of relation quality only. It does not modify the Stage-1 graph or produce a new chart.

## Fixed inputs and semantics

- Dataset: read `datasets/adp_old_shard4/{H.npy,timestamps.npy,UE_pos.npy}` directly with the existing shape/order semantics; assert `H` is `(N,32,13)`, reshape to `(N,4,8,13)`, and assert timestamp/position rows match the stride-1 sample order.
- Sample order: stride 1, original N=20,997 rows; preserve the order in `distance_data.npz` and assert `indices == arange(N)`.
- Stage-1 graph: `results/phase2c_h1_h2/stage1_graph.npz`, the verified directed k=20 graph. Use its original outgoing relations and saved `fused.npy`; do not use `geodesic_top512.npy`.
- Full ADP and fused-distance semantics remain those in `run_adp_fused.py`: normalize each antenna vector after reshaping `(N, arrays, antennas, taps)` to `(N, arrays*taps, antennas)`, then `D_ADP(i,j)=sum_l(1-|<z_i^l,z_j^l>|^2)`, and `D_fused=min(D_ADP, gamma*|dt|)` with saved baseline `gamma=12.983333587646484`.
- The two fixed views are array blocks A={0,2} and B={1,3}. Each view uses the same normalization and ADP formula restricted to its two arrays; no view, threshold, or weight is selected using GT.

## Fixed candidate relations

1. Read every directed k=20 Stage-1 relation `(i,j)` and collapse reciprocal entries to one unordered pair `(min(i,j), max(i,j))`, retaining each pair once.
2. Keep a pair only when `|timestamp_i-timestamp_j| > 60 s` and the saved full ADP branch is active: `D_ADP(i,j) <= gamma*|dt|`. The branch condition is evaluated from the saved original full ADP/gamma, not from a new fitted rule.
3. The candidate set is frozen before any GT statistics are read. The audit count is 23,973 unique pairs (36,811 directed qualifying relations before collapse). Report count, time-gap range, and full-ADP range in the result.

## Relation scores and controls

For every candidate, calculate `d_full`, `d_A`, and `d_B`. First verify on a fixed, GT-free sample that the view calculation is consistent with the full calculation (`d_A+d_B` equals `d_full` within numerical tolerance). The primary view statistic is fixed-total disagreement:

```text
disagreement = abs(d_A - d_B) / (d_full + 1e-12)
agreement = 1 - disagreement
```

This separates “both views are small” from “the two numbers merely match”: all comparisons condition on original `d_full`; the combined descriptive check also reports the two view magnitudes. No new graph edge weight is created.

The primary comparator is original `d_full`. The negative control is `agreement_shuffled`, obtained by a deterministic seed-10 permutation within predeclared strata of full-ADP decile × time-gap quartile × endpoint local-density tertile. The permutation preserves agreement distribution and candidate composition while removing relation identity. A secondary fixed, untuned ranking combines the two pieces as `rank(-d_full) + rank(agreement)`; it is reported only as an incremental diagnostic and is not an intervention.

## Offline labels and stratification

After candidate selection and scoring are frozen, use `UE_pos.npy` only for evaluation labels: a true-near relation is physical distance `<0.25 m`; a clearly false relation is `>0.5 m`. These cutoffs are prospectively fixed for this diagnostic; they were only posthoc descriptive cuts in the previous pilot and are not inherited success gates. Stratify by full-ADP decile, time-gap quartile, and endpoint local-density tertile, where density is the mean saved fused length of each endpoint's original outgoing k=20 edges. Require at least 30 candidates in each compared subgroup; otherwise mark that stratum insufficient rather than merging it after seeing results.

## Primary endpoints and acceptance rule

Within fixed strata, compare top and bottom agreement quartiles and report the sample-weighted true-near fraction difference, with a 200-resample endpoint-cluster bootstrap 95% CI (seed 11). Also report ROC AUC for full-ADP, agreement, fixed combined, and shuffled combined rankings using only the unambiguous subset (`physical <0.25 m` positive versus `physical >0.5 m` negative; middle distances excluded). Separately report precision at fixed top 5% of all candidates, where middle distances remain in the denominator and only `<0.25 m` counts as a hit; report the `physical >0.5 m` fraction beside every precision result. Report the number and fraction of candidates covered by eligible strata; do not regroup sparse strata after seeing labels. The endpoint bootstrap resamples sample IDs with replacement and weights each fixed candidate by the product of its two endpoint multiplicities, so relation clustering is represented explicitly; if a replicate has zero mass in a group, omit that replicate and report its count. The agreement quintile table and Spearman correlations are posthoc descriptive checks only; they cannot change the frozen decision.

The relation-level hypothesis is supported only if, before looking at labels, all of the following hold: (1) agreement top-vs-bottom true-near difference is at least 5 percentage points and its bootstrap lower bound is above zero; (2) fixed combined ranking improves AUC over full ADP by at least 0.02 and over shuffled control by at least 0.02; and (3) combined top-5% precision is no worse than full ADP. If any primary condition fails, do not modify Stage-1 or Stage-2 based on this diagnostic.

## Leakage and scope constraints

Ground-truth positions are loaded only after candidate set, scores, shuffle, strata, and endpoints are fixed. They are used only for offline labels, confidence intervals, examples, and plots. No GT enters graph construction, scores, permutations, thresholds, training, inference, or adaptive choices. This is a transductive relation diagnostic on the existing sample set; it makes no held-out or deployment claim.

## Budget, artifacts, and stopping

Compute only candidate-pair view distances; do not allocate an N-by-N view matrix. Do not recompute D_geo, run MDS, train FCF, or run Stage 2. Save frozen config, candidate table, score/evaluation JSON, small example table, and one `REPORT.md` under `results/sc2_stage1_multiview_check/`; put the minimal diagnostic entry point and tests under `diagnostics/sc2_stage1_multiview_check/` and `tests/`. Stop after this relation test. A positive result earns a separately approved intervention design; it is not itself evidence of a better final chart.
