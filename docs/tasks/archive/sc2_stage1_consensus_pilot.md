# SC2-inspired Stage-1 consensus pilot (frozen before runs)

- Date: 2026-09-25. Status: completed, negative at the pre-FCF MDS gate; see `results/sc2_stage1_consensus_pilot/REPORT.md`. User steering before FCF training: evaluate new D_geo first, including the repository's metric MDS, and train FCF only after a positive independent MDS gate.
- Post-result interpretation correction (2026-09-25): the low-common-neighbor edge-length penalty was an unverified design assumption, not SC2-PCR's intervention. The result tests only that penalty and cannot decide whether third-party support can improve Channel Charting when used in a suitable compatibility/consensus mechanism.
- Question: Can common-neighbor support on the original Stage-1 20-NN graph improve the final channel chart, beyond an equal overall edge-length increase or a shuffled support assignment?
- Scope: one full-size CG-A transductive run, N=20,997, seed=10. This is a new SC2-inspired study, not a continuation or revision of the frozen Phase-2C H1 counterfactual.

## Inputs and graph semantics

- Reuse `results/phase2c_h1_h2/stage1_graph.npz`, the verified directed weighted CSR, and `results/adp_old_fused/stride1_n20997_seed10/{fused.npy,geodesic.npy,distance_data.npz}`. Assert shape, sample indices, exactly 20 non-self outgoing edges and connectedness. Preserve directed CSR structure and original edge data; shortest paths use `directed=False` exactly as in the baseline. Do not use `geodesic_top512.npy` as the Stage-1 graph.
- Baseline Y0 and Z records: `results/adp_old_siamese/stride1_n20997_seed10` and `results/diffusion_refined_charting/stride1_n20997_k128_lambda0.15_seed10`. Re-run the baseline training in this study for a same-environment reference; do not overwrite retained outputs.
- GT positions are evaluation-only. No GT enters scores, weights, shortest paths, training targets, or selection.

## One intervention

For scoring only, form a binary undirected union A of the original directed graph, with zero diagonal. For each original directed edge (i,j), let c_ij=(A^2)_ij, excluding i,j automatically by the zero diagonal. Set s_ij=c_ij/max(1,min(deg_i-1,deg_j-1)); clip to [0,1]. This is the SC2 common-support principle adapted to a sample-neighbor graph, not the original rigid-registration compatibility measure. Do not claim independent witnesses or a probability of correctness.

Define length multiplier m_ij=1+0.5*(1-s_ij). The treatment graph has original directed edge support and edge data w'_ij=w_ij*m_ij. No deletion, new edges, shortest-path symmetrization, H1 ranking, parameter scan, or GT-based tuning. The 0.5 is a frozen moderate perturbation for this pilot, not an optimized value.

## Controls and runs

1. `baseline`: original graph, original full Stage-1 training schedule and Stage-2 baseline configuration.
2. `uniform`: multiply every directed edge length by a single factor such that its total edge-weight sum equals the treatment's sum. This controls a global increase in geodesic scale.
3. `shuffled`: assign the treatment multipliers to other directed edges by a seed-10 permutation within original edge-length deciles and undirected endpoint-degree bins, preserving as much of the multiplier distribution and distance/density strata as possible. Record exact binning and distribution checks. If a bin has fewer than two edges, keep its assignments unchanged and report the affected fraction.
4. `consensus`: treatment graph as above.

For all four graphs, compute full `D_geo` by the original sparse shortest-path semantics. Assert finite distances, zero diagonal, symmetry to numerical tolerance, and original graph's sampled `D_geo` reproduction. Save matrices and graph summaries under `results/sc2_stage1_consensus_pilot/`, never over existing files. Then evaluate Stage-1 distances using the gate below. Only if its frozen positive criterion passes, train the existing `run_adp_siamese.py` with original full schedule, dataset `datasets/adp_old_shard4`, seed=10, and each generated distance-run directory, followed by existing `run_diffusion_refined_charting.py` with k=128, lambda=0.15, iterations=200, top-k=512, seed=10 on each matching Y0 and D_geo. Every variant must use identical sample indices, evaluation subset, architecture, schedule and metric code. If the gate fails, stop before FCF, report the distance results, and retain negative evidence. If a hardware/runtime issue prevents a full run, mark it incomplete rather than ranking abbreviated runs against the retained baseline.

## Frozen independent D_geo evaluation and MDS gate (added on user instruction before FCF runs)

- For the primary pre-FCF gate, take exactly `mds_indices=np.arange(0,20997,10)` (2,100 samples) for all four *full-graph* D_geo matrices. Call the existing `run_adp_fused.run_mds` on each fixed submatrix with seed=10, max_iter=80 and n_init=4, exactly matching the original MDS implementation's settings apart from the fixed reduction to control resource use. Compare affine-aligned mean/p95 error and CT/TW/KS on these same positions and sample indices. GT is evaluation-only, not an MDS input. Save MDS embeddings and per-variant results. This fixed-subset chart proxy is not a full FCF result or a full-N MDS result.
- MDS gate positive iff consensus aligned mean error is at least 0.005 m lower than all three controls, p95 is no more than 0.01 m above baseline, CT/TW are no more than 0.005 below baseline, and KS is no more than 0.01 above baseline. All thresholds are fixed before seeing MDS results; failure means no FCF training. This pilot gate tests whether altered D_geo yields a better 2-D chart through a non-neural mapping, without claiming that MDS and FCF are equivalent.

- Sample one fixed set of 100,000 unordered calibration pairs, 50,000 uniform test pairs, and 50,000 local test pairs using `np.random.default_rng(20260925)`. Reject self-pairs and duplicate unordered pairs across all three sets. Uniform pairs sample distinct point IDs uniformly. Local test pairs sample a source uniformly and its target uniformly from the saved original `geodesic_top512.npy` first 128 columns; this is a sampling frame only, not a Stage-1 graph. Record pair indices or exact generator metadata. Do not select pairs or thresholds using GT or candidate results.
- For each variant separately, fit monotone isotonic regression from its `D_geo(i,j)` to physical Euclidean distance on the same calibration pairs, then score the held-out sets. Calibration is evaluation-only and does not enter graph construction or FCF training. Per-variant recalibration intentionally removes trivial global distance rescaling as an explanation. Report uniform and local held-out MAE, p95 absolute residual, signed under/over-estimation, and Spearman rank correlation. Distinguish this physical-distance proxy from the latent geodesic training target: it is a diagnostic, not proof of chart quality.
- The isotonic pair test is a secondary diagnostic, not an additional train/no-train gate. It helps explain MDS results and catch localized deterioration. Do not relax the MDS gate or substitute this diagnostic as the deciding endpoint after observing results.
- The prior H1 frozen metrics and labels remain untouched. This new gate asks whether the SC2-derived intervention improves the *whole* Stage-1 distance matrix versus controls, including collateral harm, before costly FCF training.

## Frozen readout and decision

- Conditional primary endpoint after the independent D_geo gate passes: final Z aligned mean localization error (m). Compare consensus with all three same-run controls. Call the full pilot positive only if consensus improves Z mean by at least 0.005 m versus every control, with Z p95 not worse than baseline by more than 0.005 m. This is a screening threshold, not statistical proof.
- Secondary: Y0 mean/p95, Z median/p90/max, CT/TW/KS on the same metric subset, number of edges near zero support, multiplier distribution, component counts and sampled geodesic change. Report per-sample aligned error improvements and harms separately; do not infer OOS performance.
- Interpretation guardrails: Uniform matching performance means general length rescaling is sufficient. Shuffled matching performance means correct support assignment is unproven. Any baseline replication discrepancy must be disclosed; do not silently compare incompatible metrics. A negative result is retained without tuning the coefficient or changing endpoints.
- Budget: one coefficient, one seed, four graph/distance runs and one fixed-subset MDS gate; only on gate success, four training/refinement runs. No full-N MDS, no H2, no full predecessor matrix, no extra seeds or broad parameter sweep. Results are screening evidence only.

## Code and records

- Reuse `run_adp_siamese.py`, `run_diffusion_refined_charting.py`, `common_metrics.py`, and the Stage-1 graph artifact. A single new entry under `methods/sc2_stage1/` is justified because no existing entry computes common-neighbor scores and builds these controlled variants. Keep config and artifacts in the one result family; write one `REPORT.md`. On completion, append one experiment to `docs/research/experiment_log.md` and register retained results in `docs/research/experiment_registry.md`. Do not change `research_state.md` without a user decision.
