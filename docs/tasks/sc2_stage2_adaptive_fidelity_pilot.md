# SC²-inspired Stage-2 adaptive-fidelity pilot

Status: approved to execute in isolated worktree only. No merge into master before user confirmation.
Branch: codex/sc2-adaptive-fidelity-pilot. Seed: 10. One bounded transductive pilot, with no parameter search.

## Question and boundary

Does a local second-order consistency score from the original D_geo and original Siamese Y0 identify where Stage-2 smoothing should trust Y0 more or less, and improve the fixed advisor baseline? Only the Y0 fidelity coefficient varies. Stage-1 graph, fused distance, D_geo, FCF, Stage-2 W, k=128, lambda=0.15, and 200 Jacobi iterations remain fixed. GT is used only for offline diagnostics and evaluation.

## Input and protocol audit

1. Read original geodesic.npy, original Y0 embedding.npz, distance_data.npz, and saved baseline Z embedding.npz from D:\diffusion_refined_charting. All D: access is read-only.
2. Check shape, finite values, and exact index order across all inputs. Check original geodesic top-512 cache shape/order and reconstruct W using the tracked baseline function; verify N, symmetry, nnz, and baseline graph semantics. Do not regenerate or write the cache to D:.
3. Record paths, file hashes, versions, exact command, N, graph nnz, and metric subset indices/seed. Stop dependent experiments and report any provenance conflict.

## Reliability computation

1. Directed N_i is the original top-64 geodesic neighborhood, excluding self. For those directed pairs, compute dY_ij and D_ij. If any local dY or D is nonpositive/nonfinite, report it before choosing a numerical remedy; never silently alter the log formula.
2. Compute log s as median(log dY - log D), r=abs(log dY-log(sD)), tau=median(r)+epsilon, C=exp(-r/tau). Record epsilon and all distribution summaries.
3. Create sparse symmetric C by C_directed.maximum(C_directed.T): union of local pairs. This preserves single-direction scores. Compute q_FO as mean of C_ij over each original directed N_i.
4. Compute H_ij as cosine similarity between symmetric sparse C rows, S_ij=C_ij H_ij, and q_SC2 as mean of S_ij over each original directed N_i. Save first_order_reliability.npy and sc2_reliability.npy. Do not construct a dense N-by-N compatibility or common-neighbor matrix.

## Diagnostic (does not gate refinement)

Use common_metrics.affine_transform separately on full-data Y0 and saved baseline Z, then point errors against GT. Rank q_SC2 into five equal-count groups with deterministic index tie-break; report counts, Y0 mean/median, baseline Z mean, and mean pointwise Y0-Z error difference. Report Spearman correlations of both q_SC2 and q_FO with Y0 error and with Y0-Z improvement. Plot SC2 quintile against Y0 error and improvement. This is an offline evaluation, not a learned GT rule.

## Four fixed variants

Use the exact same W, Y0, lambda and 200-step Jacobi update for all. Initialize Z^0=Y0. Percentile p_i=(rank_i-0.5)/N using full-data average ranks for ties, then a_i=0.5+p_i.

- uniform: a_i=1. This must reproduce saved baseline Z within documented numerical tolerance before adaptive outcomes are interpreted.
- first_order: percentile map of q_FO.
- sc2: percentile map of q_SC2.
- sc2_shuffled: permutation of the entire sc2 fidelity vector using NumPy default_rng(10), preserving its exact distribution.

## Evaluation and deliverables

Use the baseline seed-10 random.sample 20% subset for CT/TW/KS, identically across all variants. Use full-data affine alignment per variant for mean, median, P90, P95, max error and four-method CDF. Provide the two quintile plots and CDF. Save code under methods/sc2_stage2/ because the tracked baseline solver hardcodes a_i=1; keep original baseline code unchanged. Save machine outputs and a single REPORT.md under results/sc2_stage2_adaptive_fidelity_pilot/. Update experiment log/registry in the isolated worktree with the actual protocol and outcome. Compare SC2 with uniform, FO, and shuffled descriptively; do not tune based on this result or claim out-of-sample performance or novelty.

Stop after this pilot and report negative results as clearly as positive results. No Stage-1 modification, D_geo recomputation, FCF retraining, MDS gate, or follow-on sensitivity run.

## Uniform numerical reproduction audit (execution addendum)

The first CUDA run exceeded the provisional coordinate threshold of 1e-5. The tracked baseline solver itself differed from saved Z by max_abs 2.746582e-4, and two runs of the new uniform solver differed by max_abs 3.051758e-4. On the shared metric subset, CT/TW/KS and full-data affine error summaries matched the saved baseline to reporting precision. These independent checks support a max_abs <= 5e-4 numerical tolerance for the CUDA index_add_ run-to-run variation. The report must retain exact tracked/new/repeated differences and metric agreement; this addendum does not change lambda, iterations, W, fidelity formula, or research endpoints.
