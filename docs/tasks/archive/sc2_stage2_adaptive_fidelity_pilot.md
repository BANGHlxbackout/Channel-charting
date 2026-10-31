# SC2-inspired Stage-2 adaptive-fidelity pilot

- Date: 2026-09-25. Status: completed; negative under the frozen screening rule. Results: `results/sc2_stage2_adaptive_fidelity_pilot/REPORT.md`.
- Question: Does second-order local agreement identify which existing `Y0` samples should be held more strongly during the advisor Stage-2 refinement, beyond first-order agreement and a shuffled assignment?
- Scope: one full-size CG-A transductive pilot, N=20,997, seed=10. This is distinct from the failed Stage-1 edge-length pilot. No Stage-1 graph/distance/FCF modification, no Stage-2 graph-weight modification, no MDS gate, no parameter sweep.

## Frozen inputs and protocol

- `results/adp_old_fused/stride1_n20997_seed10/geodesic.npy`, the corresponding `geodesic_top512.npy` cache, and `results/adp_old_siamese/stride1_n20997_seed10/embedding.npz` must have identical sample order and N=20,997. Validate cache shape, unique non-self neighbors, ascending distances, and sampled top-k distance boundaries; permit index ties at the boundary rather than requiring a unique ordering. Use the saved original Stage-2 embedding/JSON for reference; reproduce uniform in the new solver.
- Use the original `build_local_scaled_graph` from `run_diffusion_refined_charting.py`, with the same D_geo, k=128, lambda=0.15, 200 Jacobi iterations, float32 data and device. Keep `W` exactly identical for all four variants. Use `z^(0)=Y0`.
- Ground-truth positions are evaluation-only. They cannot enter neighbor selection, scale, compatibility, reliability, percentile mapping, fidelity, solve, or method selection.

## Local compatibility and reliability

- For each i, use the original directed top-64 geodesic neighbors `N_i`; exclude self and validate the cache ordering. Compute dY_ij=||Y0_i-Y0_j|| and D_ij=D_geo[i,j]. Use a fixed epsilon `1e-8` in all logarithms/denominators. Set `log_s=median(log(dY+eps)-log(D+eps))` over all 64N directed pairs, `r_ij=abs(log(dY+eps)-log_s-log(D+eps))`, `tau=median(r)+eps` and `C_ij=exp(-r_ij/tau)`.
- For computing second-order support, form the sparse symmetric *union* of the directed top-64 relations. Where only one direction is present, use its C value; where both are present, the same symmetric pairwise formula gives the same value. Set the diagonal to zero. Do not add any other edges or change the Stage-2 W. Report max directional disagreement and finite/nonnegative checks.
- First-order node reliability: `q_FO[i]=mean(C[i,j] for j in original directed N_i)`. Second-order edge score on each original directed (i,j): `H_ij=(C_i dot C_j)/(||C_i||_2 ||C_j||_2 + eps)`, using the symmetric sparse C over all witness k, and `S_ij=C_ij*H_ij`. Second-order node reliability: `q_SC2[i]=mean(S_ij for j in original directed N_i)`. Do not form a dense N-by-N matrix or the full second-order product. Save `first_order_reliability.npy`, `sc2_reliability.npy` and score summaries.
- Convert each complete node score vector to percentile by `(scipy.stats.rankdata(q,method='average')-1)/(N-1)`. Tied values receive average ranks. Let `a=0.5+p`; thus 0.5<=a<=1.5 and mean(a)=1 (up to rounding). Do not reverse this direction after observing diagnostics.

## Diagnostic (always continue to all variants)

- Affine-align Y0 and saved baseline Z independently to physical positions with the existing `common_metrics.affine_transform`. Use aligned per-sample errors for diagnostics only. Divide nodes into five equal-count bins by q_SC2 using a stable sort by (q,index), and report Y0 mean/median error, baseline Z mean error, and mean `(Y0_error-Z_error)` in each bin. Also report Spearman correlations of both q_FO and q_SC2 with Y0 error and baseline refinement improvement. Report ties. Do not stop or change mappings based on diagnostic outcomes.

## Four refinements and evaluation

1. `uniform`: a_i=1; must reproduce the advisor Stage-2 algorithm and agree on published metrics within numerical tolerance. The initial `max_abs <=1e-4` chart-coordinate gate was too strict for the existing CUDA `index_add_` scatter: before any adaptive outcomes, an independent rerun of the *unchanged original solver* differed from its own saved Z by max_abs `0.00024414062` (mean_abs `1.98e-5`); adaptive uniform versus current original differed by max_abs `0.00024414062` (mean_abs `1.87e-5`). The resolved pre-outcome numerical gate is `max_abs <=5e-4` and `mean_abs <=5e-5` versus saved Z, plus aligned mean/p95 and CT/TW/KS differences each `<=1e-4` versus the saved baseline. Record both the initial failed check and this source-versus-source evidence. If the revised checks fail, stop and resolve protocol mismatch before interpreting variants. This numerical clarification changes no method, input, or efficacy endpoint.
2. `first_order`: a_i=0.5+percentile(q_FO[i]).
3. `sc2`: a_i=0.5+percentile(q_SC2[i]).
4. `sc2_shuffled`: RNG `np.random.default_rng(10)` permutes the finished sc2 fidelity vector across all N nodes; preserve its exact value distribution. Record the permutation. No stratification or extra shuffled seeds.

All variants use `Z_i^(t+1)=(a_i Y0_i + lambda sum_j W_ij Z_j^t)/(a_i+lambda d_i)` for 200 iterations with identical initialization and graph. Use the *same* seed-10, 20%-of-N subset selected by the original `random.sample(range(N), int(0.2*N))` for CT/TW/KS. Compute affine mean/median/p90/p95/max on all N with the original `common_metrics` implementation. Also retain aligned per-sample errors and report improvement and harm versus uniform separately. Produce three figures: reliability quintile vs Y0 error, reliability quintile vs baseline refinement improvement, and four-method error CDF on all N.

- Primary comparison: `sc2` final Z aligned mean versus uniform, first_order and sc2_shuffled. Pilot support requires sc2 mean at least 0.005 m lower than each, with sc2 p95 no more than 0.005 m above uniform. These are screening thresholds, not statistical proof. Regardless of outcome, report every metric and retain negative results. Do not tune tau, 0.5+p, k, lambda, iterations or seed after seeing results. No OOS claim.

## Code, outputs, budget and stop rules

- A single minimal entry under `methods/sc2_stage2_adaptive_fidelity/` is justified because the existing solver accepts scalar fidelity only; import/reuse existing graph construction, alignment and metrics. The other conversation's isolated worktree may contain concurrent implementation; avoid touching its files and do not silently mix code/results from checkouts.
- Output one family under `results/sc2_stage2_adaptive_fidelity_pilot/`: `config.json`, reliability arrays, fidelity vectors/permutation, per-variant embeddings and JSON, diagnostic JSON, three figures, and one `REPORT.md`. Record exact environment and commands. Append a completed experiment to `docs/research/experiment_log.md` and retained results to `docs/research/experiment_registry.md`; do not change `research_state.md` without user decision.
- Before full run, focused tiny-graph tests must verify symmetric C, H/S definitions, tie percentiles, zero/constant-score behavior, and that a_i=1 matches the original solver update. Budget: one score construction plus four 200-iteration solves and original evaluation. If an input/protocol mismatch occurs, stop dependent execution and report the exact mismatch. Diagnostic correlations never gate execution.
