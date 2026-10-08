# SC²-inspired Stage-2 edge-consensus test

Status: frozen before scoring or evaluation. Isolated branch codex/sc2-stage2-edge-consensus; never alter master or the earlier node-fidelity branch.

## Research question
Can agreement supported by third-party samples identify Stage-2 graph edges that would pull physically unrelated points together? This transfers the *relationship/consensus* mechanism of SC²-PCR, not a node confidence or fidelity rule. Original paper: https://arxiv.org/html/2203.14453; author code: https://github.com/ZhiChen902/SC2-PCR/blob/main/SC2_PCR.py.

## Inputs and ownership
Read original D_geo, top512 neighbor cache, Y0, saved Z, UE positions. No Stage-1 recomputation or FCF retraining. Build the same Stage-2 W (k128, lambda .15, 200 iterations) from the tracked baseline. GT positions are offline evaluation only.

## Frozen score and diagnostic
For each directed top128 candidate pair, let r_ij=abs(log(||Y0_i-Y0_j||)-log(D_geo_ij)-log_s), with log_s the directed-pair median log ratio. Define the hard first-order compatibility B_ij=1 if r_ij <= median(r)+MAD(r), else 0. Use the symmetric union of this sparse B; diagonal zero. The paper uses a hard first-order compatibility and its common witnesses. For each undirected W edge, score T_ij=B_ij * sum_k B_ik B_kj. Do not aggregate T to nodes or change Y0 fidelity.

GT-assisted edge diagnostic: within each of ten D_geo-distance bins, define unusually long physical edges as the top 10% of GT endpoint distances in that bin. Report the bad-edge prevalence in bottom versus top T quintiles, also within B=1, and compare with first-order r. This label is for checking the score, not for selecting edges or tuning the method. The mechanism has edge-level support only if lower T enriches these edges and the relationship is not explained solely by D_geo length or B.

## Single fixed intervention
Among undirected W edges with B_ij=1, downweight the bottom 10% ranked by common-witness count T (seed-10 random tie break fixed before GT evaluation) by factor 0.5. Leave B_ij=0 edges unchanged in every targeted variant; this isolates the incremental value of second-order support once first-order compatibility already holds. Preserve W symmetry. Compare:
1. original uniform W (must reproduce saved Z),
2. SC²-ranked edge downweight,
3. first-order-ranked edge downweight on the same B=1 candidate pool using r (largest residual first),
4. shuffled edge assignment of the SC² downweight mask within B=1 and D_geo deciles, seed 10,
5. uniform W scaling matching the SC² total edge-weight mass.

This is a decision test, not a parameter sweep. Same Y0, lambda .15, 200 Jacobi iterations, seed 10 and evaluation subset for all variants. GT never enters graph edits. Report mean/median/P90/P95/max affine error and CT/TW/KS, plus graph weight totals and edge-score diagnostic. Success requires SC² mean below the original baseline and all controls, without P95 exceeding baseline by >0.005 m; otherwise retain as a negative test. No post-result threshold/strength adjustment.

## Stop
If exact input/order/W reproduction fails, stop dependent execution. If the edge diagnostic contradicts the premise, still report it and complete the fixed five-way comparison; do not revise the score or selection rule after seeing GT. No merge to master.



## User-authorized follow-up, frozen before repeated runs

Question: Is the single shuffled result representative, and where do SC² and random edge masks repair or harm the chart? Keep the same original W, eligible B=1 pool, selected edge count per D_geo decile, attenuation 0.5, lambda .15, iterations 200, Y0, and evaluation protocol. Run 20 independent shuffled masks with seeds 10 through 29. For each, report full-N affine mean, median, P90, P95, max and total graph weight. Compare the fixed SC² outcome against the random distribution; report the number of random seeds that beat SC² and baseline. Also report per-sample improvement/harm counts and magnitudes for fixed SC² and the seed-10 shuffled run, always comparing to the same uniform result. GT is evaluation-only. No mask strength, fraction, score, or seed-range changes after outcomes. This follow-up does not change the original frozen five-way result.

## User-authorized all-edge random control, frozen before run

To test whether the B=1 restriction itself matters, run another 20 random masks with seeds 10..29. For each mask, select the same number of undirected W edges in each D_geo decile as the fixed SC² mask, but sample from *all* W edges in that bin (including B=0). Apply the same 0.5 attenuation, original lambda=0.15 and 200 Jacobi iterations. Report all-N affine error summaries and total W. Compare its distribution with the B=1-restricted random group, SC², and the original baseline. This is a mechanism control, not method tuning. Also compare the already-retained original-graph lambda family using identical N/k/seed/protocol before claiming any stronger baseline.

## User-authorized incompatible-edge pool control, frozen before run

The 20 eligible-only random controls improved more than 20 all-edge random controls. To test whether keeping first-order-incompatible edges intact matters, run seeds 10..29 with the same selected count per D_geo decile but sample only B=0 edges. Apply the same 0.5 attenuation and baseline solve/evaluation. This is a readout of candidate-pool choice, not a tuned final method. Compare against both prior 20-seed distributions and the original-graph lambda family.

## Existing-best-lambda cross-check, frozen before run

The retained original-W lambda family has the lowest recorded mean at lambda=0.1344 (0.318395704 m, same N/k/seed). To test whether the eligible-edge random effect is complementary to this stronger control, repeat exactly the existing B=1 random masks for seeds 10..29 with lambda=0.1344 and 200 Jacobi iterations. Do not alter eligibility threshold, selected fraction, D_geo bin counts, attenuation, Y0, W source, or random masks. Compare against the saved original-W lambda=0.1344 embedding/result and report full-N affine error summaries. This is a post-result cross-check, not a fresh lambda search or an independent test set.

## Complete metric comparison requested by user

The earlier repeat controls saved all-N affine errors but not CT/TW/KS for every random seed. Re-run the exact fixed masks (seeds 10..29, B=1/all/B=0 pools at lambda=0.15, and B=1 at lambda=0.1344) with the unchanged 200-step solver. Use exactly the baseline seed-10 random.sample 20% subset for CT/TW/KS and full-N affine alignment for mean, median, P90, P95, max. Record each run's full metrics and group medians/min/max. Verify repeat all-N mean/p95 against already saved JSON within CUDA run variability. No mask, score, graph rule, seed set, lambda, or evaluation protocol change.

## 2026-10-08 advisor-requested post-hoc audit and Gitee preparation

Question: Why did the fixed low-common-support mask fail to beat distance-stratified random edge attenuation, despite enriching physically long edges? No new mask selection, training, graph solve or tuning is authorized in this audit. Reuse the saved scores, edge indices, fixed masks, baseline and variant embeddings; recompute only evaluation diagnostics and presentation plots. GT positions are offline diagnostics only.

Freeze outputs before reading outcomes: document the exact first-order residual, median+MAD threshold, symmetric union B, second-order witness count and tie break; list the selected-edge count, weight/length/score distributions and D_geo-decile counts for SC2 and the seed-10 matched random mask. Within each D_geo decile among B=1 edges, label the top 10% physical endpoint lengths as an exploratory GT-only suspect group and compare selected fractions. This label is not a causal edge error.

For outcomes, affine-align each saved chart separately exactly as baseline evaluation; compute per-sample baseline-minus-variant errors, repair/harm counts and sums for SC2 and seed-10 random; report spatial map, initial-error quintiles and a few explicitly post-selected sample/edge examples with sample IDs, coordinates, score and incident selected-edge counts. State that simultaneous edge changes propagate and cannot assign a sample outcome to a single edge. Inspect whether distinct antenna-view arrays are available, but do not construct, fit or tune a new view rule in this audit.

Organize a local tracked config/reproduction manifest, exact source paths and hashes, code pointers, machine results and one updated REPORT.md in this isolated branch. The Gitee remote URL may be documented, but do not push or merge. Later upload requires a user instruction and should cite the then-current remote commit, not an unpushed local SHA.
