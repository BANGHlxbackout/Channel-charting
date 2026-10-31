# Experiment Registry

## Purpose

This file is the structured index of existing experiment results and artifacts.

It answers:

- what result exists;
- what role it plays;
- which evaluation protocol it used;
- which results are directly comparable;
- where the source artifact is stored.

It does **not** determine the research direction.

Current research decisions belong in:

```text
docs/research/research_state.md
```

Detailed experiment history belongs in:

```text
docs/research/experiment_log.md
```

---

## Comparability Rule

Do not directly rank methods unless they share the same `comparable_group` and compatible evaluation protocol.

Potentially non-comparable differences include:

- transductive vs held-out / test split;
- dataset subset;
- stride;
- sample count;
- train/test usage;
- seed policy;
- affine-alignment fitting protocol;
- CT/TW/KS evaluation sampling;
- hyperparameter-selection protocol;
- evaluation implementation.

When protocol details are uncertain, use:

```text
status = verification_required
```

and do not assign the result to a direct-ranking group.

---

## Role Definitions

- `main_baseline` — strongest current baseline used as the primary comparison target inside its comparable group.
- `baseline` — standard baseline in the same protocol.
- `reference_baseline` — useful reference result, but not necessarily directly comparable.
- `ablation` — controlled variant used to isolate one design choice.
- `diagnostic` — experiment intended to understand a mechanism / failure mode, not to win the main metric table.
- `exploratory` — research exploration not yet promoted to a main method.
- `failed` — valid negative result kept for research history.

---

## Status Definitions

- `active` — current valid result used in ongoing research.
- `historical` — retained for reference, not necessarily part of the current main comparison.
- `verification_required` — protocol / artifact details must be audited before direct comparison.
- `negative` — valid result that did not support the tested method / hypothesis.
- `invalid` — should not be used for scientific comparison.

---

## Comparable Groups

### `CG-A-transductive-stride1-current`

Current full-sample / transductive Stage-1 / Stage-2 experiments using the current stride-1 protocol.

Only entries explicitly assigned to this group may be directly compared.

### `CG-B-P2-test-split`

P2 held-out / test-split results.

Do not directly compare numerical localization errors in this group with `CG-A` as if they came from the same protocol.

### `CG-C-triplet-notebook-stride4`

Historical triplet notebook / stride-4 protocol.

Reference only unless the protocol is reproduced under a common controlled setup.

### `UNVERIFIED`

Protocol requires audit before assignment to a direct-comparison group.

---

## Current Registry

| experiment / method | role | protocol | comparable_group | split | stride / N | seed | status | key recorded result | source artifact |
|---|---|---|---|---|---|---|---|---|---|
| P2A-R matched-eta Ball-GLR | numerical / method control | frozen B0/G128; eta .03/.08/.16/.32/.50; original common4199 metrics/all-N affine; no GT parameter generation | CG-A-transductive-stride1-current | same-node transductive | stride1 / N20997 | dataset/eval10; deterministic solver | active | 5/5 numerical PASS; eta=.08 Mean .319205518/P95 .640418255, no S2 G3 hit. Eta=.50 30k cap too short, complete stop32250; slow dual transient diagnosed without changing objective or tolerance | `results/stage2_modern_graph_20261019/p2a_ic8_scale_adaptation/REPORT.md`; `parameter_scan.csv`; `solver_budget_diagnosis.json` |
| P2A-R antenna-scale IC8-P | exploratory method / task negative with limited mechanism signal | 3 input-quantile alpha ×5 eta; matched Ball controls; same common4199 sample/all-N affine; no post-GT grid extension | CG-A-transductive-stride1-current | same-node transductive | stride1 / N20997 | dataset/eval10; deterministic solver | negative | 15/15 numerical PASS; P-active10, IC-distinct8, 3 same-eta signals ateta=.03 (Mean/P95 improve1.17–1.33%/1.22–1.51%, Max worsens38.88–50.13%). All0 ADVISOR/SURVIVOR/TARGET => NO_GO_EXPLORATORY vs S2. Minimum Mean .319195292/P95 .640425846 atq995 eta=.08; no demonstrated weight-perturbation robustness. Delivered2026-10-25 and STOP | `results/stage2_modern_graph_20261019/p2a_ic8_scale_adaptation/REPORT.md`; `comparison/comparison.csv`; `mechanism_summary.csv`; `G2.json`; `G3.json` |
| Stage-2 P0 data/graph/baseline preflight | diagnostic | original top512/G128/lambda0.15/J200, float64 incidence/Laplacian check, immutable input hashes | not applicable | same-node preflight, no training | stride1 / N=20997 | 10 | active | G0 14/14 PASS; DTWD/L relative5.34e-16, L1=0; J historical replay within all tolerances; δ0=0.09517167368; P0 historical scope only; P1 subsequently authorized and recorded separately | `results/stage2_modern_graph_20261019/p0_preflight/G0.json`; `results/stage2_modern_graph_20261019/p0_preflight/REPORT.md` |
| Stage-2 P1 frozen mathematical smoke | diagnostic / numerical gate | independent CVXPY references; synthetic IC/FN16/32/64 and Edge256; gauge tests; no real-data fitting | not applicable | synthetic same-node; GT offline only | N16/32/64 and256; not full-N | 20261019+N; metric10 | verification_required | IC/FN math PASS (gap<=2.995e-9/4.040e-7); SEF reference/constraints pass but native NONCONVERGED; FE/DoT both initializations constraint FAIL and NONCONVERGED. P1 delivered with Edge BLOCKER; no real-data performance claim/P2A | `results/stage2_modern_graph_20261019/p1_math_sanity/REPORT.md`; `gates.csv`; `reference_objectives.csv` |
| P1-Edge Solver Repair | diagnostic / numerical solver gate | fixed objectives and original P1 fixtures, A fixed vs B single residual balancing; strict complete stop; no GT/quality metrics | not applicable | synthetic numerical only | N32/64/256 | 20261019+N | verification_required | Small20/20 PASS; original256 B FE-prior PASS8180, other4 NONCONVERGED/constraint FAIL; aggregateSEF/FE/DoT FAIL. All fixedA U/S/E replay difference0. NUMERICAL_BLOCKER and STOP, no full-N or further retry | `results/stage2_modern_graph_20261019/p1_edge_solver_repair/REPORT.md`; `gates.csv`; `comparison/comparison.csv` |
| P2A IC8-P full-N frozen grid | exploratory method control | frozen B0/G128, 3alpha×3eta, common4199 metrics/all-N affine; GT screening only | CG-A-transductive-stride1-current | same-node transductive | N20997 stride1 | dataset/eval10, deterministic solver | active | 9/9 numerical PASS; alpha.5/.8 mid SURVIVOR only, near-identical output. Representative KS.0646966962 Mean.3179181352 P95.6225886429; J-relative .565%/.260%/1.899% improvement; noADVISOR_HIT/TARGET_HIT | `results/stage2_modern_graph_20261019/p2a_b0_ic_fulln/REPORT.md`; `parameter_scan.csv` |
| P2A IC9-P full-N frozen grid | exploratory method / valid negative | same frozen B0/G128, 3alpha×3eta, same common metrics and affine | CG-A-transductive-stride1-current | same-node transductive | N20997 stride1 | dataset/eval10, deterministic solver | negative | 9/9 numerical PASS but noSURVIVOR, NO_GO_EXPLORATORY; representative mid KS.0714141272 Mean.3399127660 P95.6552081386, worse than J/CG; P=0 for all9 so alpha mostly scalar objective factor | `results/stage2_modern_graph_20261019/p2a_b0_ic_fulln/REPORT.md`; `G3.json` |
| P2A IC9-G full-N frozen grid | geometry adaptation / valid negative | same alpha/eta and actualdelta pairs against IC9-P; fixed B0/G128 | CG-A-transductive-stride1-current | same-node transductive | N20997 stride1 | dataset/eval10, deterministic solver | negative | 9/9 numerical PASS but noSURVIVOR, NO_GO_EXPLORATORY; mid KS.0711076252 Mean.3423121866 P95.6670494984; vsP mid KS+.429% improvement but Mean/P95 worsen .706%/1.807%; no consistent group gain | `results/stage2_modern_graph_20261019/p2a_b0_ic_fulln/ic9_group_paired_comparison.csv`; `REPORT.md` |
| B0+S2-CG | numerical control | same `(I+0.15L)Z=Y` as S2-J; float64 CG, no new regularization; common seed10 evaluation subset | `CG-A-transductive-stride1-current` | same-node transductive | stride1 / N=20997 | 10 | active | true relative residual9.95e-11; KS0.065064569, Mean0.318747938m, P950.634639233m; geometrically indistinguishable from J200 at this precision | `results/stage2_modern_graph_20261019/p0_preflight/s2_cg/result.json` |
| ADP + timestamp fused + geodesic + Siamese FCF (`Y0`) | baseline | current transductive Stage-1 chart; common seed10 20% metric subset re-evaluated in G1 comparison | `CG-A-transductive-stride1-current` | no held-out split demonstrated | stride1 / N=20997 | 10 | active | CT 0.995712; TW 0.995987; KS 0.084027; mean 0.409347 m; P95 0.855793 m; original result/chart restored from 61178b5; total pipeline elapsed 1633.55 s, training-only time/VRAM not recorded | `results/adp_old_siamese/stride1_n20997_seed10/result.json`; `results/adp_old_siamese/stride1_n20997_seed10/embedding.npz`; `results/g1_acf_graph_fcf/comparison/comparison.csv` |
| G1-ACF Identity propagation | ablation | full-N Stage-1, common seed10 20% metric subset; all-N affine evaluation | `CG-A-transductive-stride1-current` | same-node transductive | stride1 / N=20997 | 10 | active | CT 0.996966; TW 0.996868; KS 0.079401; mean 0.398343 m; median 0.361716 m; P95 0.843197 m; 2000 steps, best1920 | `results/g1_acf_graph_fcf/stride1_n20997_seed10_identity_full/result.json` |
| G1-ACF fused-graph propagation | exploratory method | full-N Stage-1, identical G1 model/training/pair rules; only adjacency differs from Identity | `CG-A-transductive-stride1-current` | same-node transductive | stride1 / N=20997 | 10 | negative | CT 0.996947; TW 0.996849; KS 0.079340; mean 0.399771 m; median 0.364578 m; P95 0.846337 m. Better than B0, but only KS marginally better than Identity; propagation benefit unsupported in this run | `results/g1_acf_graph_fcf/stride1_n20997_seed10_graph_full/result.json`; `results/g1_acf_graph_fcf/REPORT.md` |
| G1 loss direction on saved embeddings | diagnostic | fixed fused-k20 diffusion operator; original/checkpoint Graph and Identity; one equal-RMS coordinate gradient step; no model training or GT step selection | not applicable; not a trained-method ranking | same-node transductive offline diagnosis | stride1 / N=20997 | pair 20261026; metric 10 | historical | Graph proposed step Mean/P95 0.399771/0.846337→0.398884/0.843249 m; 54.43% nodes improve, 45.57% worsen; Identity improves similarly and stays better. Only local gradient direction supported, no retraining claim. | `results/g1_acf_graph_fcf/loss_direction_diagnostic/REPORT.md`; `comparison/comparison.csv` |
| EXP-G1-ERROR-SOURCE-20261027 Stage-1 target/fit diagnosis | diagnostic | same saved G1 Graph/Identity; 399983 random pairs and all228773 fused-k20 edges; offline GT scalar calibration only | not applicable; not a trained-method ranking | same-node transductive offline attribution | stride1 / N=20997 | pairs20261027; saved charts10 | active | D_geo versus GT distance Spearman .982725, but global-scale short/long bias +.4536/-.3904m. Graph/Identity residuals similar; node error weakly associated with random target bias (.1705) and barely with local edge bias (-.0257) or model target-fit error (.0597). No justified new loss yet. | `results/g1_acf_graph_fcf/error_source_diagnostic/REPORT.md`; `pair_groups.csv`; `spatial_regions.csv` |
| Graph-harmonic diffusion refinement of ADP/time geodesic chart | main_baseline | current transductive Stage-2 refinement; same-operator E1 replay | `CG-A-transductive-stride1-current` | no held-out split demonstrated | stride1 / N=20997 | 10 | active | CT 0.998220; TW 0.998223; KS 0.065065; mean 0.318748 m; P95 0.634640 m; historical JSON from Git61178b5, replay Mean identical | `results/g1_acf_graph_fcf_followup/e1_stage2/b0/result.json`; historical `results/diffusion_refined_charting/stride1_n20997_k128_lambda0.15_seed10/embedding.npz` |
| G1-Identity + fixed Stage-2 | exploratory pipeline | common top512/k128/lambda0.15/200 Jacobi, common 4199-node metrics/all-N affine | `CG-A-transductive-stride1-current` | same-node transductive | stride1 / N=20997 | 10 | negative | KS0.066559; Mean0.323004 m; P95 0.648439 m; all primary endpoints worse than B0+S2 | `results/g1_acf_graph_fcf_followup/e1_stage2/g1_identity/result.json` |
| G1-Graph + fixed Stage-2 | exploratory pipeline | same shared S2 graph/operator as B0 and Identity | `CG-A-transductive-stride1-current` | same-node transductive | stride1 / N=20997 | 10 | negative | KS0.066692; Mean0.324695 m; P95 0.649174 m; worse than B0+S2 and Identity+S2 on primary endpoints | `results/g1_acf_graph_fcf_followup/e1_stage2/g1_graph/result.json` |
| G1 frozen-checkpoint propagation sensitivity | diagnostic | Graph@Graph, Graph@I, Identity@I; native output/stress replay; eval/no_grad | not applicable | same-node sensitivity; not retrained-method ranking | stride1 / N=20997 | 10 | active | Graph swap to I stress1.872658→3.208952 (+71.36%); native residual delta/h medians0.860/0.598. Graph dependence established, extra geometric benefit unsupported; original biased pair sampler retained | `results/g1_acf_graph_fcf_followup/e2_propagation/manifest.json`; `results/g1_acf_graph_fcf_followup/REPORT.md` |
| Graph-harmonic refinement with Euclidean kNN graph on `Y0` | ablation | current transductive Stage-2 graph ablation | `CG-A-transductive-stride1-current` | no held-out split demonstrated | stride1 / N=20997 | 10 | historical | CT 0.9955; TW 0.9956; KS 0.0804; mean 0.3944 m; P95 0.7645 m | `results/euclidean_knn_refined_charting/stride1_n20997_euclidean_k128_lambda0.15_seed10/result.json` |
| ADP + timestamp fused + geodesic + metric MDS | reference_baseline | historical MDS evaluation; protocol details require audit | `UNVERIFIED` | unknown / requires audit | source name says stride1 / N=20997 | 10 | verification_required | CT 0.9845; TW 0.9899; KS 0.1063; mean 0.5275 m; P95 1.1303 m | `results/adp_old_fused/stride1_n20997_seed10/result.json` |
| P2 baseline | reference_baseline | saved test-split metrics | `CG-B-P2-test-split` | test split | N / stride require source audit | 10 | historical | CT 0.9777; TW 0.9612; KS 0.2292; mean 1.4700 m; P95 3.1100 m | `results/p2_baseline_shard4/p2_baseline_shard4_seed10-lambda_b1.0-lambda_box1.0-powtype0-Pthresh16.2-Mp3-Tc5-Mt0.8-1tperanch-bs100-lr0.001-100epochs-hidden832-fereim-schpar50_result.json` |
| Triplet baseline from `ChannelCharting.ipynb` | reference_baseline | historical notebook / stride-4 protocol | `CG-C-triplet-notebook-stride4` | protocol requires audit | stride4 / N=20997 | 10 | historical | CT 0.9755; TW 0.9596; KS 0.2647; mean 1.6477 m; P95 4.1381 m | `results/triplet_baseline_notebook/stride4_n20997_seed10/result.json` |
| Stage-1 `k=20` graph provenance sanity check | diagnostic | diagnostic-only sampled `D_geo` reproduction | not applicable | no split; 32 fixed sources | stride1 / N=20997 | 10 | active | connected; degree 20/20/20; 658 boundary-tie rows; 671,904/671,904 distances exact; provenance consistent | `results/phase2c_h1_h2/audit.json` |
| Phase 2C H1 edge-anomaly and sampled path-concentration diagnostic | diagnostic | offline GT-assisted diagnostic; discovery-ranked and disjoint validation | not applicable | 100,000 calibration pairs; 128x512 discovery; 128x512 validation | stride1 / N=20997 | 10 | active | strong preliminary signal: top 25/50/100 cover 22.78%/32.14%/39.65% of severe validation paths; counterfactual reported separately | `results/phase2c_h1_h2/h1_summary.json` |
| Phase 2C H1 frozen targeted-removal counterfactual | diagnostic | offline GT-assisted fresh-test counterfactual; frozen ranking and calibration | not applicable | 128x512 fresh test; pair-disjoint from calibration/discovery/prior validation | stride1 / N=20997 | 11 | negative | frozen criterion failed: targeted repair 17.30%/18.44%/23.60% far exceeded both controls, but median endpoint was insensitive and overestimation/path inflation exceeded controls; `H1_supported=null` | `results/phase2c_h1_h2/h1_counterfactual_summary.json` |

---

## Historical Artifacts Recovered in the 2026-09-13 Inventory

These entries make existing files discoverable. They are not newly executed experiments or promoted research directions. Paths and selected saved summaries were inspected; split identity, leakage, tuning history, and evaluation equivalence have not been fully re-audited. All remain `UNVERIFIED` / `verification_required` for direct comparison. Dataset/subset labels below are recorded metadata or directory labels, not an independent provenance certification.

| experiment / method | role | recorded protocol / split | dataset / subset; stride / N | seed | comparable_group | status | recorded evidence / reuse | source artifact |
|---|---|---|---|---|---|---|---|---|
| Historical held-out FCF | reference_baseline | held-out train/test | adp_old_shard4; 16798 train / 4199 test; stride requires audit | 10 | `UNVERIFIED` | verification_required | reusable train/test chart and evaluation pipeline; verify split and selection protocol before promotion | `results/heldout/seed10/adp_siamese/result.json` |
| Historical train-only diffusion | reference_baseline | refinement on training chart | adp_old_shard4; training subset; stride/N require audit | 10 | `UNVERIFIED` | verification_required | training-side refinement artifact used by OOS; not itself unseen-sample evidence | `results/heldout/seed10/diffusion_train/result.json` |
| Historical diffusion-refined OOS | reference_baseline | saved `diffusion_refined_oos_heldout`; train-fitted affine applied to test | adp_old_shard4; 16798 train / 4199 test; stride requires audit | 10 | `UNVERIFIED` | verification_required | saved test mean 0.4250 m; existing extension implementation to inspect before new inductive work | `results/heldout/seed10/diffusion_oos/result.json` |
| CF-HRD residual-student evaluation | exploratory | saved train-fitted affine / held-out test | adp_old_shard4; train/test subset; split identity requires audit | 10 | `UNVERIFIED` | verification_required | saved `y0_q` test mean 0.4871 m; teacher/context/student intermediates retained in the same family | `results/CF_HRD/seed10/stage5_evaluation/result.json` |
| Direct-support edge filtering | exploratory | historical filtering experiment; compatibility requires audit | adp_old_shard4; folder label stride1 / N20997 | 10 | `UNVERIFIED` | verification_required | historical report says acceptance failed; preserve raw, energy-preserving, and lower-lambda controls | `results/direct_support_edge_refined_charting/summary.md` |
| Repair-gated residual correction | exploratory | historical gate/control evaluation; compatibility requires audit | adp_old_shard4; folder label stride1 / N20997 | 10 | `UNVERIFIED` | verification_required | report records `repair_beats_best_global_mean=False`; retain unsuccessful mechanism and controls | `results/repair_gated_residual_correction/stride1_n20997_seed10_phys128/result_summary.md` |
| Lambda sensitivity family | ablation | historical parameter sweep; selection protocol requires audit | folder label stride1 / N20997 | 10 | `UNVERIFIED` | verification_required | 13 existing run directories; reuse for parameter-history audit, not a novelty claim | `results/lambda_sensitivity_refined_charting/` |
| Diffusion refinement diagnostics | diagnostic | historical offline diagnostic | folder label stride1 / N20997 | 10 | `UNVERIFIED` | verification_required | retained summary and diagnostic outputs; not a deployable method result | `results/diffusion_refinement_diagnostics/stride1_n20997_seed10_k32/diagnostic_summary.md` |
| Local consistency proxy diagnostics | diagnostic | historical original and revised diagnostics | folder labels stride1 / N20997 | 10 | `UNVERIFIED` | verification_required | both original and revised folders retained; read definitions before reuse | `results/local_consistency_proxy_diagnostics/stride1_n20997_seed10_k32_phys128_revised/local_consistency_proxy_diagnostics.md` |
| Stage-1 low-common-neighbor edge-length penalty pilot | diagnostic / failed assumption | full transductive Stage-1 graph variants with fixed pre-FCF MDS gate and secondary held-out isotonic diagnostic | stride1 / N=20997; no FCF run | 10 / pair seed 20260925 | `not applicable` | negative | This added penalty rule failed the MDS gate: mean 0.5825 m vs baseline 0.4581 m; it does not test SC2-PCR's consensus mechanism or third-party support in general; FCF and Stage-2 were not run | `results/sc2_stage1_consensus_pilot/REPORT.md` |
| SC2-inspired Stage-2 adaptive fidelity pilot | method pilot | CG-A transductive; fixed Y0/D_geo/W, k=128, lambda=0.15, 200 iterations; comparable within this four-variant run | stride1 / N=20997 | 10 | `CG-A-stage2-adaptive-fidelity-seed10` | negative | All-N aligned mean: uniform 0.318748 m, first-order 0.320131 m, second-order 0.322221 m, shuffled second-order 0.318710 m; second-order p95 0.648274 m vs uniform 0.634639 m. Frozen support rule failed; no held-out claim. | `results/sc2_stage2_adaptive_fidelity_pilot/REPORT.md` |

| Stage-1 two-view numerical-equality score diagnostic | diagnostic | transductive relation-only check on original Stage-1 k=20 edges; no chart intervention | not applicable | no held-out split; 23,973 relation pairs | stride1 / N=20997 | 10; bootstrap 11 | negative | Numerical equality score failed: full/equality/combined/shuffled AUC 0.7933/0.4528/0.6949/0.6865; matched near-rate difference -0.0024, CI [-0.0243, 0.0196]. Does not test both-views-near or SC2 third-party support. | results/sc2_stage1_multiview_check/REPORT.md |
| SC2-inspired Stage-1 harm attribution diagnostic | diagnostic | transductive fixed-subset MDS; T/R counterfactual weight ablations; no FCF or Stage-2 | `not applicable` | no held-out split; GT offline only | stride1 / N=20997; MDS source subset 2100 | 10 | active | T_only mean increment 0.122366 m vs full 0.124331 m; R_only 0.007007 m; T_only P95 increment 0.231889 m. Frozen rule supports T as main harm source; no improvement method claim. | `results/sc2_stage1_harm_attribution/REPORT.md` |
| EXP-MV-G0-001 multiview edge risk Gate 0 | diagnostic / implementation gate | GT-free original Stage-1 graph and four-view ADP regression; no chart intervention or ranking | `not applicable` | not applicable | stride1 / N=20997 | deterministic 32 sources; no random seed | active | 13/13 checks passed; ADP sum max error 9.54e-6; 32×20997 baseline geodesic entries exact; original 2759 reciprocal edge pairs have tiny weight asymmetry | `results/multiview_edge_uncertainty/REPORT.md` |
---

## Notes on the Legacy `method_summary.md`

The historical file:

```text
results/method_summary.md
```

should be treated as a **legacy snapshot** because it mixes results from different protocols in one flat table.

Recommended handling:

1. Preserve it for provenance.
2. Do not use it as the authoritative method-ranking table.
3. Optionally move it to:

```text
results/archive/method_summary_legacy_2026-09-02.md
```

4. Use this `experiment_registry.md` for all future result indexing.

---

## Adding New Results

For every new result:

1. first record the detailed experiment in `docs/research/experiment_log.md`;
2. determine its protocol;
3. assign a `comparable_group` only when justified;
4. choose its role;
5. add the source artifact path;
6. add a concise recorded result;
7. never place incompatible protocols into the same direct-ranking group.

For Phase 2C diagnostics, use:

```text
role = diagnostic
```

and do not promote the result to the main method-comparison group merely because it produces a numerical improvement.
