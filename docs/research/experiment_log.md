# Experiment Log

## Purpose

This file records **what experiments were actually executed**, including negative results, incomplete experiments, and historical observations with missing records.

It is not the current research consensus.  
Current research decisions belong in:

```text
docs/research/research_state.md
```

Result artifact indexing belongs in:

```text
docs/research/experiment_registry.md
```

Do not delete negative experiments.

---

## Status Definitions

- `confirmed` — configuration, artifacts, results, and interpretation are sufficiently recorded.
- `preliminary` — experiment exists, but evidence is not yet strong enough for a stable claim.
- `negative` — valid experiment that did not support the tested idea / hypothesis.
- `invalid` — experiment cannot support a conclusion because of leakage, protocol error, implementation error, or other fatal issue.
- `missing-records` — historical experiment is remembered, but key configuration / numerical artifacts are missing, so it cannot be used as formal evidence.

---

## Historical Records

### EXP-HIST-001 — Stage-1 Training Sufficiency Check

**Date:** historical, before 2026-09-02  
**Question:** Could the Stage-2 improvement be explained mainly by insufficient Stage-1 Siamese FCF training?

**Known observation:**  
Additional Stage-1 training / tuning was previously tested. According to the retained research discussion, the result did not remove the performance gap relative to the advisor Stage-2 graph regularization baseline.

**Available records:**  
Incomplete in the current research memory. Exact epoch schedule, learning-rate settings, seed policy, detailed metrics, and artifact paths are not currently documented here.

**Interpretation:**  
This historical observation weakens the simple explanation that Stage 2 only compensates for obvious Stage-1 under-training.

It does **not** yet qualify as paper-grade evidence.

**Status:** `missing-records`

**Action if needed later:**  
Reconstruct or rerun a minimal controlled Stage-1 training-sufficiency experiment before using this point in a paper or rebuttal.

---

### EXP-HIST-002 — Refinement Distillation Attempt

**Date:** historical, before 2026-09-02  
**Question:** Can the refined chart `Z` be used as a pseudo-target / teacher so that a parametric FCF learns the refined geometry and matches or exceeds the advisor graph-regularization baseline?

**Known observation:**  
A refinement-distillation experiment was previously executed. According to the retained research discussion, it did not exceed the advisor graph-regularization baseline.

**Available records:**  
The detailed experiment record was deleted / not retained. Exact configuration, code path, metrics, and artifacts are unavailable here.

**Interpretation:**  
This is only a historical negative observation. It must not be cited as formal experimental evidence.

**Status:** `missing-records`

**Action if needed later:**  
Only rerun if the research direction returns to inductive distillation or if a reviewer-facing comparison requires it.

---

## Current Records

### EXP-P2C-GATE05-001 — Stage-1 Graph Provenance Sanity Check

**Date:** 2026-09-03

**Research question / hypothesis:**  
Can the directed weighted Stage-1 `k=20` graph reconstructed from the saved fused dissimilarity credibly represent the graph that generated the saved `D_geo`?

**Phase / task:**  
Phase 2C Gate 0.5 — Minimal Stage-1 Graph Provenance Sanity Check

**Code entry point:**  
`diagnostics/phase2c_h1_h2/phase2c_io_graph.py`

**Configuration:**

```text
dataset: adp_old_shard4 Stage-1 artifacts
subset: full saved fused matrix; 32 sampled shortest-path sources
stride: 1
N: 20997
seed: 10
k: 20
neighbor algorithm: NearestNeighbors(metric="precomputed") + kneighbors_graph(mode="distance")
shortest path: method="auto", directed=False
near-exact tolerance: rtol=1e-6, atol=1e-5
```

**Input artifacts:**

- `results/adp_old_fused/stride1_n20997_seed10/fused.npy`
- `results/adp_old_fused/stride1_n20997_seed10/geodesic.npy`

**Output artifacts:**

- `results/phase2c_h1_h2/audit.json`
- `results/phase2c_h1_h2/stage1_graph.npz`

**Evaluation protocol:**  
Diagnostic-only provenance reproduction. No GT, H1 edge screening, edge removal, H2 analysis, or method comparison was performed.

**Key results:**

| metric | value |
|---|---:|
| connected components (`directed=False`) | 1 |
| directed CSR nnz | 419,940 |
| non-self outgoing degree min / median / max | 20 / 20 / 20 |
| boundary-tie rows | 658 / 20,997 (3.1338%) |
| sampled validation distances | 671,904 |
| exact match fraction | 1.0 |
| near-exact match fraction | 1.0 |
| beyond-tolerance mismatch count | 0 |
| maximum absolute / relative error | 0 / 0 |

**Observation:**  
All 32 reconstructed shortest-path rows matched the saved float32 `D_geo` bit-for-bit. Boundary ties exist, but they produced no sampled reproduction discrepancy under the recorded environment.

**Interpretation:**  
The reconstructed directed weighted CSR is sufficiently supported for use as the Stage-1 graph in the next Phase 2C diagnostic step. This provenance result is not evidence for H1 or H2.

**Falsification / failure criterion:**  
The gate would fail for disconnection, incorrect non-self degree, self edges, or any beyond-tolerance sampled `D_geo` mismatch. None occurred.

**Limitations / caveats:**

- Reproduction was checked on 32 fixed random sources, not by recomputing full APSP.
- `658` rows have a tie exactly at the 20/21 boundary, so this result validates the current reconstruction through sampled `D_geo` agreement rather than proving globally unique neighbor identities.
- The run used Python 3.11.15, NumPy 2.4.6, SciPy 1.17.1, and scikit-learn 1.8.0; the original artifact did not record package versions.

**Status:** `confirmed`

**Registry entry updated:** `yes`

---

### EXP-P2C-H1-001 — Stage-1 Edge Anomaly and Sampled Path Concentration

**Date:** 2026-09-03

**Research question / hypothesis:**  
Does the verified Stage-1 wireless-derived local graph contain a small set of physically inconsistent edges with disproportionate impact on severely underestimated global geodesic paths?

**Phase / task:**  
Phase 2C H1 — first-stage diagnostic only

**Code entry point:**  
`diagnostics/phase2c_h1_h2/run_h1_diagnostic.py`

**Configuration:**

```text
dataset: adp_old_shard4 saved Stage-1 artifacts
stride / N: 1 / 20997
seed: 10
graph: Gate-0.5-verified directed weighted CSR; shortest_path directed=False
effective undirected edge weight: minimum stored direction for reciprocal edges
physical local scale: median q=16 physical-neighbor distance
wireless local scale: median incident effective Stage-1 edge weight
calibration: 100000 independent pairs; isotonic log(D_geo) on log(d_phys)
discovery: 128 density-stratified sources x 512 targets
validation: 128 disjoint density-stratified sources x 512 targets
severe residual threshold: calibration 5th percentile
controls per concentration cutoff: 50 uniform + 50 matched
matched strata: fused-weight decile x maximum-endpoint-degree quartile
candidate score: robust-mismatch percentile x positive severe-usage percentile
primary decision cutoff: top 1% of 228773 effective undirected edges
```

**Input artifacts:**

- `results/phase2c_h1_h2/stage1_graph.npz`
- `results/phase2c_h1_h2/audit.json`
- `results/adp_old_fused/stride1_n20997_seed10/adp.npy`
- `results/adp_old_fused/stride1_n20997_seed10/geodesic.npy`
- `datasets/adp_old_shard4/distance_data.npz`
- `datasets/adp_old_shard4/timestamp_data.npz`
- `results/adp_old_fused/stride1_n20997_seed10/config.json`

**Output artifacts:**

- `results/phase2c_h1_h2/config.json`
- `results/phase2c_h1_h2/h1_summary.json`
- `results/phase2c_h1_h2/h1_edge_table.parquet`
- `results/phase2c_h1_h2/h1_pair_table.parquet`
- `results/phase2c_h1_h2/h1_calibration_curve.npz`
- `results/phase2c_h1_h2/h1_concentration.csv`
- `results/phase2c_h1_h2/figures/`
- `results/phase2c_h1_h2/REPORT.md`

**Evaluation protocol:**  
Offline diagnostic only. UE ground truth is used only to measure physical inconsistency and calibrated geodesic residuals. Calibration, discovery, and validation pair sets are deterministic and mutually disjoint. Edge selection and path-impact ranking use discovery data only. No edge removal, H2, model training, or deployable GT-assisted mechanism was run.

**Key results:**

| metric | value |
|---|---:|
| effective undirected edges | 228,773 |
| discovery / validation pairs | 65,536 / 65,536 |
| discovery / validation severe pairs | 3,309 (5.0491%) / 3,516 (5.3650%) |
| top-1% validation severe-path coverage | 75.2275% |
| matched-random top-1% mean / P95 | 5.0375% / 6.6169% |
| lift over matched-random mean | 14.933x |
| top-1% median robust mismatch | 1.029 |
| tie-excluded top-1% coverage retention | 99.1682% |
| timestamp branch, all edges / top 1% | 78.3078% / 85.8829% |
| timestamp branch enrichment | 1.0967x |
| total runtime | 28.63 s |

**Observation:**  
Bad-path impact is highly concentrated: the discovery-ranked top 1% covers 75.23% of severe validation paths, far above matched-random controls. This result is not driven by k=20 boundary-tie rows. However, the selected set's median robust wireless/physical mismatch is only 1.029, below the frozen threshold of 2. The top set is modestly enriched for timestamp-dominated edges, but this enrichment alone does not establish a causal wireless failure mechanism.

**Interpretation:**  
The run shows a strong preliminary path-impact concentration signal. The generated artifact records `proceed_to_H1_counterfactual = false` under an implementation-specific rule that also required the median robust mismatch of the entire top 1% (2,288 edges) to be at least 2. A subsequent protocol review found that this additional hard gate was not part of the active task specification and was mismatched to the task's actual removal batches `X={25,50,100}`. It therefore does not constitute an H1 falsification. The task-specified targeted counterfactual is recorded separately in `EXP-P2C-H1-CF-001`.

**Falsification / failure criterion:**  
The implementation-specific first-stage decision required all five recorded checks to pass: validation top-1% coverage at least 20%, at least 2x matched mean, above matched P95, top-1% median robust mismatch at least 2, and at least 80% coverage retention after excluding boundary-tie-ambiguous candidates. Four passed; the top-1% robust-mismatch check failed. Protocol review subsequently established that this extra check cannot override the active task's `screening signal -> X={25,50,100}` counterfactual chain. At those actual batch sizes, median robust mismatch is 3.184, 2.443, and 2.051 respectively.

**Limitations / caveats:**

- Only the pre-specified main physical-scale setting `q=16` was executed; the planned `{8,16,32}` sensitivity belongs to a later step and was not run.
- Eleven co-located samples had a zero q=16 median physical scale; only those rows used the median of strictly positive distances within the same q=16 neighbor set.
- For 2,759 reciprocal graph pairs with tiny direction-dependent float differences, the effective undirected weight is the smaller stored direction, matching SciPy `directed=False` shortest-path semantics; the verified directed CSR itself was not changed.
- Sampled path counts are estimates from 128 sources rather than exact all-pairs edge betweenness.
- No counterfactual edge-removal evidence is included in this entry; the completed task-specified counterfactual is recorded separately.

**Status:** `preliminary`

**Registry entry updated:** `yes`

---

### EXP-P2C-H1-CF-001 — Frozen Targeted-Removal Counterfactual

**Date:** 2026-09-03

**Research question / hypothesis:**  
Does removal of the frozen discovery-ranked high-mismatch/high-impact Stage-1 edges repair severe calibrated geodesic underestimation on a fresh test set more than traffic- and structure-matched controls, without disproportionate connectivity, overestimation, or path-inflation damage?

**Phase / task:**  
Phase 2C H1.6 — targeted-removal counterfactual only

**Code entry point:**  
`diagnostics/phase2c_h1_h2/run_h1_counterfactual.py`

**Configuration:**

```text
dataset: adp_old_shard4 saved Stage-1 artifacts
stride / N: 1 / 20997
baseline graph: Gate-0.5-verified directed weighted CSR
shortest-path semantics: scipy directed=False
counterfactual seed: 11
fresh test: 128 q16-density-stratified sources x 512 targets = 65536 pairs
test separation: sources disjoint from prior discovery/validation; unordered pairs disjoint from calibration/discovery/prior validation
calibration / thresholds: frozen from EXP-P2C-H1-001; no refit
targeted batches: frozen discovery-rank prefixes X={25,50,100}
controls: 50 nested matched-random + 50 nested suspicious-low-impact sequences
critical balance: |SMD| <= 0.10 for every replicate and X
secondary balance: |SMD| <= 0.20 for every replicate and X
decision: targeted P1 and P2 above both control P95 values for at least 2/3 X, with connectivity preserved and harm not above either control P95
```

**Input artifacts:**

- `results/phase2c_h1_h2/stage1_graph.npz`
- `results/phase2c_h1_h2/h1_edge_table.parquet`
- `results/phase2c_h1_h2/h1_pair_table.parquet`
- `results/phase2c_h1_h2/h1_calibration_curve.npz`
- `results/adp_old_fused/stride1_n20997_seed10/geodesic.npy`

**Output artifacts:**

- `results/phase2c_h1_h2/h1_counterfactual_config.json`
- `results/phase2c_h1_h2/h1_counterfactual_design.npz`
- `results/phase2c_h1_h2/h1_counterfactual_control_edges.parquet`
- `results/phase2c_h1_h2/h1_counterfactual_control_balance.csv`
- `results/phase2c_h1_h2/h1_counterfactual_runs.parquet`
- `results/phase2c_h1_h2/h1_counterfactual_test_pairs.parquet`
- `results/phase2c_h1_h2/h1_counterfactual_summary.json`
- `results/phase2c_h1_h2/H1_COUNTERFACTUAL_REPORT.md`
- `results/phase2c_h1_h2/figures/h1_counterfactual_repair.png`
- `results/phase2c_h1_h2/figures/h1_counterfactual_harm.png`

**Evaluation protocol:**  
Offline GT-assisted diagnostic only. The edge ranking, candidate sets, isotonic calibration, thresholds, fresh test, control construction, covariate-balance limits, and decision rule were frozen before any removal outcome was computed. The fresh test is pair-disjoint from calibration, discovery, and the previously inspected validation set. Logical undirected removal deletes every stored direction of the selected edge. No H2, sensitivity, training, reranking, or method development was executed.

**Key results:**

| metric | X=25 | X=50 | X=100 |
|---|---:|---:|---:|
| targeted repair rate | 17.3031% | 18.4368% | 23.5979% |
| matched-random repair P95 | 2.5447% | 3.0087% | 6.3097% |
| suspicious-low-impact repair P95 | 0.8950% | 0.9845% | 1.2232% |
| targeted mean delta r_geo, baseline-severe | 0.033005 | 0.034782 | 0.043936 |
| larger control mean-delta P95 | 0.003089 | 0.003515 | 0.008115 |
| targeted median delta r_geo, baseline-severe | 0 | 0 | 0 |
| severe fraction after removal | 4.2297% | 4.1718% | 3.9078% |
| new overestimation increase | 0.6271% | 0.6760% | 0.9872% |
| larger control overestimation-increase P95 | 0.3467% | 0.4552% | 0.6199% |
| targeted P95 log inflation, well-estimated | 0.062203 | 0.063413 | 0.070019 |
| larger control inflation P95 | 0.016549 | 0.021119 | 0.043806 |
| connected components / unreachable pairs | 1 / 0 | 1 / 0 | 1 / 0 |

Fresh-test baseline severe count was 3,352 / 65,536 (5.1147%). All 303 modified graphs remained connected with zero unreachable test pairs. Summed per-graph shortest-path runtime was 331.05 s.

**Observation:**  
Targeted removal repaired a much larger fraction of baseline-severe pairs and produced a much larger mean severe-cohort residual correction than either control family at every X. The frozen median residual endpoint had no resolution: targeted and control medians were all zero because most severe pairs were unchanged by these sparse interventions. Targeted removal also caused larger new-overestimation growth and larger P95 inflation on previously well-estimated pairs than both control families at every X.

**Interpretation:**  
The frozen counterfactual support criterion was not satisfied, so `H1_counterfactual_supported = false` and `H1_supported = null`. The ranked edges are causally influential for a concentrated subset of severe geodesic underestimation, beyond the matched traffic and structural controls, but their deletion is not a clean low-collateral-damage intervention. This negative status is specific to the frozen support criterion and must not be misread as evidence that the ranked edges have no effect.

**Falsification / failure criterion:**  
No X passed both frozen primary endpoints: repair rate exceeded both control P95 values at all X, but median delta residual did not strictly exceed the zero control P95. In addition, all X exceeded both control families on the frozen new-overestimation harm check, and X=50/100 also failed the median well-estimated-inflation check. Therefore zero of three X were fully supported.

**Limitations / caveats:**

- The median primary endpoint is insensitive to a sparse path effect and cannot be interpreted as evidence of no effect.
- The absolute control-relative harm guardrail compares targeted removal with control interventions having much smaller repair magnitude; it favors selective low-damage effects rather than repair/harm efficiency. It was frozen before outcomes and was not changed post hoc.
- Suspicious-low-impact controls had positive, matched total path usage and matched mismatch, but their realized median discovery bad-path enrichment was zero. This was an outcome-free result of the secondary low-enrichment objective, not an eligibility rule.
- Several balance values were close to the admissible SMD boundaries; all nevertheless satisfied the pre-frozen constraints.
- Fifty replicates give a coarse empirical P95.
- No q/source sensitivity was run, so this artifact does not set `H1_supported`.

**Status:** `negative`

**Registry entry updated:** `yes`

---

## 2026-09-13 — Documentation and Artifact Index Maintenance

This is repository documentation maintenance, not a new experiment or a rerun.

- The current registry's nine source artifacts were checked for existence. Historical held-out/OOS, CF-HRD, filtering/gating, sensitivity, and diagnostic families were indexed separately as `UNVERIFIED` / `verification_required`; no new comparable group or method ranking was assigned.
- Correction to the input-path transcription in **EXP-P2C-H1-001**: the current code `diagnostics/phase2c_h1_h2/run_h1_diagnostic.py` uses `results/adp_old_fused/stride1_n20997_seed10/distance_data.npz`, `datasets/adp_old_shard4/timestamps.npy`, and `results/adp_old_fused/stride1_n20997_seed10/result.json` (including gamma). These files exist. The entry above instead names `datasets/adp_old_shard4/distance_data.npz`, `datasets/adp_old_shard4/timestamp_data.npz`, and the fused run's `config.json`, which do not exist in this checkout. This is a correction based on current code and files, not proof of byte-identical historical inputs; the original entry is retained.
- The navigation and outstanding evidence gaps are recorded in `docs/PROJECT_GUIDE.md`. Frozen experiment outcomes were not changed. After the user clarified that the advisor had already supplied the baseline and direction, the working scope was synchronized in `research_state.md` v2.2, with v2.1 preserved in `docs/research/archive/`; this workflow update is not new experimental evidence.

---

## 2026-09-25 — EXP-SC2-STAGE1-CONSENSUS-PILOT

**Research question / hypothesis:**  
Can SC2-inspired common-neighbor support on the inherited Stage-1 directed 20-NN graph improve the Stage-1 distance geometry beyond a uniform edge-length increase and a structure-matched shuffled assignment?

**Phase / task:**  
Pre-FCF Stage-1 graph intervention pilot. The frozen MDS gate was added before FCF training after user steering; FCF and Stage-2 were not run because the gate failed.

**Code entry point:**  
`methods/sc2_stage1/run_pilot.py`

**Configuration:**

```text
dataset: datasets/adp_old_shard4
subset: full CG-A transductive sample set
stride / N: 1 / 20,997
graph: results/phase2c_h1_h2/stage1_graph.npz, directed weighted CSR, k=20
seed: 10 for graph variants and MDS; 20260925 for held-out pair diagnostic
intervention: c_ij=(A^2)_ij; s_ij=c_ij/max(1,min(deg_i-1,deg_j-1)); m_ij=1+0.5*(1-s_ij)
controls: original, equal-total uniform, stratified shuffled multipliers
MDS gate: np.arange(0,20997,10), 2,100 points, seed=10, max_iter=80, n_init=4
secondary isotonic diagnostic: 100,000 calibration + 50,000 uniform + 50,000 local disjoint pairs
```

**Executed commands and environment:**

```text
D:\anaconda3\Scripts\conda.exe run -n diffusion_charting python methods/sc2_stage1/run_pilot.py --build-only --device cuda:0
D:\anaconda3\Scripts\conda.exe run -n diffusion_charting python methods/sc2_stage1/run_pilot.py --skip-build --device cuda:0
D:\anaconda3\Scripts\conda.exe run -n diffusion_charting python -m unittest tests.test_sc2_stage1 -v
Python 3.11.15; torch 2.11.0+cu128; NVIDIA GeForce RTX 3050 Laptop GPU
```

The focused semantic test suite passed 3/3 tests. No FCF or Stage-2 command was executed after the MDS gate failed.

**Input artifacts:**

- `results/phase2c_h1_h2/stage1_graph.npz`
- `results/adp_old_fused/stride1_n20997_seed10/{fused.npy,geodesic.npy,distance_data.npz,geodesic_top512.npy}`

**Output artifacts:**

- `results/sc2_stage1_consensus_pilot/REPORT.md`
- `results/sc2_stage1_consensus_pilot/config.json`
- `results/sc2_stage1_consensus_pilot/variant_summaries.json`
- `results/sc2_stage1_consensus_pilot/{baseline,uniform,shuffled,consensus}/geodesic.npy`
- `results/sc2_stage1_consensus_pilot/{baseline,uniform,shuffled,consensus}/mds.npz`
- `results/sc2_stage1_consensus_pilot/mds_gate.json`
- `results/sc2_stage1_consensus_pilot/gate_pairs.npz`
- `results/sc2_stage1_consensus_pilot/geodesic_gate.json`

**Evaluation protocol:**  
GT-assisted offline evaluation only. GT positions were excluded from support scores, edge weights, graph construction, shortest paths, and training targets. The primary MDS proxy used identical fixed indices for all variants. The secondary isotonic diagnostic used disjoint deterministic calibration and held-out uniform/local pairs. No FCF or Stage-2 run was started.

**Key results:**

| variant | MDS mean m | MDS p95 m | CT | TW | KS | uniform isotonic MAE m | local isotonic MAE m |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 0.458130 | 0.961453 | 0.991671 | 0.993360 | 0.090692 | 0.399153 | 0.109400 |
| uniform | 0.458130 | 0.961453 | 0.991671 | 0.993360 | 0.090692 | 0.399153 | 0.109400 |
| shuffled | 0.559366 | 1.156367 | 0.988663 | 0.991845 | 0.105277 | 0.449695 | 0.118659 |
| consensus | 0.582461 | 1.173664 | 0.987895 | 0.991686 | 0.110143 | 0.468074 | 0.123378 |

**Observation:**  
Uniform is numerically identical to baseline on the fixed MDS subset after affine alignment, as expected for global positive rescaling. Shuffled and consensus both worsen MDS mean/p95 and KS, with consensus worse than shuffled. The isotonic diagnostic shows the same ordering on both uniform and local held-out pairs. The shuffled multiplier assignment has total edge weight `6,665,088.973521` versus consensus `6,669,263.784317` (0.0626% lower); this small mismatch is retained as a control limitation.

**Interpretation:**  
The frozen low-common-neighbor edge-length penalty did not improve the Stage-1 distance geometry in the pre-FCF chart proxy. Its failure is evidence against proceeding to the expensive FCF/Stage-2 comparison for this coefficient and seed. Post-result design review identified that "low common support implies this graph edge must be lengthened" was an unverified assumption added by this experiment, not a step supplied by SC2-PCR. This result does not test whether a properly defined third-party compatibility/consensus mechanism can improve Channel Charting. A separate read-only, GT-assisted exploratory check of original edges found smaller median physical endpoint separation for low-support than high-support edges in 8/10 original fused-edge-length bins; this is not a frozen endpoint or a general bad-edge label.

**Falsification / failure criterion:**  
MDS gate failed: consensus mean was not at least 0.005 m below each control; consensus p95 exceeded baseline by more than 0.01 m; and consensus KS exceeded baseline by more than 0.01. CT and TW guardrails passed. The secondary isotonic diagnostic was negative on all four specified comparison conditions, but it was not used as the train/no-train decision.

**Limitations / caveats:**

- The MDS result is a fixed stride-10 2,100-sample proxy from full D_geo, not a full-N MDS or FCF result.
- No FCF/Stage-2 performance conclusion or out-of-sample claim can be made from this stopped pilot.
- The shuffled control preserves multiplier distribution within length/degree bins, but its realized total edge weight is slightly below consensus as recorded above.
- The fixed coefficient 0.5 and seed were not tuned after observing the result.
- The original research question about the broader SC2 idea was not identified by this penalty experiment. `negative` applies only to the implemented rule; it must not be cited as a failure of third-party support in general.

**Status:** `negative`

**Registry entry updated:** `yes`

---

## 2026-09-25 — EXP-SC2-STAGE2-ADAPTIVE-FIDELITY-001

**Research question / hypothesis:**  
Does second-order local agreement identify which existing `Y0` samples should be held more strongly during the advisor Stage-2 refinement, beyond first-order agreement and a shuffled assignment?

**Phase / task:**  
SC2-inspired Stage-2 adaptive-fidelity pilot; full-size CG-A transductive `Y0 -> Z` refinement.

**Code entry point:**  
`methods/sc2_stage2_adaptive_fidelity/run_pilot.py`

**Focused tests:**  
`tests/test_sc2_stage2_adaptive_fidelity.py` — 4/4 passed. The tests cover sparse symmetric-union semantics, `H/S` witness scores, average-rank ties and constant scores, and exact `a_i=1` agreement with the existing solver update on a tiny graph.

**Configuration:**

```text
dataset: datasets/adp_old_shard4; full CG-A transductive sample set
stride / N: 1 / 20,997
seed: 10
Stage-2 graph: existing build_local_scaled_graph, k=128, float32, unchanged W
lambda / iterations: 0.15 / 200 Jacobi iterations
compatibility: original directed geodesic top-64; epsilon=1e-8; log-median scale; tau=median residual + epsilon
SC2 support: sparse symmetric union; H cosine over all sparse witnesses; S=C*H
fidelity: average-rank percentile p=(rankdata(q, average)-1)/(N-1), a=0.5+p
variants: uniform, first_order, sc2, sc2_shuffled
evaluation: same Python random.sample seed-10 subset, int(0.2*N)=4,199, CT/TW/KS; all-N affine error summaries
```

**Executed commands and environment:**

```text
D:\anaconda3\envs\diffusion_charting\python.exe -m unittest tests.test_sc2_stage2_adaptive_fidelity -v
D:\anaconda3\envs\diffusion_charting\python.exe -u methods/sc2_stage2_adaptive_fidelity/run_pilot.py
Python 3.11.15; numpy 2.4.6; scipy 1.17.1; torch 2.11.0+cu128; NVIDIA GeForce RTX 3050 Laptop GPU; device=cuda:0
```

The original strict uniform coordinate check (`max_abs <= 1e-4`) observed `2.74658e-4` in the current environment. A current original solver run showed the same inherited drift relative to the saved baseline; the adaptive uniform update matched the current solver within the same drift envelope. The parent-approved verification gate used `max_abs <= 5e-4`, `mean_abs <= 5e-5`, and absolute metric/error agreement `<=1e-5`, all passed. The initial strict failure is retained in `results/sc2_stage2_adaptive_fidelity_pilot/protocol_mismatch.json`; source-vs-source evidence is in `protocol_reproduction_audit.json`.

**Input artifacts:**

- `results/adp_old_fused/stride1_n20997_seed10/geodesic.npy`
- `results/adp_old_fused/stride1_n20997_seed10/geodesic_top512.npy`
- `results/adp_old_fused/stride1_n20997_seed10/distance_data.npz`
- `results/adp_old_siamese/stride1_n20997_seed10/embedding.npz`
- `results/diffusion_refined_charting/stride1_n20997_k128_lambda0.15_seed10/{embedding.npz,result.json}`

**Output artifacts:**

- `results/sc2_stage2_adaptive_fidelity_pilot/REPORT.md`
- `results/sc2_stage2_adaptive_fidelity_pilot/config.json`
- `results/sc2_stage2_adaptive_fidelity_pilot/{uniform,first_order,sc2,sc2_shuffled}.json`
- `results/sc2_stage2_adaptive_fidelity_pilot/{uniform,first_order,sc2,sc2_shuffled}_embedding.npz`
- `results/sc2_stage2_adaptive_fidelity_pilot/{first_order,sc2}_reliability.npy` and fidelity/permutation arrays
- `results/sc2_stage2_adaptive_fidelity_pilot/diagnostics.json`
- `results/sc2_stage2_adaptive_fidelity_pilot/{reliability_quintile_vs_y0_error,reliability_quintile_vs_baseline_improvement,four_method_error_cdf}.png`

**Evaluation protocol:**  
Transductive full-N Stage-2 refinement. `W`, `D_geo`, Stage-1 graph, FCF training, and `Y0` were unchanged. Ground-truth positions entered only affine diagnostics and evaluation after reliability construction and solving. No diagnostic gate, tuning, or out-of-sample claim was used.

**Key results:**

| variant | CT | TW | KS | mean m | p95 m |
|---|---:|---:|---:|---:|---:|
| uniform | 0.998219594 | 0.998222527 | 0.065064573 | 0.318748 | 0.634639 |
| first_order | 0.998219431 | 0.998214786 | 0.064944251 | 0.320131 | 0.634043 |
| sc2 | 0.998215811 | 0.998226793 | 0.064834713 | 0.322221 | 0.648274 |
| sc2_shuffled | 0.998220783 | 0.998221842 | 0.064948429 | 0.318710 | 0.633945 |

**Observation:**  
The SC2 variant worsened all-N mean error relative to uniform by `0.003474 m`, first-order by `0.002090 m`, and shuffled by `0.003512 m`; its P95 was `0.013635 m` above uniform. The frozen screening rule therefore failed all four conditions. The shuffled control was nearly equal to uniform in mean (`-0.000038 m` net improvement versus uniform), so the pilot does not support a second-order fidelity benefit.

The reliability diagnostic was mixed: q_SC2 Spearman rho was `0.081392` with Y0 error and `-0.011574` with saved-baseline improvement; q_FO rho was `-0.030812` and `-0.057270`, respectively. SC2 quintile mean baseline improvements were `0.119338`, `0.085547`, `0.076627`, `0.082441`, and `0.089038 m` from low to high reliability. Both score vectors had no exact ties in this run; average-rank tie handling and constant-score behavior remain covered by the focused tests.

**Interpretation:**  
Under the frozen full-size transductive protocol, second-order local agreement did not identify a useful node-fidelity assignment beyond first-order or shuffled controls. This is a valid negative pilot for the specified adaptive rule; it does not establish that every reliability-guided Stage-2 mechanism is ineffective.

**Falsification / failure criterion:**  
Pilot support required SC2 mean at least `0.005 m` lower than uniform, first-order, and shuffled controls, with SC2 P95 no more than `0.005 m` above uniform. None of these thresholds passed.

**Limitations / caveats:**

- This is transductive evidence on the existing chart and graph; it provides no held-out or unseen-CSI deployment evidence.
- The current environment's original solver has a small saved-embedding coordinate drift documented in `protocol_reproduction_audit.json`; metrics and all-N error summaries agree within the amended verification tolerance.
- No tau, percentile offset, graph k, lambda, iteration count, or seed was tuned after observing outcomes.

**Status:** `negative`

**Registry entry updated:** `yes`

---

## 2026-09-25 — EXP-SC2-TEMPORAL-FRAGMENTS-001

2026-10-08：按用户要求清除本次跨时间匹配实验的专用代码、测试、任务和结果。原始数据、baseline及其他实验保留。本条仅记录删除事实，不再保留结果明细或失效链接。

## 2026-10-08 — EXP-SC2-STAGE1-MULTIVIEW-RELATION-CHECK-001

**执行后范围纠正：** 实际评分 `1-|d_A-d_B|/d_full` 测量的是两个子视角距离数值是否接近；两个距离都大时也可得高分。它没有直接检验“两个视角都认为相近”，更没有检验 SC² 的第三方关系共同支持。以下负结果只针对这个具体评分；原先的问题仍未得到有效检验。先完成上一轮共同邻居边长实验的损害归因，再考虑多视角。

**Research question / hypothesis:**  
Among existing Stage-1 directed k=20 relations that are ADP-dominated and span more than 60 seconds, does agreement between two fixed disjoint array views provide relation-level reliability information beyond the original full-array ADP?

**Phase / task:**  
Stage-1 relation-only diagnostic; no graph intervention.

**Code entry point:**  
`diagnostics/sc2_stage1_multiview_check/run_pilot.py`

**Configuration:**

```text
dataset: datasets/adp_old_shard4; direct H/timestamp/position reads with shape/order checks
stride / N: 1 / 20,997
Stage-1 graph: results/phase2c_h1_h2/stage1_graph.npz; directed k=20
candidate: unique unordered original graph relation, abs(dt)>60 s, saved ADP <= saved gamma*abs(dt)
array views: A={0,2}, B={1,3}; gamma=12.983333587646484
stratification: full ADP decile x time-gap quartile x endpoint-density tertile
shuffle: seed=10 within fixed strata; endpoint bootstrap seed=11, 200 replicates
labels: physical <0.25 m near; >0.5 m far; middle excluded from AUC
```

**Input artifacts:**

- `datasets/adp_old_shard4/{H.npy,timestamps.npy,UE_pos.npy}`
- `results/adp_old_fused/stride1_n20997_seed10/{adp.npy,fused.npy,distance_data.npz,result.json}`
- `results/phase2c_h1_h2/stage1_graph.npz`

**Output artifacts:**

- `results/sc2_stage1_multiview_check/REPORT.md`
- `results/sc2_stage1_multiview_check/{config.json,pre_gt_check.json,summary.json,candidate_scores.npz,evaluation.npz,agreement_quintiles.csv,examples.csv}`
- `tests/test_sc2_stage1_multiview_check.py`

**Evaluation protocol:**  
Transductive relation diagnostic on the existing sample set. Candidate selection, view scores, strata, and shuffle control were fixed and saved before `UE_pos.npy` was read. GT was used only for offline labels, bootstrap, examples, and descriptive checks. No D_geo recomputation, MDS, FCF training, Stage-2 solve, or graph modification.

**Key results:**

| metric | value |
|---|---:|
| candidate pairs | 23,973 |
| view decomposition max abs error | 9.54e-06 |
| full ADP AUC (near vs far) | 0.7933 |
| view agreement AUC | 0.4528 |
| fixed full ADP + agreement AUC | 0.6949 |
| shuffled agreement combined AUC | 0.6865 |
| full ADP top-5% near fraction | 0.9583 |
| combined top-5% near fraction | 0.9883 |
| shuffled combined top-5% near fraction | 0.9825 |
| stratified top-minus-bottom near fraction | -0.0024; bootstrap CI [-0.0243, 0.0196] |

**Observation:**  
On this candidate set, the fixed agreement score has raw AUC below 0.5; its high-score quintile also has lower near fraction than its low-score quintile. The combined top-5% number is higher than full ADP, but the shuffled control is nearly as high (+0.58 percentage points relative to shuffled), while global combined AUC is lower than full ADP.

**Interpretation:**  
This does not support integrating this agreement score into Stage 1 or Stage 2. It rules out the tested candidate definition and score as a useful incremental relation test; it does not rule out every multiview or SC² relation-level construction.

**Falsification / failure criterion:**  
Failed all three substantive frozen requirements except the top-5% guard: stratified difference did not reach +0.05 with positive CI lower bound; combined AUC did not beat full ADP or shuffled control by +0.02; the top-5% guard passed.

**Limitations / caveats:**

- The data are transductive and do not establish held-out or deployment performance.
- The .25 m and .5 m thresholds were fixed prospectively for this diagnostic; they were only posthoc descriptive cuts in the earlier pilot.
- Agreement quintiles and Spearman correlations in the report are posthoc descriptive analyses and did not alter the decision.
- The two array views are correlated observations, not independent sensors.

**Status:** `negative`

**Registry entry updated:** `yes`
## 2026-10-08 — EXP-SC2-STAGE1-HARM-ATTRIBUTION-001

**Research question / hypothesis:**  
Is the MDS degradation from the old all-edge common-neighbor length penalty mainly caused by edges with time gap >60 s for which the ADP branch supplies the smaller fused distance?

**Phase / task:**  
Stage-1 common-neighbor penalty harm attribution; relation diagnostic only.

**Code entry point:**  
`diagnostics/sc2_stage1_harm_attribution/run_pilot.py`

**Configuration:**

```text
dataset: datasets/adp_old_shard4; stride1 / N=20,997
Stage-1 graph: results/phase2c_h1_h2/stage1_graph.npz; directed k=20; shortest_path directed=False
edge artifact: results/sc2_stage1_consensus_pilot/edge_scores.npz, aligned by (row,col) edge key
gamma: 12.983333587646484; T: abs(dt)>60 s and ADP <= gamma*abs(dt); R: complement
multiplier: original fused edge weight * [1 + 0.5 * (1 - support)] on selected group only
MDS: sources=np.arange(0,20997,10), 2100 samples, seed=10, max_iter=80, n_init=4
```

**Input artifacts:**

- `results/phase2c_h1_h2/stage1_graph.npz`
- `results/sc2_stage1_consensus_pilot/edge_scores.npz`
- `results/adp_old_fused/stride1_n20997_seed10/{adp.npy,fused.npy,geodesic.npy,result.json,distance_data.npz}`
- `results/sc2_stage1_consensus_pilot/{baseline,consensus}/{geodesic.npy,mds.npz}`

**Output artifacts:**

- `results/sc2_stage1_harm_attribution/REPORT.md`
- `results/sc2_stage1_harm_attribution/{pre_gt_manifest.json,edge_groups.npz,group_summary.json,reproduction_check.json,mds_metrics.json,attribution.json,fixed_examples.json,gt_path_diagnostic.json,result.json}`
- `results/sc2_stage1_harm_attribution/variants/{baseline_recomputed,T_only,R_only}/`
- `tests/test_sc2_stage1_harm_attribution.py`

**Evaluation protocol:**  
Diagnostic-only transductive fixed-subset MDS. Edge-key alignment, T/R groups, graph variants, and baseline source-path reproduction were completed before GT labels were used. GT positions were used only for affine MDS evaluation, fixed examples, and the posthoc T-low-support path check. No edge support change, D_geo full-matrix rewrite, FCF training, or Stage-2 run.

**Key results:**

| metric | value |
|---|---:|
| directed T edges | 36,811 / 419,940 |
| T share of weighted multiplier increment | 24.22% |
| baseline MDS mean / P95 (m) | 0.458130 / 0.961453 |
| T_only MDS mean / P95 (m) | 0.580495 / 1.193341 |
| R_only MDS mean / P95 (m) | 0.465136 / 0.961160 |
| consensus MDS mean / P95 (m) | 0.582461 / 1.173664 |
| full mean increment (m) | 0.124331 |
| T_only mean increment (m) | 0.122366 |
| R_only mean increment (m) | 0.007007 |
| T_only P95 increment (m) | 0.231889 |
| baseline source/MDS reproduction | passed; max differences 0 |

**Observation:**  
Applying the old multiplier only to T produces almost all of the full degradation in mean error and a larger P95 increase than the full penalty. R_only is close to baseline and slightly improves P95. T_only improves 519 samples and harms 1,581; the collateral pattern is heterogeneous. Sample 4540 changes from 0.4813 m baseline to 9.1947 m under T_only, while sample 5720 changes from 8.8965 m to 0.1016 m under T_only.

**Interpretation:**  
The frozen attribution rule supports T edges as the main harm source for this old intervention. It does not prove every T edge is bad, does not give an exact causal contribution percentage because T/R effects can interact, and does not establish a new improvement method.

**Falsification / failure criterion:**  
Passed the predeclared attribution rule: T_only increment 0.122366 m is at least half of the full 0.124331 m increment and exceeds R_only by more than 0.01 m; full and T_only P95 both worsened.

**Limitations / caveats:**

- The result is a fixed 2,100-source MDS diagnostic and does not claim full-N FCF or Stage-2 improvement.
- The T/R split is a GT-free rule, but the selected low-support physical-near path examples are posthoc GT-assisted diagnostics.
- `D_geo` path changes are in fused-distance units, not meters.
- T/R ablations are not additive causal contributions; interaction can remain.

**Status:** `confirmed`

**Registry entry updated:** `yes`
---

### EXP-MV-G0-001 — 多视角 Stage-1 边风险 Gate 0 回归

**Date:** 2026-10-09  
**Research question:** 四阵列 ADP、原 fused 20-NN 图、边分数与零干预最短路能否在本机无 GT 参与地复现？  
**Phase / task:** 多视角边不确定性方案 Gate 0；不含 Gate 1–3 效果验证。  
**Code entry point:** `methods/multiview_edge_uncertainty/gate0.py`；测试 `tests/test_multiview_adp_gate0.py`。  
**Configuration:** `datasets/adp_old_shard4`，stride=1，N=20997；baseline `neighbors=20`、`gamma=12.983333587646484`；per-view `k_view=20`；GPU block=64；`kappa=0/0.5` 仅作图不变量检查；32 个源点为 `np.linspace(0,N-1,32,dtype=int)`；无随机种子或 GT 调参。  
**Input artifacts:** `datasets/adp_old_shard4/{H.npy,timestamps.npy}`、原始 cf02 TFRecord/offsets、`datasets/specv2.0.json`、`results/adp_old_fused/stride1_n20997_seed10/{adp.npy,fused.npy,geodesic.npy,distance_data.npz,result.json}`。  
**Output artifacts:** `results/multiview_edge_uncertainty/{REPORT.md,edge_risk/gate0_regression.json,edge_risk/edge_observations.npz}`。  
**Protocol:** GT-free implementation/provenance regression；非方法性能实验，comparable_group 不适用。无 MDS、FCF 或 Stage-2 求解。  
**Key results:** 13/13 Gate 0 machine checks and 6/6 unit tests passed；四视角 ADP 全矩阵求和对缓存最大绝对差 `9.536743e-6`、超容差元素 0；重建 `419940` 条有向边，32×20997 个测地距离最大差 0；唯一无向候选边 `228773`，ADP-source `49626`；`kappa=0` 图逐项相同。  
**Interpretation:** 数据、图语义、四视角计算和零干预路径在本机可复现。原图有 2759 对互为邻居边值存在最大 `1.144409e-5` 的两向差；风险图保持原值并对两向使用相同乘因子。初次把两向绝对代价完全相等设为条件导致误报，修正实现检查并完整重跑；未对称化原图。  
**Limitations:** 只抽查 1024 条边的单 view 数值对称性及 cf02 前两条保留原始样本；ADP 求和覆盖全矩阵。无 GT 边有效性、测地干预收益、`Y0`/`Z` 效果证据。  
**Status:** `confirmed`（仅 Gate 0）。  
**Registry entry updated:** `yes`。

---

### EXP-G1-ACF-SANITY-001 — Graph/Identity implementation and resource checks

**Date:** 2026-10-16  
**Question:** Can both G1 modes preserve baseline graph provenance, train with finite gradients, and fit the available 4 GB GPU before full-N resource approval? This entry does not test final method benefit.  
**Code:** `run_g1_acf_graph_fcf.py`, `g1_acf_graph_utils.py`; checks in `tests/test_g1_acf_graph_fcf.py`.  
**Configuration:** DICHASUS `adp_old_shard4`; seed 10; Sanity stride10/N2100, graph k35, Graph and Identity each 100 steps, 65,536 training pairs/step, 200,000 validation pairs, eval every 10; Pilot stride1/N20997, graph k20, Graph only 5 steps with CPU-memmap feature streaming. AdamW lr 1e-3, weight decay 1e-5, 2 GCN layers, hidden 512, dropout 0.1.  
**Inputs:** `datasets/adp_old_shard4/`; `results/adp_old_fused/stride10_n2100_seed10/`; `results/adp_old_fused/stride1_n20997_seed10/`.  
**Outputs:** `results/g1_acf_graph_fcf/REPORT.md`; three run folders named `stride10_n2100_seed10_{graph,identity}_sanity` and `stride1_n20997_seed10_graph_pilot`.  
**Protocol:** Same-node transductive Sanity/resource Pilot. GT used only after checkpoint selection for diagnostic chart metrics. No full-N efficacy claim.  
**Results:** 5/5 unit checks passed; full-N 32-source geodesic replay mismatch 0, max difference 0; stride10 Graph/Identity mean errors 3.5470/3.1737 m, P95 6.6317/6.1681 m after 100 steps; full-N five-step Graph train time 18.80 s, peak PyTorch GPU allocation 707,292,672 bytes.  
**Interpretation:** Both modes execute and the full-N streaming path fits observed GPU memory. Short-run metrics do not decide Graph versus Identity; complete paired training is pending resource approval.  
**Limitations:** One seed; stride10 has different graph k and distance scale; full-N Pilot has only five steps; no common B0/G1 full-N comparison yet.  
**Status:** `preliminary` (Sanity and Pilot only).  
**Registry entry updated:** `no` (temporary checks; no completed method result).

---

### EXP-G1-ACF-FULL-001 — Full-N Graph versus Identity

**Date:** 2026-10-16  
**Question:** Does fused-graph neighborhood propagation improve the real DICHASUS channel chart beyond the identical G1 model with unit adjacency, and does the complete G1-Graph outperform saved B0?  
**Code:** `run_g1_acf_graph_fcf.py`, `g1_acf_graph_utils.py`, `compare_g1_acf_results.py`; tests `tests/test_g1_acf_graph_fcf.py`.  
**Configuration:** `datasets/adp_old_shard4/`, stride1/N20997, seed10; saved Stage-1 fused kNN graph with actual k20 and original `geodesic.npy`; 26,624-D B0 ACF; Linear→BN→2 residual sparse GCN→2D, hidden512/dropout0.1, 14,162,434 parameters. Only adjacency changes: symmetric affinity on fused graph support versus I. AdamW lr1e-3, weight decay1e-5, warmup100/cosine to1e-5, gradient clip5; 65,536 training pairs/step, 200,000 fixed validation pairs; max2000 steps, eval every20, 12-eval patience. CPU-memmap feature streaming with chunk128. Both modes ran 2000 steps, best step1920. Train/validation pair hashes match across modes.  
**Inputs:** `datasets/adp_old_shard4/`; `results/adp_old_fused/stride1_n20997_seed10/{fused.npy,geodesic.npy,distance_data.npz,result.json}`; B0 `results/adp_old_siamese/stride1_n20997_seed10/embedding.npz`.  
**Outputs:** `results/g1_acf_graph_fcf/{stride1_n20997_seed10_graph_full,stride1_n20997_seed10_identity_full}/` (best/last weights, full history, embedding, metadata, run command); `results/g1_acf_graph_fcf/comparison/` (CSV, protocol, common subset, raw/affine four-panel charts, aligned charts); `results/g1_acf_graph_fcf/REPORT.md`.  
**Protocol:** Same-node transductive chart quality. CT/TW/KS on identical seed10 20% subset of 4,199 nodes; affine-aligned Mean/Median/P95 on all 20,997 nodes. GT only for post-training evaluation/plots, never model graph/loss/checkpoint.  
**Results:** B0 CT/TW/KS 0.995712/0.995987/0.084027, Mean/Median/P95 0.409347/0.372228/0.855793 m; Identity 0.996966/0.996868/0.079401, 0.398343/0.361716/0.843197 m; Graph 0.996947/0.996849/0.079340, 0.399771/0.364578/0.846337 m. Graph/Identity training times 6,224.64/5,292.06 s, PyTorch peak allocations 707,461,120/698,310,656 bytes; combined training 3.199 GPU hours, below confirmed ~6 h cap.  
**Interpretation:** Complete G1-Graph modestly improves all listed metrics versus B0, but the same architecture without neighbor propagation is slightly better on CT/TW/Mean/Median/P95; Graph is ahead only on KS by 0.000062. Thus this fixed experiment does not support a positive neighborhood-propagation effect. Visually both G1 charts are similar after affine alignment; Graph has no clear unique structural repair.  
**Limitations:** One seed/dataset, fully transductive; pair holdout is not a node-held-out generalization test. The existing advisor Stage-2 Z baseline has lower mean error; G1+Stage-2 was not run. Historical B0 training time/VRAM unavailable because its saved `result.json` is absent.  
**Status:** `negative` for graph-propagation benefit under this frozen setting; valid positive package comparison versus B0 is recorded separately.  
**Registry entry updated:** `yes`.

---

### 2026-10-17 — G1 code organization, plotting correction, and B0 artifact recovery

Maintenance only; no new training or method experiment. G1 code moved from the three root entry files to `methods/g1_acf_graph_fcf/{run.py,graph_utils.py,compare.py}`; tests remain `tests/test_g1_acf_graph_fcf.py`. Current entry points use `python -m methods.g1_acf_graph_fcf.run` and `.compare`; historical run commands above remain unchanged.

The four-column plots now directly invoke `run_adp_siamese.plot_chart`, with the original point size, colors, grid, equal axes and dpi. Only optional external axes and raw-chart support were added to that helper; the default two-panel plot matches the previous implementation pixel for pixel. Five relocated tests passed. Raw and affine figures were regenerated from saved embeddings; all six comparison metrics are unchanged.

Correction to the missing-record observation in EXP-G1-ACF-FULL-001: original B0 `result.json` and `chart.png` were found in master/origin/master commit `61178b5`, absent from the current `codex/new-stage1` ancestry. Both were restored byte for byte to `results/adp_old_siamese/stride1_n20997_seed10/`; model/embedding were preserved. Historical metrics exactly match the G1 common evaluation. The original `elapsed_s=1633.554753` is total pipeline time, not a separately measured training time; peak VRAM remains unavailable.

The user retains G1 as an exploration direction. The current frozen Graph/Identity result remains unchanged and does not terminate the whole direction; no follow-up training was launched.

---

### EXP-G1-S2-001 — Frozen Stage-2 operator on B0 and G1 charts

**Date:** 2026-10-17  
**Question:** Does the Stage-1 G1 package improvement survive the same advisor Stage-2 refinement, and does Graph add value over trained Identity after S2?  
**Code:** `methods/g1_acf_graph_fcf/run_stage2_comparison.py`, shared `compare.py`; original imported S2 functions unchanged.  
**Configuration:** adp_old_shard4, stride1/N20997, seed10; D_geo from the existing full-N distance run; fresh shared top512, graph_k128, lambda0.15, 200 original Jacobi iterations. Exactly one shared graph object; no training or parameter selection.  
**Inputs:** Original B0/Identity/Graph `embedding.npz`, D_geo, saved common 4199-node metric subset. Historical S2 JSON read from Git61178b5; historical embedding/cache read-only.  
**Outputs:** `results/g1_acf_graph_fcf_followup/e1_stage2/` (shared cache, three new embeddings/results/configs/charts, six-method comparison); root `REPORT.md`, `preflight.json`, `e1_console.log`.  
**Protocol:** Same-node transductive; CT/TW/KS on identical 4199 nodes, all-N affine Mean/Median/P90/P95/Max; GT only after refinement.  
**Results:** B0+S2 KS/Mean/P95=0.065064569/0.318747967/0.634639800; Identity+S2=0.066559264/0.323004276/0.648439467; Graph+S2=0.066692393/0.324695349/0.649173915. Graph+S2 minus Identity+S2 Mean/P95=+0.001691073/+0.000734448 m; only Max better by0.013983 m. Fresh neighbor calculations match each other and the old cache exactly; B0+S2 Mean reproduces historical value. GPU Jacobi repeat raw max difference0.000274658, within recorded absolute/relative tolerances. Full E1 processing/evaluation27.17s; sparse solver peaks111771648 bytes, top-k construction peak not separately captured.  
**Interpretation:** Stage-1 improvements do not translate into a better fixed complete S2 pipeline. This fixed Graph configuration has no extra combined benefit on primary KS/Mean/P95.  
**Limitations:** Single seed/dataset, transductive; historical S2 hyperparameter-selection provenance not independently verified. An initial CUDA memory-stat reset before initialization failed before graph calculation; helper initialization fixed, failed log retained; no algorithm change. Original files/cache unchanged.  
**Status:** `negative` for improved G1+S2 pipeline / propagation increment.  
**Registry entry updated:** `yes`.

### EXP-G1-PROP-001 — Frozen checkpoint adjacency sensitivity

**Date:** 2026-10-17  
**Question:** Does the trained Graph checkpoint depend on adjacency, and are its residual updates negligible? This is not a retrained graph-ablation comparison.  
**Code:** `methods/g1_acf_graph_fcf/diagnose_graph_propagation.py`, forward hooks only; original GCN class unchanged.  
**Configuration:** Same full-N CSI/ACF cache, fused k20 propagation graph, best step1920 Graph/Identity weights, eval/no_grad, chunk128. First reproduce Graph@Graph and Identity@I, then Graph@I; reuse original margin51.4951897 and exact200000 validation-pair hash. No training, no BN updates, no checkpoint choice.  
**Outputs:** `results/g1_acf_graph_fcf_followup/e2_propagation/` (three probe results/embeddings/configs, native checks, per-layer node arrays and distributions, raw displacement arrays, validation keys, full comparison/charts); root report and `e2_console.log`.  
**Results:** Native Graph/Identity raw max reproduction differences0.0000610352/0; original validation stress exactly reproduced. Graph@Graph stress1.872657776 versus Graph@I3.208952188 (+71.358%). Raw displacement Mean/Median/P95=15.747149/12.967083/37.699429 raw-chart units, relative centered RMS9.25946%. Graph native layer1/2 delta/h medians0.859600/0.597705; (Ah-h)/h medians0.357220/0.239588. Graph@I affine Mean/P95/Max=0.491343/1.025123/8.185935 m versus native0.399771/0.846337/1.463179. All frozen parameters/BN buffers unchanged. E2 total38.80s, measured inference peaks400580096 bytes.  
**Interpretation:** Adjacency materially affects this trained network; simple explanations of no graph dependence or vanishing residual updates are unsupported. Dependence is not additional benefit over the separately trained Identity model, which remains better on major geometric endpoints.  
**Limitations:** Post-training adjacency swap changes input distribution and is not a retraining causal control. Original validation sampler sorts/truncates keys: all smaller pair endpoints are<=6155, so stress is on this fixed biased set, not uniform all-pairs risk. Sampler/checkpoints deliberately preserved. Optional Identity@Graph and GT spatial map not executed.  
**Status:** `confirmed` (frozen-model sensitivity only).  
**Registry entry updated:** `yes`.

---

### EXP-S2-P0-20261019 — Repo, incidence, Jacobi and same-target CG preflight

**Date:** 2026-10-19  
**Scope:** User authorized P0 only. No Robust-IC/Framelet/DoT, no P1 or training.  
**Question:** Are the immutable data/graph and original J200 baseline reproducible, does the incidence implement the same Laplacian, and does CG solve exactly the same quadratic target?  
**Code / command:** `diagnostics/stage2_p0/preflight.py`; `python -m diagnostics.stage2_p0.preflight --device cuda:0 --out-dir results/stage2_modern_graph_20261019/p0_preflight`. Tests `python -m unittest tests.test_stage2_p0 -v` (3 passed).  
**Inputs/config:** Three saved B0/G1 embeddings, original D_geo/distance_data and historical S2 JSON, fixed seed10/4199-node metric subset; N20997, stride1. Original top512/G128/local-scale weights/lambda0.15/Jacobi200; CG float64 on the label-free centered/isotropic gauge, same `(I+0.15L)Z=Y`, diagonal preconditioner, rtol1e-10/atol0/maxiter5000. Each undirected edge appears once in D.  
**Output:** `results/stage2_modern_graph_20261019/p0_preflight/`: REPORT, G0 JSON/CSV (14 individual rows with actual/threshold/evidence/commands), input_file_hashes.csv and manifest, saved G128/D/L, solver checks and residual histories, delta0.json/chart_gauge.npz, J/CG embeddings/results/configs/charts, standard comparison. Console log at the independent family root.  
**Results:** G0 14/14 PASS. G1283088334 symmetric entries/1544167 undirected edges; no self loops, min weight0.0826393, unweighted degrees128–397. Float64 DTWD/CSR matvec relative error5.344659e-16, maxabs(L1)=0. Historical J KS/Mean/P95 differences1.398394e-9/0/5.960464e-7m. CG converged in42 iterations per coordinate; max true normalized/raw relative residual9.950825e-11. J residual2.345190e-6. J versus CG KS=0.065064572353/0.065064568674, Mean0.318747967482/0.318747938339m, P950.634639978409/0.634639233193m. δ0=0.09517167367991 from J, numerator3220.179574061183/denominator33835.48328562125.  
**Protocol / interpretation:** Fixed same-node transductive evaluation; no GT in graph/solver/δ0. CG is a numerical control, no new regularizer or meaningful geometric gain. Full execution31.30s; J0.477s, CG0.663s; construction peak PyTorch allocation1.805GiB, J solve0.104GiB, CG onCPU. All29 protected source/metadata/code files and source distance-directory contents unchanged, including AGENTS.  
**Limits / stop:** Only P0; no P1 framelet test implemented. Future normalized-Laplacian zero mode is sqrt(degree)*1, separate from this unnormalized L1 check. No gate relaxation, baseline modification or historical overwrite. Stopped for user review; current branch codex/new-stage1, startHEAD f63652f, not committed/uploaded.  
**Status:** `confirmed` (P0 baseline/engineering gate only).  
**Registry entry updated:** `yes`.

---

## Current / Future Experiment Entry Template

Copy this section for every completed experiment.

### EXP-XXX — <Short Name>

**Date:** YYYY-MM-DD

**Research question / hypothesis:**  
<What exactly is being tested?>

**Phase / task:**  
<e.g. Phase 2C — H1 Diagnostic>

**Code entry point:**  
`path/to/script.py`

**Configuration:**

```text
dataset:
subset:
stride:
N:
seed:
key hyperparameters:
```

**Input artifacts:**

- `...`
- `...`

**Output artifacts:**

- `...`
- `...`

**Evaluation protocol:**  
<transductive / held-out / diagnostic-only / test split / etc.>

**Key results:**

| metric | value |
|---|---:|
| ... | ... |

**Observation:**  
<What happened?>

**Interpretation:**  
<What does this support or rule out?>

**Falsification / failure criterion:**  
<Was the pre-defined criterion met?>

**Limitations / caveats:**

- ...
- ...

**Status:** `confirmed | preliminary | negative | invalid | missing-records`

**Registry entry updated:** `yes / no`

---


### EXP-S2-P1-MATH-20261019 — IC / Framelet / Edge mathematical smoke

**Date / scope:** 2026-10-19; explicitly authorized P1 only, after accepted P0; stopped before P2A. No G1 retraining or real-data method fitting.
**Question:** Do frozen IC8-P/IC9-P/IC9-G, FN-P/FN-G-nu/FN-G-delta and paper SEF/FE/DoT objectives agree with independent small-graph numerical references and satisfy feasibility/stationarity?
**Code / config:** `diagnostics/stage2_p1/run.py`, `reference.py`, `report.py`; solvers `methods/stage2_modern_graph/`; tests `tests/test_stage2_p1_math.py`. Synthetic irregular weighted path+two-hop N16/32/64, Edge256; seed20261019+N. Gauge centered/RMS chart, alpha=.5, beta=.5, eta=.09517167367990964 from P0, synthetic nu=.1 (not B0/P2/P3 selection). Group-delta <=8 new GT-free solver calls. CVXPY1.9.3/CLARABEL, tolerance1e-9. Active low2/high1/high2, low penalty0; exact PCG U system. Edge signed dense S only at256, lambda1=lambda2=1, author/prior initializations, initial5000+one15000 retry; no third attempt.
**Inputs / outputs:** P0 delta0 and manifest read-only; fixed author source commit4db39904e4099d82cef686d4732c33150814ece4. All sources/hash/environment/config/commands and artifacts under `results/stage2_modern_graph_20261019/p1_math_sanity/` (main REPORT, gates, reference_objectives, histories, native/reference arrays, charts and comparison). No historical inputs/results overwritten.
**Results:** 82 checks:73 PASS/9 FAIL. IC max objective gap2.995006e-9, feasibility1.995505e-17, stationarity1.480038e-6. FN max gap4.040297e-7, primal1.993402e-7, PCG9.990262e-12, dense-U error9.850982e-12. Selected minimum n3/3/4; sqrt(degree)*1 zero mode, ordinary1 auxiliary only. All group gauge equivariance <=9.127340e-12. SEF gap5.337677e-4/equality3.867752e-4 reaches frozen math thresholds but solver remains NONCONVERGED. FE author/prior row residual .01953216/.003098536; DoT .02374957/.03061898, >1e-3; all four NONCONVERGED after retry. FE/DoT joint objective nonconvex; only convex S and U block references, no global optimum claim.
**Protocol / limits:** Synthetic math only, not DICHASUS or a main method ranking. Charts use common_metrics and original plot_chart; seed10 20% metric nodes, affine error fit/evaluation all same synthetic nodes, complete metrics includingP90/Max. Failed Edge charts labeled NONCONVERGED. No GT parameter selection. Small-N chart metrics are artifact checks, not evidence of antenna-array performance. 5 unit tests PASS; formal IC/FN/Edge wall9.545380/26.874687/735.936634s including reference/plotting, CPU only/GPU allocation0; final CPU peak unknown (observed high-water868552704bytes). No NaN/OOM or PCG/reference failure; final Edge constraint oscillation remains a numerical blocker, not evidence against paper methods.
**Status:** `confirmed` for IC/FN math and observed Edge nonconvergence; Edge outputs `invalid` for performance ranking. Not a valid negative performance result. P1 execution delivered with BLOCKER; P2A/P2B/P3/P4/P5 STOP awaiting review. Branch codex/new-stage1, HEAD f63652f, no commit/upload.
**Registry entry updated:** yes.


### EXP-P1-EDGE-SOLVER-REPAIR-20261019 — Independent numerical audit and one bounded penalty strategy

**Scope:** User authorized only SEF/FE/DoT numerical diagnosis/repair; no IC/FN Node reruns/modification, no Channel Chart/affine/CT/TW/KS/localization metrics, no later phases.
**Entry / inputs:** `diagnostics/stage2_p1/edge_repair.py` (init/small/full), `edge_repair_report.py`; modified only Edge family solver `methods/stage2_modern_graph/edge_small.py`, tests `tests/test_stage2_edge_repair.py`. Same saved P1 observations/graph/operator: N32/64/256, seed20261019+N, original raw weighted degree, lambda1=lambda2=1, DoTnu=.1 and original active bands/order. No UE GT used.
**Frozen control / intervention:** A mu1/mu3/mu4=10 fixed; B one normalized residual-balancing rule (ratio10, factor2, every50 from100..3000 then frozen, bounds1e-3..1e4; mu2 originaldegree.mean fixed). Scaled multipliers rescaled old_mu/new_mu; max unscaled-preservation error0. Max15000 cold-start per configuration, no retries; totalcap3600s. Both policies share all old stop quantities plus missing Y-self/Y-row, complete prox and linearized dual checks, tolerance2e-5. No lambda/nu/grid or new method.
**Audit:** 32/64 nonzero-dual augmented Y/S/U/E/Q CVXPY references PASS (max solution error2.413588e-7); tightened reference tolerance only to1e-12 to resolve initial near-threshold vector accuracy, initial evidence retained. Matrix updates/prox/dual signs mathematically consistent with frozen Lagrangian; author's unweighted E threshold and S-based partial dual updates not imported. 3 Edge-only tests PASS. All 20 small A/B configs PASS. All5 fixed256 A U/S/E exactly reproduce previous failed attempt2 (difference0), preserving old P1 facts.
**256 B results:** SEF15000 NONCONVERGED (max constraint4.129758e-4, primal5.654260e-5, dual3.627826e-5, stationarity3.627825e-5, referencegap5.713346e-4); FE-author15000 FAIL (row.02025511, gap.001588036); FE-prior8180 PASS (primal3.961215e-7, dual1.672208e-6, stationarity9.058880e-7, row1.776009e-5, gap6.666556e-7, solver19.284899s); DoT-author/prior15000 FAIL (row.02113828/.03338486; gaps.001643445/.002046810). Overall NUMERICAL_BLOCKER, SEF/FE/DoT aggregate G1 FAIL. No NaN/OOM/PCG failure or abnormal collapse; all finite.
**Interpretation / limits:** No tested update-algebra error; logging/stopping coverage repaired. Split residual propagates into S constraints: fixed FE-author Y-row7.71e-5 but S-row.01953, amplification8.7623; L1-dominated total objective conceals feasibility error. One penalty rule repaired only one initialization. Cannot attribute all failure uniquely to penalties/order/nonunique solutions or call objective unsolvable; SEF has optimal convex reference, joint FE/DoT only convex block references. No performance ranking/valid negative quality conclusion.
**Artifacts / resource:** `results/stage2_modern_graph_20261019/p1_edge_solver_repair/REPORT.md`; comparison/gates/reference_objectives/objective_components CSV, full state/multipliers and numerical histories, code diff and convergence plots. 434 protected hashes unchanged. small/full wall130.891326/892.739993s including references/IO; CPU only, GPU0, RAM peak unknown. Updated Lambda2 export representation only after formal runs for old NPZ/JSON serializer compatibility; no new solver run. Branch codex/new-stage1 HEADf63652f; no commit/upload.
**Status:** confirmed numerical observations; partial solver repair with remaining NUMERICAL_BLOCKER, Edge outputs failing constraints invalid for method ranking. STOP pending user review, no further retry/P2A/P2B/P3/SparseDoT/full-N. Registry updated.


### EXP-P2A-B0-ROBUST-IC-20261020 — Frozen full-N 27-config Robust-IC grid

**Date / authorization:** 2026-10-20, P2A only after user reviewed P1-Edge. No P2B/P3/P4/P5, no Stage-1/G1 training or Edge tuning.
**Question / controls:** Do fixed IC8-P, IC9-P or group IC9-G improve B0 refinement over both advisor S2-J and same-objective S2-CG, and does the group form help at identical alpha/eta/movement?
**Code:** `methods/stage2_modern_graph/ic_cuda.py`, `run_ic_fulln.py`, `evaluate_ic_fulln.py`; original `ic_solver.py`, metrics and plotting helpers unchanged. Tests `tests/test_stage2_ic_cuda.py` (3) and `test_stage2_p2a_g3.py` (2) PASS.
**Inputs / protocol:** P0-frozen B0 N20997, G1281544167 undirected edges, delta0=.09517167367990964, J/CG embeddings; original node order/positions. W_IC=W/mean undirected weight(.5592081927929078), graph unchanged. Gauge from observed chart mean/RMS only; GT only post-run common evaluation/plots. Same fixed4199 sample (seed10), all20997 affine fit/eval, common_metrics CT/TW/KS and mean/median/P90/P95/max. Comparable group CG-A-transductive-stride1-current; no held-out claim, GT-based research screening not generalization validation.
**Frozen grid / numerical:** Three methods × alpha{.2,.5,.8} × eta_factor{.5,1,1.5}=27. CUDAfloat64 same-objective PDHG, min_radius/beta scalar step-ratio balance preserving product .9025; no GT selection. Exact guarded50-step CUDA replay, zero-aux shortcut only when certified and full-iteration rollback on violation. Initial30000 / one allowed120000 cold retry, prox2e-6 / feas1e-9; all27 first-attempt VALID, retries0. CPU/GPU objective maxdiff6.344454e-16; maxprox1.997941e-6, maxfeas3.510502e-17. Baseline re-evaluation exact difference0 for all8 metrics.
**Results:** J KS.0650645723534/Mean.318747967482/P95.634639978409; CG .0650645686743/.318747938339/.634639233193. IC8(.5,1delta0) KS.0646966961908/Mean.317918135155/P95.622588642895; relative improvements vsJ .565402%/.260341%/1.898925% and vsCG .565396%/.260332%/1.898810%. IC8alpha.8 mid is effectively identical (normalized output diff2.210902e-11). Those2 cells pass SURVIVOR only; ADVISOR_HIT=0,TARGET_HIT=0 across all27. IC9-P/G each9/9 valid but zeroSURVIVOR => NO_GO_EXPLORATORY. IC9-P mid KS.071414127169/Mean.339912766006/P95.655208138576; IC9-G mid .071107625194/.342312186594/.667049498412; group vsP KS improves .429190%, Mean/P95 worsen .705893%/1.807267% at equal actualdelta. No consistent group benefit.
**Mechanism / limits:** All18 IC9 outputs P=0, so beta is only positive objective scale at those solutions and alpha cells nearly duplicate. IC8 can activate P (max173563 edges), but bothSURVIVOR outputs P=0, not positive evidence of sparse-aux/weight-uncertainty mechanism. IC8 andJ macroscopic charts nearly overlap; IC9 more rounded/narrow branch at stronger movement but poorer metrics, no collapse. A valid negative result for frozen IC9 on this B0/G128, not a claim of universal method failure. Limited tail improvement of a near-quadratic point is not broad superiority, innovation or independent replication.
**Artifacts / resource:** `results/stage2_modern_graph_20261019/p2a_b0_ic_fulln/REPORT.md`, complete parameter_scan.csv, comparison all30 rows+9 matched groups, original-style raw/affine/native chart.png, embedding/normalized output, each attempt state and histories, G2/G3/family and pairedIC9 CSV. 49protected hashes unchanged; 84comparison rows match JSON. Full grid wall1344.573848s; solver loop1322.156822s; peak allocatorVRAM468262912bytes, sampledRSS1341370368bytes. These are refinement times, not training. Sanity100/1000 checks kept separate. No historical overwrite.
**Status:** confirmed execution/metrics; IC8 survivor preliminary and limited, IC9-P/G negative. P2A completed and STOP for review. Branch codex/new-stage1, HEADf63652f; not committed/uploaded. Registry updated.


### EXP-P2AR-IC8-SCALE-20261020 — Antenna-scale IC8 activation and matched-eta Ball-GLR

**Dates / authorization:** Full-N numerical runs completed 2026-10-20; interrupted chat resumed, unified evaluation and delivery completed 2026-10-25. User authorized P2A-R only, subsequently allowed eta and step budgets to adapt. No P2B/P3/G1/Stage-1 training/Edge retries or weight perturbations. Eta was not extended because existing points already covered activation; no GT-driven supplementary grid.
**Question / controls:** Separate IC8 sparse-aux mechanism gain over matched-eta Ball-GLR from final task gain over frozen S2-J and S2-CG. B0 N20997, cached G128 with1544167 undirected edges, same incidence/order/positions/gauge. W_IC=W/mean(W_e); topology and relative weights unchanged. All1122 protected old files and frozen calibration hashes unchanged.
**Code / sources:** methods/stage2_modern_graph/ic8_scale_cuda.py (shared enabled/disabled P), ic8_scale_experiment.py, ic8_scale_sanity.py, ic8_scale_evaluate.py, ic8_scale_report.py, ic8_scale_budget_diagnostic.py. Original P2A/P1/P0/baseline code untouched. Exact pre-budget and metric-run source snapshots retained under methods/stage2_modern_graph/archive/. Main output: results/stage2_modern_graph_20261019/p2a_ic8_scale_adaptation/REPORT.md.
**Fixed parameters / protocol:** Input-only pooled abs(sqrt(W_IC)DX) q95/q99/q995 gives kappa .1614905933767557/.22579795228060293/.2624168804298217 and alpha .2441313527262478/.31110304399606037/.3441908057988971. Eta .03/.08/.16/.32/.50, 5 Ball +15 scalar IC8 configurations. P-active edge-union fraction>=1e-4; IC-distinct also requires relative raw RMS>=1e-4 using centered B0 denominator; Ball-centered alternative saved. Independent scalar x/y balls/box, same objective and float64 PDHG as accepted P2A, no added fidelity. Same frozen4199 seed10 CT/TW/KS subset, CT/TW209 neighbors, all20997 separate affine fit/evaluation for mean/median/P90/P95/max; common_metrics and original plot_chart. Comparable group CG-A-transductive-stride1-current; no held-out claim. GT only evaluation and plotting.
**Sanity / budget diagnosis:** 24 N16/32/64 independent convex cases plus update regression and alpha roundtrip =26 PASS; max objective gap4.437895e-10; max prox1.513232e-6; 200-step old/new U/P/dual differences0. Initial schema and CLARABEL inaccurate logs retained; fixed SCS same-objective reference passed. Full-N Ball eta=.50 failed original30k cap with prox4.590913e-6, then passed32250. Primal3.003541e-16 vs dual4.590913e-6; U relative difference30k vsaccepted2.383315e-13. Measured decay.000372579/step matches theoretical.000372465, predicts32229 steps. This is a cap/slow-dual-transient issue, not divergence or evidence the convex target is intractable. User-authorized remaining3 IC eta=.50 limits60000, unchanged kernel/tau/sigma/tolerances, passed32700/32850/32950 without cold retry. In-progress q95 was deliberately stopped at logged15550 and archived as INTERRUPTED_FOR_STEP_BUDGET_UPDATE, not a valid result or numerical failure. Recorded time retained.
**G2:** 20/20 final configurations VALID; max prox1.998587837e-6, feasibility7.02100435e-17, minimum spread/input.546306595, max identical-point fraction1/20997. All indices/positions exact; CPU objective and raw/gauge checks pass; old baseline 8-metric evaluation differences0. No NaN/OOM/collapse; large eta causes observable structure degradation.
**Mechanism result:** 13/15 exact P nonzero; 10 P-active (q95 all eta; q99 .03/.08/.16; q995 .03/.08), 8 IC-distinct. Two eta=.50 q99/q995 are QUADRATIC_REGIME and match Ball raw-relative RMS<=1.46e-14. Only3 positive mechanism signals, all eta=.03. Relative to matched Ball, Mean improves1.173870%–1.329649%, Median1.565333%–1.899983%, P901.383613%–1.648915%, P951.223998%–1.514022%; Max worsens38.876209%–50.125364%. Q95 eta=.03 full metrics: CT .9971475224530463, TW .9973052993421252, KS .07522584431388547, mean .36516309137520464m, median .33329818255099786m, P90 .64372927184439m, P95 .77057602812172m, max6.765775514640418m. Matched Ball mean .37008390838242666, P95 .7824220703148643, max4.506750452824082. Weak conditional gain with extreme-error harm, not uniform improvement or verified edge-weight robustness.
**Task result:** All15 IC cells have0 ADVISOR_HIT/SURVIVOR/TARGET_HIT => NO_GO_EXPLORATORY on this grid. Descriptive minimum mean q995 eta=.08: CT .9982442426959292, TW .9982586846526882, KS .06484460157261719, mean .31919529168903293m, median .29349387357972945m, P90 .5603401661838695m, P95 .640425845884509m, max1.2854438345884078m. S2-J mean .31874796748161316/P95 .6346399784088135; CG .31874793833859877/.6346392331934765. Best IC KS improves.338081% vsJ but Mean/P95 worsen.140338%/.911677%, no endpoint hit. At eta=.32/.50 charts narrow/bend and stress/error deteriorate; no point collapse. Sparse-aux participation is established; value near the strong S2 operating region is not established. Coarse grid is not continuous-eta impossibility proof; no post-GT refinement.
**Resources / artifacts:** Recorded solver loops6222.973625s including failed and interrupted logged work; cumulative numerical run segments6274.218926s (104.570315min; excludes manual diagnostic pause), evaluation146.521585s; peak allocated VRAM468262400bytes, sampled process RAM1367719936bytes. Not training time. REPORT contains all9 research answers/all8 metrics; complete20-config scan/mechanism CSV, G2/G3, 5 eta comparison groups/raw+affine/same subset/protocol/aligned arrays, per-run chart/embedding/config/state/history/solver. 58 comparison rows exactly match JSON; 15 images manually reviewed; delivery_audit PASS. Baseline train-only resources unknown, not inferred. Run commands/config/input and source hashes in manifest and REPORT.
**Interpretation / stop:** Confirmed numerical/evaluation facts; task-performance negative, limited same-eta mechanism signal preliminary with Max harm. At most grounds for separately reviewed small mechanism validation, not a proven robust-weight method or a full-N expansion. Completed and STOP awaiting user review. Branch codex/new-stage1, HEADf63652fc3bec725f4fddd1105d13e288cbf233dc; no commit/upload. Registry updated. Old P2A/P1/P0/B0/S2 facts preserved.

### EXP-G1-LOSS-DIR-20261026 — Saved-chart direction of proposed geodesic plus diffusion loss

**Date / question:** 2026-10-26. On the already trained G1-Graph chart, does the proposed loss push coordinates toward correcting current chart error? This is an embedding-level offline diagnostic, not model retraining or a Stage-2 method result.

**Code / exact configuration:** `diagnostics/g1_loss_direction/run.py`; original full-N fused k20 support/Gaussian affinity plus self-loops, row-normalized `P`; `L_new=L_geo+L_diff`, lambda1, `L_diff=mean_i(||z_i-(Pz)_i||²/(sqrt(sum_j P_ij D_geo(i,j)²)+m))`. Original `m=51.49518966674805`. Fixed seed20261026 sampled399980 nonself geodesic pairs for `L_geo` gradient; `L_diff` exact on N20997. One direct coordinate step in each gradient direction, RMS raw displacement0.6542382717132569 (5% of median original-edge geodesic distance), normalized separately and chosen without GT. Graph/Identity saved full checkpoints/embeddings; no weights or optimizer updated. Original seed10 common4199-node metric sample and all-N separate affine evaluation. GT only post-step evaluation. Analytic gradients checked against toy central differences (max absolute differences geo1.04e-11, diff1.85e-12).

**Inputs / outputs:** `results/adp_old_fused/stride1_n20997_seed10/{fused,geodesic}.npy`; `results/g1_acf_graph_fcf/stride1_n20997_seed10_{graph,identity}_full/embedding.npz`; original B0 source unchanged. All new arrays, source JSONs, complete metrics, pair indices, per-node changes, original-format raw/affine Channel Charts and `REPORT.md` under `results/g1_acf_graph_fcf/loss_direction_diagnostic/`.

**Observed result:** Graph `L_geo/L_diff` at start1.995200581/0.196716118; gradient cosine0.075664. Proposed step decreases `L_new` 2.191916699→2.122706167, Graph Mean/P95 .399770707/.846337259→.398883700/.843248606 m, CT/TW .996947/.996849→.996982/.996895, KS .079340→.079146. Same-size original-loss-only step Mean/P95 .399704337/.845223427. Proposed step helps54.43% of nodes, harms45.57%; initial top-error quintile mean change-.001907 m, pointwise change P95 +.011682 m. Identity proposed step Mean/P95 .398343086/.843197107→.397427529/.840959072 m, so Graph remains worse and cannot claim propagation-specific gain. Full values and visual evidence in report/CSV; images visually near-identical at this scale.

**Interpretation / limits / status:** `preliminary` local-gradient evidence only: the new term provides a partially error-aligned direction at the saved embedding, with meaningful collateral harm and a similar effect for Identity. Direct coordinate movement has freedoms unavailable to a shared-parameter G1 network; no new loss model was trained or checkpoint selected, no full-N improvement claim. ADP-affinity full training remains paused. Branch `codex/new-stage1`; diagnostic code/results and record edits currently local and uncommitted. Registry updated as diagnostic, outside trained-method comparable group.

### EXP-G1-ERROR-SOURCE-20261027 — Stage-1 target, fit, and chart error attribution

**Date / question:** 2026-10-27. User authorized read-only diagnosis of whether G1 Stage-1 error is attributable to biased `D_geo` targets, incomplete model fit, or local fused-edge noise. No new model/loss training.

**Code / inputs:** `diagnostics/g1_error_source/run.py`; `results/adp_old_fused/stride1_n20997_seed10/{distance_data,fused,adp,geodesic}` and saved full-N `G1-Graph`/`G1-Identity` embeddings. Same 20997 indices/positions verified. Seed20261027 generated399983 nonself random pairs, plus all228773 unique original fused-k20 undirected edges. Edge source defined by `fused < adp - 1e-5` (time179147, ADP49626); distance quantile bins0/10/40/70/90/100%; GT spatial quartile grid4×4. Offline scalar `beta=sum(D_geo*dGT)/sum(D_geo²)=.01839743258516511 m/unit` fitted only for diagnosis, never used in training or checkpoint selection. Original `common_metrics.affine_transform` separately aligned saved charts on all nodes.

**Outputs / protocol:** `results/g1_acf_graph_fcf/error_source_diagnostic/REPORT.md`, `summary.json`, `pair_groups.csv`, `spatial_regions.csv`, `node_associations.json`, saved pair/edge index files; no new embeddings or charts. Same-node transductive diagnostic; groupwise model residual measured in original `D_geo` units, target/chart errors in meters. Code/result/CSV consistency checked. No GT enters any deployable method.

**Observed:** `D_geo` versus physical pair distance Spearman .982724913. After one global scalar fit, nearest10% D_geo pairs have mean target bias +.45360m and farthest10% -.39035m. Graph's native predicted-minus-target means -19.01776/+21.59658 original units in these bins; Identity similar -19.11464/+20.89111. Graph/Identity random-pair weighted geodesic stress1.9910/1.9846; affine physical pair absolute error mean .28794/.28727m. Original edge target abs error mean time-source .0415m vs ADP-source .1768m, but node error versus incident edge target-error Spearman Graph -.0257/Identity -.0294. For random-pair target abs error it is .1705/.1735; node error versus incident model-to-D_geo abs residual .0597/.0547. Spatial 16-cell table shows high node error does not consistently accompany greater target or model residual.

**Interpretation / status:** `confirmed` descriptive distance-scale mismatch and paired model observations; `preliminary` causal attribution. A single global scaling of `D_geo` does not fit short and long distances equally; Graph/Identity outputs depart from this target in directions that partly counter the scale mismatch. Neither local edge bias nor incomplete target fit is established as the principal cause of high node error. This does not justify the proposed smoothing loss, indiscriminate ADP-edge downweighting, or closer fitting of all `D_geo` targets. Further GT-free predictor testing would need distance/region matching before any loss design. Branch `codex/new-stage1`, local uncommitted/unuploaded; registry indexed as diagnostic outside method ranking.
