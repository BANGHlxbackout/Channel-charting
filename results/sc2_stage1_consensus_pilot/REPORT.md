# SC2-inspired Stage-1 consensus pilot

**Interpretation correction after design review (2026-09-25):** This pilot tested our added rule, "fewer common neighbors -> longer original Stage-1 edge." SC2-PCR defines compatibility between candidate correspondences and uses their common compatible correspondences to choose consensus sets and estimate a rigid model; it does not prescribe this edge-length penalty. The pilot therefore cannot test or reject the broader third-party-support idea. Its measured negative result remains valid for the specified penalty. No frozen endpoints or output values were changed.

An additional read-only, post-result check examined stored Stage-1 edges within each of ten original fused-edge-length bins. Comparing the bottom versus top quartile of common-neighbor scores within each bin, the low-support group had a *smaller* median physical endpoint separation in 8/10 bins. This is GT-assisted exploratory evidence that "low support means a physically too-short edge" was unjustified here; it is not a preregistered endpoint or a general edge-validity test.

The frozen independent Stage-1 distance gates were evaluated before any FCF training. Because the primary MDS gate result is `False`, this run stopped before all Siamese and Stage-2 training pipelines.

## Primary fixed-subset MDS gate

The existing repository MDS implementation was run on exactly `np.arange(0,20997,10)` (2,100 samples) from each full-graph D_geo matrix, with seed 10, max_iter 80, and n_init 4. The same physical positions were used only for affine evaluation.

The uniform graph matches the consensus total edge weight to float32 summation precision. The shuffled graph preserves the multiplier distribution within bins but its realized edge-weight sum is 6665088.973521, versus consensus 6669263.784317; this is a 0.0626% lower total and is recorded as a control limitation.

| variant | MDS mean (m) | MDS p95 (m) | CT | TW | KS |
|---|---:|---:|---:|---:|---:|
| baseline | 0.458130 | 0.961453 | 0.991671 | 0.993360 | 0.090692 |
| uniform | 0.458130 | 0.961453 | 0.991671 | 0.993360 | 0.090692 |
| shuffled | 0.559366 | 1.156367 | 0.988663 | 0.991845 | 0.105277 |
| consensus | 0.582461 | 1.173664 | 0.987895 | 0.991686 | 0.110143 |

Frozen MDS conditions: consensus mean at least 0.005 m lower than every control; p95 no more than 0.01 m above baseline; CT/TW no more than 0.005 below baseline; KS no more than 0.01 above baseline.

- Consensus mean condition: `False`
- Consensus p95 condition: `False`
- Consensus CT condition: `True`
- Consensus TW condition: `True`
- Consensus KS condition: `False`
- MDS gate positive: `False`

## Secondary held-out physical-distance diagnostic

Each variant was independently calibrated with monotone isotonic regression on 100,000 disjoint unordered pairs. The uniform test has 50,000 uniformly sampled unordered pairs; the local test has 50,000 source/target pairs drawn from the saved original `geodesic_top512.npy` first 128 columns. Pair generation used seed 20260925 and is retained in `gate_pairs.npz`. Ground truth positions were used only for calibration and held-out scoring.

| variant | uniform MAE (m) | uniform p95 abs (m) | local MAE (m) | local p95 abs (m) | uniform Spearman | local Spearman |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 0.399153 | 1.055826 | 0.109400 | 0.301187 | 0.983722 | 0.619399 |
| uniform | 0.399153 | 1.055826 | 0.109400 | 0.301187 | 0.983722 | 0.619400 |
| shuffled | 0.449695 | 1.158382 | 0.118659 | 0.318485 | 0.980326 | 0.558412 |
| consensus | 0.468074 | 1.199447 | 0.123378 | 0.328810 | 0.978645 | 0.524923 |

## Frozen decision

- Consensus uniform-test MAE at least 1% lower than baseline, uniform, and shuffled: `False`
- Consensus local-test MAE within 1% of baseline: `False`
- Consensus uniform-test p95 within 1% of baseline: `False`
- Consensus local-test p95 within 1% of baseline: `False`

The isotonic pair test is a secondary diagnostic proxy for physical-distance fidelity after per-variant monotone recalibration; it does not decide whether to train. Since the primary MDS gate failed, the budgeted FCF and Stage-2 runs were not started. No coefficient, endpoint, or threshold was tuned after observing outcomes.

Artifacts: `config.json`, `variant_summaries.json`, `mds_gate.json`, per-variant `mds.npz`/`mds_result.json`, `gate_pairs.npz`, `geodesic_gate.json`, per-variant `geodesic.npy`, `graph_summary.json`, and `gate_isotonic.npz`.

Executed commands: `D:\anaconda3\Scripts\conda.exe run -n diffusion_charting python methods/sc2_stage1/run_pilot.py --build-only --device cuda:0`; then `D:\anaconda3\Scripts\conda.exe run -n diffusion_charting python methods/sc2_stage1/run_pilot.py --skip-build --device cuda:0`. Environment: Python 3.11.15, torch 2.11.0+cu128, NVIDIA GeForce RTX 3050 Laptop GPU. Focused semantic tests: `D:\anaconda3\Scripts\conda.exe run -n diffusion_charting python -m unittest tests.test_sc2_stage1 -v` (3/3 passed).
