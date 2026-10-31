# Phase 2C — H1/H2 Empirical Failure Diagnosis

> **历史任务，非当前待执行项。** 2026-09-13 整理注：当前任务以 `docs/research/research_state.md` v2.2 为准。保留下文以追溯 Phase 2C 原始设计；其中 H1/H2 顺序和 sensitivity 要求不自动生效。已完成结果见 `results/phase2c_h1_h2/H1_COUNTERFACTUAL_REPORT.md`；冻结判据未通过，不据此自动补跑 sensitivity 或 H2。

## Status

This is a diagnostic / falsification task.

Do not design a new refinement model.
Do not modify the existing baseline.
Do not require metric improvement over the Stage-2 baseline.

The execution order is:

```text
H1 diagnostic
→ if H1 supported
H2 diagnostic
→ research-direction decision later
```

## Stable Baseline Facts

### Stage 1

```text
CSI
→ ADP/timestamp fused dissimilarity
→ 20-NN graph
→ shortest-path D_geo
→ Siamese FCF
→ Y0
```

### Stage 2

```text
Y0 + D_geo
→ k=128 geodesic-neighbor refinement graph
→ locally scaled Laplacian/Tikhonov
→ Z
```

Do not mix the Stage-1 20-NN graph with the Stage-2 k=128 graph.

## Step 0 — Read-only Code / Artifact Audit

Before implementing anything, locate and report:

1. Stage-1 fused distance implementation and stored artifacts.
2. Stage-1 20-NN sparse graph, symmetrization rule, and edge weights.
3. Existing `D_geo` artifact and storage format.
4. GT UE positions.
5. timestamps.
6. `Y0`.
7. refined `Z`.
8. whether per-edge `d_ADP` and `gamma * |Δt|` can be recovered so each Stage-1 edge can be classified as:
   - ADP-dominated, or
   - timestamp-dominated.

Report exact file paths, variable names, shapes, dtypes, and approximate memory sizes.

Do not edit code until this audit is complete.

# H1 — Harmful local edge / geodesic shortcut diagnosis

## H1 Research Question

Does the actual Stage-1 wireless-derived 20-NN graph contain a small subset of local edges that are physically inconsistent and disproportionately responsible for severe global geodesic underestimation?

## H1.1 Edge-level screening

Use the actual Stage-1 graph.

For each undirected Stage-1 edge `(i,j)`, compute:

- fused Stage-1 edge distance;
- physical GT distance, diagnostic only;
- endpoint degree;
- wireless local-scale normalized edge length;
- physical local-scale normalized edge length;
- robust mismatch score;
- ADP/timestamp branch attribution when recoverable.

Do not use Stage-2 `sigma_i` for H1.

### Physical local scale

Use a GT KD-tree only for offline diagnosis.

Default:

```text
q = 16
```

Sensitivity:

```text
q ∈ {8, 16, 32}
```

Use a robust local statistic such as median physical distance to the q nearest physical neighbors.

## H1.2 Calibrated D_geo error

Do not assume:

```text
d_phys = alpha * D_geo
```

Use an independent calibration pair subset to estimate a monotonic relationship:

```text
log(D_geo) ≈ f(log(d_phys))
```

Preferred:
- isotonic regression.

Fallback:
- physical-distance bins + robust median curve.

Define the residual:

```text
r_geo = log(D_geo) - f(log(d_phys))
```

Large negative residual means `D_geo` is unusually short for that physical separation and is a shortcut-style candidate.

Calibration data must not be reused for final validation.

## H1.3 Discovery / Validation Split

Use fixed random seeds and separate:

1. calibration pairs;
2. discovery sources / pairs;
3. validation sources / pairs.

Discovery data may be used to rank suspicious edges.

Validation data must never affect the edge ranking.

## H1.4 Sampled shortest-path impact

Do not compute exact all-pairs edge betweenness.

Initial configuration:

```text
discovery_sources ≈ 128
targets_per_source ≈ 512
```

Stratify sources over spatial coverage and sampling-density quartiles when possible.

Using sparse weighted shortest paths, estimate for each edge:

- sampled path usage;
- severe-underestimation path usage;
- bad-path enrichment.

A harmful-edge candidate must combine:

```text
large physical/wireless mismatch
+
high impact on bad geodesic paths
```

## H1.5 Concentration Test

Rank candidate edges using discovery data only.

Plot:

```text
x = top edge fraction / count
y = fraction of severe validation-style bad paths explained
```

Include matched-random baselines.

The hypothesis is stronger if a very small fraction of edges explains a disproportionate fraction of severe geodesic underestimation.

## H1.6 Counterfactual Removal

Only after H1 screening shows a signal.

Remove batches such as:

```text
X ∈ {25, 50, 100}
```

Compare:

1. targeted suspicious/high-impact removal;
2. matched random removal;
3. suspicious-but-low-impact removal.

Matched random controls should match important confounders where possible, such as:

- edge-weight percentile;
- endpoint degree;
- graph region.

Evaluate only on independent validation sources / pairs.

Report:

- calibrated geodesic residual distribution;
- severe underestimation tail;
- overestimation tail;
- rank agreement with physical distance;
- giant-component ratio;
- number of connected components;
- unreachable validation pairs;
- path inflation on previously well-estimated pairs.

Do not count improvement caused by graph disconnection as evidence.

## H1.7 Wireless-specific Attribution

If `d_ADP` and `gamma * |Δt|` are available, compare suspicious/high-impact edges with ordinary edges in terms of:

- ADP-dominated proportion;
- timestamp-dominated proportion;
- mismatch distribution;
- bad-path enrichment;
- spatial distribution.

This is important because the research value depends on whether the failure can be tied to Channel-Chart-specific wireless evidence rather than only generic kNN noise.

## H1 Go / No-Go

Mark `H1_supported = true` only if all of the following hold:

1. clear edge anomaly;
2. significant bad-geodesic concentration;
3. targeted removal improves held-out validation more than matched controls;
4. graph connectivity is essentially preserved;
5. the result is stable under at least two reasonable q / source-sampling settings.

Otherwise report H1 as weak or unsupported and stop before H2.

# H2 — Same-source correction limitation

Execute only if `H1_supported = true`.

## H2 Research Question

When `D_geo` is structurally wrong, does `Y0` inherit the same distortion, and does Stage 2 using the same `D_geo` correct, preserve, or amplify it?

Do not call this confirmation bias by default.

## H2.1 Pairwise errors

For sampled validation pairs, compute pairwise distances for:

- `D_geo`;
- `Y0`;
- `Z`;
- physical GT distance, diagnostic only.

Use scale-normalized / calibrated pairwise errors with consistent definitions.

For `Y0` and `Z`, also add a robustness analysis using affine transforms fitted only on calibration nodes and evaluated on disjoint validation nodes. Never fit the transform on the same samples used for evaluation.

## H2.2 H1-affected cohort

Define the main H2 cohort from validation pairs that:

1. have severe negative `D_geo` residual, and
2. traverse H1 high-impact suspicious edges.

Construct matched controls using physical-distance bins and other available confounders such as timestamp difference and local density.

## H2.3 Inheritance

Test:

```text
D_geo error → Y0 error
```

Report:

- sign agreement;
- Spearman correlation;
- robust regression;
- error-severity distribution;
- comparison with matched controls.

If `Y0` does not inherit the H1-identified error pattern, H2 is unsupported.

## H2.4 Refinement Response

For the same pairs, classify:

```text
Y0 → Z
```

as:

- corrected;
- preserved;
- amplified.

The key evidence is conditional response:

```text
ordinary pairs improve
but
H1 structural-error pairs improve much less / remain wrong / worsen
```

Do not rely only on global CT/TW/KS or a single global correlation.

Use bootstrap confidence intervals for cohort-level effects.

Only use “reinforcement” or “amplification” if the error magnitude increases and the update moves farther from physical truth.

# Outputs

Create a new isolated directory, for example:

```text
diagnostics/phase2c_h1_h2/
results/phase2c_h1_h2/
```

Recommended outputs:

```text
results/phase2c_h1_h2/
  audit.json
  config.json
  h1_edge_table.parquet
  h1_pair_table.parquet
  h1_summary.json
  h2_summary.json        # only if H1 supported
  REPORT.md
  figures/
```

Recommended figures:

1. wireless-vs-physical edge mismatch;
2. suspicious-edge spatial overlay on GT positions;
3. top-edge concentration curve with controls;
4. targeted-vs-control removal;
5. calibrated `D_geo` error vs `Y0` error;
6. H1-error severity vs `Y0 → Z` corrective response.

# Computational Constraints

With `N ≈ 20,997`:

- do not construct a new full `N x N` physical-distance matrix;
- do not construct full error matrices;
- do not construct an all-pairs predecessor matrix;
- do not recompute APSP after every individual edge removal;
- use sparse graphs, sampled pairs, sampled Dijkstra sources, and batch counterfactuals.

# Experiment Log

After execution, append a structured entry to:

```text
docs/research/experiment_log.md
```

Record:

- date;
- question;
- code entry point;
- exact config;
- seeds;
- inputs;
- outputs;
- results;
- H1/H2 interpretation;
- caveats;
- status.

Negative results must be retained.
