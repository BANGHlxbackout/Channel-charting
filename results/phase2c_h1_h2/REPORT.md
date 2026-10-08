# Phase 2C H1 First-Stage Diagnostic

## Scope

This run covers edge anomaly, calibrated geodesic error, sampled path impact,
concentration, and wireless-branch attribution only. It does not perform edge
removal, H2, or method development. GT positions are used for offline diagnosis only.

## Frozen configuration

- Seed: `10`
- Stage-1 graph: verified directed CSR, shortest paths with `directed=False`
- Undirected edges: `228773`
- Physical local scale: q=`16` median physical-neighbor distance
- Calibration pairs: `100000`
- Discovery: `128` sources x `512` targets
- Validation: `128` sources x `512` targets
- Severe threshold: calibration residual 5th percentile = `-0.183408`
- Random controls: `50` uniform + `50` matched

## Main observations

- Discovery severe pairs: `3309` / `65536`
  (`5.0491%`).
- Validation severe pairs: `3516` / `65536`
  (`5.3650%`).
- Top-1% discovery-ranked edges explain `75.2275%` of severe
  validation paths.
- Matched-random top-1% mean / P95: `5.0375%` / `6.6169%`.
- Lift over matched-random mean: `14.933x`.
- Top-1% median robust mismatch score: `1.029`.
- Removing boundary-tie-ambiguous candidates from the top-1% set retains
  `99.1682%` of its severe validation-path coverage.

## Decision

`proceed_to_H1_counterfactual = false`

Checks: `{"tie_excluded_coverage_retains_at_least_80pct": true, "top1pct_median_robust_mismatch_at_least_2": false, "validation_top1pct_coverage_at_least_0p20": true, "validation_top1pct_exceeds_matched_p95": true, "validation_top1pct_lift_at_least_2x_matched_mean": true}`

This is not an `H1_supported` decision. Full H1 support would still require the
pre-specified targeted-removal counterfactual and sensitivity checks.

## Subsequent protocol review

The generated `false` value above is retained as the actual output of the
first-stage implementation. It depended on an additional requirement that the
median robust mismatch of the entire top 1% (2,288 edges) be at least 2. That
requirement is not present in the active Phase 2C task specification and is not
aligned with its actual removal batches `X={25,50,100}`. Their median robust
mismatch values are `3.184`, `2.443`, and `2.051`, respectively.

The research interpretation of this first-stage result is therefore
`preliminary strong signal`, with `H1_supported = null`. The task-specified
targeted-removal counterfactual is required before an H1 decision.
