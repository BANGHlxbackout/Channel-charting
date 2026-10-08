# Phase 2C H1 Targeted-Removal Counterfactual

## Scope

This diagnostic uses the frozen first-stage discovery ranking and an entirely
new pair-disjoint test sample. It does not rerank edges, refit calibration, run
sensitivity, execute H2, or design a refinement method.

## Fresh test

- Seed: `11`
- Sources / targets per source: `128` / `512`
- Pairs: `65536`
- Baseline-severe pairs: `3352` (`5.1147%`)
- Baseline-overestimated pairs: `3428` (`5.2307%`)
- Baseline-well-estimated pairs: `32038`
- Baseline shortest paths exactly reproduce saved `D_geo` in float32: `true`
- Controls: `50` matched-random and `50` suspicious-low-impact nested sequences

## Targeted results

| X | repair rate | mean delta r_geo | median delta r_geo | severe fraction after | new overestimation | median well-estimated inflation | P95 well-estimated inflation | components | unreachable pairs |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 25 | 17.3031% | 0.033005 | 0.000000 | 4.2297% | 0.6271% | 0.000000 | 0.062203 | 1 | 0 |
| 50 | 18.4368% | 0.034782 | 0.000000 | 4.1718% | 0.6760% | 0.000000 | 0.063413 | 1 | 0 |
| 100 | 23.5979% | 0.043936 | 0.000000 | 3.9078% | 0.9872% | 0.003313 | 0.070019 | 1 | 0 |

## Control-relative results

Each cell is `targeted / matched-random P95 / suspicious-low-impact P95`.

| X | repair rate | mean delta r_geo | new overestimation | P95 well-estimated inflation |
|---:|---:|---:|---:|---:|
| 25 | 17.3031% / 2.5447% / 0.8950% | 0.033005 / 0.003089 / 0.001249 | 0.6271% / 0.2253% / 0.3467% | 0.062203 / 0.016549 / 0.005112 |
| 50 | 18.4368% / 3.0087% / 0.9845% | 0.034782 / 0.003515 / 0.001351 | 0.6760% / 0.2489% / 0.4552% | 0.063413 / 0.021119 / 0.015360 |
| 100 | 23.5979% / 6.3097% / 1.2232% | 0.043936 / 0.008115 / 0.001745 | 0.9872% / 0.6199% / 0.5379% | 0.070019 / 0.043806 / 0.017995 |

All 50 replicates at every X satisfied the pre-frozen critical balance limit
`|SMD| <= 0.10`. All secondary covariates satisfied `|SMD| <= 0.20`, although
several values were close to the admissible boundary. The suspicious-low-impact
controls had positive, matched total path usage and matched mismatch. Their
realized median discovery bad-path enrichment was zero; this was an outcome-free
result of the secondary low-enrichment objective, not an eligibility rule.

## Decision

`H1_counterfactual_supported = false`

The frozen rule was not satisfied. Targeted removal exceeded both control P95
values for repair rate at every X, and its mean severe-cohort residual correction
also exceeded both control P95 values. However, the pre-frozen median residual
endpoint was zero for targeted and controls because most severe pairs were
unchanged by these sparse interventions. Targeted new overestimation and P95
well-estimated-pair inflation also exceeded both control families at every X.
The result was not caused by graph disconnection: all 303 modified graphs had one
component and zero unreachable fresh-test pairs.

The ranked edges therefore have strong causal influence on a concentrated subset
of severe underestimation, but the deletion intervention is not sufficiently
selective: it also causes substantial collateral geodesic inflation. The formal
`false` is specific to the frozen clean, low-damage counterfactual support rule;
it must not be read as evidence that the ranked edges have no effect.

The median endpoint was insensitive to a sparse effect, and the absolute harm
guardrail compares targeted removal with control interventions having much smaller
repair magnitude. Both are retained as pre-outcome design caveats rather than
changed after observing the result.

`H1_supported` remains `null`. No q/source sensitivity, H2, or method-development
experiment was run.
