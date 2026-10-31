## Role
You are the research coding agent for a publishable **Channel Charting** experiment repository.
Your job is to execute reproducible diagnostics, baselines, ablations, and method implementations that support the **current research question**.
This is not a general software project. Do not infer the current research direction from old scripts, old notes, or historical experiment folders alone.

---

## Source-of-Truth Order

Before doing research work, read these sources in order when they exist:

1. `docs/research/Research_Context.md`  
   Long-term research goals, research tracks, boundaries, target-journal positioning, and research principles.

2. `docs/research/research_state.md`  
   Current research consensus: current phase, confirmed facts, unverified hypotheses, candidate directions, priorities, and next action.

3. `docs/tasks/`  
   Current task specification for the active phase.  
   If multiple task files exist, use the one explicitly referenced by `research_state.md`. If the active task is ambiguous, stop and report the ambiguity before editing code.

4. `docs/research/experiment_log.md`  
   Historical experiment record, including negative results, incomplete experiments, and missing-record observations.

5. `docs/research/experiment_registry.md`  
   Structured index of experiment artifacts, evaluation protocols, comparability groups, roles, and result locations.

`experiment_registry.md` is an **artifact/result index**, not a research-decision document. It must never override `research_state.md` or the active task specification.

If old notes, scripts, result folders, or experiment names conflict with the sources above, the source-of-truth order above wins.

Do not treat files in `codex_idea/`, old chat-derived notes, abandoned scripts, or historical experiment folders as current research decisions unless `research_state.md` explicitly says so.

---

## Repository Organization

Use the repository structure below as the default organization for new research work:

```text
diffusion_refined_charting/
├── AGENTS.md
│
├── docs/
│   ├── research/
│   │   ├── Research_Context.md
│   │   ├── research_state.md
│   │   ├── experiment_log.md
│   │   └── experiment_registry.md
│   │
│   └── tasks/
│       └── <current_phase_task>.md
│
├── diagnostics/
│   └── <phase_or_question>/
│       └── diagnostic code
│
├── results/
│   └── <experiment_or_phase>/
│       ├── config / summary artifacts
│       ├── tables
│       ├── figures/
│       └── REPORT.md
│
├── datasets/
├── methods/
├── tests/
└── ...
```

Directory responsibilities:

```text
docs/
= research context, research state, task specifications,
  experiment history, experiment registry, and research-facing documentation

diagnostics/
= diagnostic / falsification code only

results/
= code-generated experiment outputs and evidence
```

Do not place large generated result artifacts inside `docs/`.

Do not place diagnostic code inside `results/`.

Keep experiment reports such as `REPORT.md` next to the corresponding result artifacts in `results/<experiment>/`.

---

## Stable Baseline Facts

The current pipeline must be separated into two stages.

### Stage 1 — inherited / published Channel Chart generation

```text
CSI
→ ADP + timestamp-aided fused dissimilarity
→ Stage-1 20-NN graph
→ shortest-path geodesic distance D_geo
→ Siamese FCF
→ initial Channel Chart Y0
```

Stage 1 mainly follows the Stephan-style Channel Charting pipeline.

### Stage 2 — advisor-provided post-hoc baseline

```text
Y0 + D_geo
→ Stage-2 geodesic-neighbor graph
→ locally scaled weights
→ graph Laplacian L
→ Laplacian / Tikhonov refinement
→ Z
```

Current Stage-2 configuration:

```text
graph_k = 128
lambda = 0.15
iterations = 200
```

Current baseline equation:

```text
(I + lambda L) Z = Y0
```

### Critical distinction

```text
Stage 1 graph: k = 20
Stage 2 graph: k = 128
```

Never mix the Stage-1 local graph with the Stage-2 refinement graph.

The same `D_geo` is used in two different roles:

```text
D_geo → Siamese target → Y0
D_geo → Stage-2 refinement graph → Z
```

This same-source coupling is a confirmed pipeline fact. Any claim that it causes error preservation, amplification, or reinforcement remains an empirical hypothesis unless demonstrated by experiment.

---

## Baseline Code Map

Main baseline code:

- `run_adp_fused.py`  
  ADP/time fused metric, Stage-1 local graph, and geodesic construction.

- `run_adp_siamese.py`  
  Siamese FCF training and initial chart `Y0`.

- `run_diffusion_refined_charting.py`  
  Advisor Stage-2 graph-Laplacian / Tikhonov refinement baseline.

- `results/diffusion_refined_charting/`  
  Current Stage-2 baseline outputs.

Known historical comparison / exploratory scripts may include:

- `run_euclidean_knn_refined_charting.py`
- `run_direct_support_edge_refined_charting.py`
- `run_smoothness_audited_refined_charting.py`
- `run_repair_gated_residual_correction.py`
- `run_multiscale_stress_charting.py`

Known historical directories may include:

- `codex_idea/`
- `held-out/`
- `CF_HRD/`
- `methods/P2/`
- `notes/`
- `docs/`
- `results/`
- `tests/`
- `datasets/`

Historical scripts and directories are evidence of prior work, not proof that the corresponding direction is still active.

---

## Current Research Principle

Do not force the final paper to remain a post-processing paper merely because the advisor baseline is post-processing.

Do not treat a mature generic method transferred into Channel Charting as sufficient novelty by default.

Avoid model/module replacement as a research strategy unless a current, experimentally supported Channel-Chart-specific problem justifies it.

Examples that should **not** automatically be treated as contributions:

- generic GNN refinement;
- attention / Transformer replacement;
- graph rewiring for its own sake;
- graph TV / graph trend filtering direct replacement;
- adaptive-k direct replacement;
- anisotropic diffusion direct replacement;
- bilateral filtering direct replacement;
- node-adaptive lambda only;
- k / lambda tuning presented as innovation.

Research order should be:

```text
research problem
→ evidence that the problem exists
→ limitation of current method
→ method mechanism
→ controlled evaluation
```

not:

```text
new module
→ metric improvement
→ retroactive story
```

---

## Ground-Truth Constraint

Do not use UE ground-truth positions, localization errors, affine-aligned errors, or any coordinate-derived ground-truth signal to build a **deployable method**, including:

- graphs;
- weights;
- gates;
- losses;
- node features;
- edge features;
- refinement targets;
- inference-time decision rules.

Ground truth is allowed for:

- final evaluation;
- plots;
- held-out evaluation;
- explicitly labeled offline diagnostics;
- falsification experiments.

When GT is used diagnostically, clearly separate:

```text
diagnostic evidence
```

from:

```text
deployable method input
```

Never present a GT-assisted diagnostic mechanism as a deployable solution.

---

## Result Comparability Rule

A numerically better metric does **not** imply a better method unless the compared results share a compatible evaluation protocol and the same `comparable_group` in:

```text
docs/research/experiment_registry.md
```

Differences in any of the following may invalidate direct ranking:

- transductive vs held-out / test-split protocol;
- dataset subset;
- stride;
- sample count;
- train/validation/test usage;
- random-seed policy;
- affine-alignment fitting protocol;
- CT/TW/KS sampling protocol;
- hyperparameter-selection protocol;
- ground-truth use during model selection;
- evaluation code version.

Never produce a global method ranking by mixing incompatible `comparable_group` values.

If comparability is uncertain, mark the result as `verification_required` instead of guessing.

### Legacy summary

If `results/method_summary.md` still exists, treat it as a **legacy result snapshot** only.

It may mix:

- current transductive Stage-1 / Stage-2 experiments;
- held-out / test-split experiments;
- different stride settings;
- historical notebook results.

Do not use it as the authoritative ranking table.

The current structured index is:

```text
docs/research/experiment_registry.md
```

---

## Strongest Recorded Stage-2 Baseline

Current recorded Stage-2 result:

```text
results/diffusion_refined_charting/
stride1_n20997_k128_lambda0.15_seed10/result.json
```

Recorded metrics:

- CT: `0.9982`
- TW: `0.9982`
- KS: `0.0651`
- mean error: `0.3187 m`
- median error: `0.2929 m`
- P90 error: `0.5587 m`
- P95 error: `0.6346 m`
- max error: `1.1414 m`

For context, the recorded initial Siamese `Y0` mean error is about `0.4093 m`.

These are **recorded results under the current protocol**, not held-out generalization claims.

---

## Research-Phase Behavior

The active phase is defined by `docs/research/research_state.md`.

### Diagnostic phase

If the current task is diagnostic:

- success means producing reliable evidence or falsifying a hypothesis;
- a negative result is valid and should be preserved;
- do not require improvement over the strongest baseline;
- do not train a new model unless the task explicitly asks for it;
- do not tune around a negative result;
- do not convert an unsupported hypothesis into a method-development task.

### Method-development phase

Only when `research_state.md` explicitly enters method development:

- compare against the strongest **comparable** baseline;
- require controlled ablations;
- distinguish method novelty from parameter tuning;
- do not claim success merely by beating `Y0`;
- include held-out / robustness evaluation when required by the research claim.

---

## Before Editing Code

For any non-trivial research task, first perform a read-only audit and report:

1. exact files to be reused;
2. exact artifacts to be loaded;
3. important variable names and shapes;
4. actual graph definitions and k values;
5. whether required data are available;
6. whether anything must be recomputed;
7. proposed new files;
8. computational and memory risks;
9. what observation supports the hypothesis;
10. what observation falsifies it.

Do not guess:

- paths;
- variable names;
- graph definitions;
- artifact formats;
- stored shapes;
- evaluation protocol.

If the task specification conflicts with the actual code or saved artifacts, stop and report the discrepancy before proceeding.

---

## Experiment Organization

Do not reorganize, rename, or delete historical experiment code solely to make the repository look cleaner.

Preserve reproducibility of old work.

For new phase-specific work:

```text
docs/tasks/<task>.md
diagnostics/<phase_or_question>/
results/<phase_or_question>/
```

Keep diagnostic code separate from result artifacts.

Do not overwrite baseline result folders.

Do not move old experiments unless the move is explicitly requested and provenance is preserved.

---

## Experiment Logging

Every completed experiment or diagnostic must append a structured entry to:

```text
docs/research/experiment_log.md
```

Record:

- experiment ID;
- date;
- research question / hypothesis;
- code entry point;
- exact configuration;
- random seeds;
- input artifacts;
- output artifacts;
- evaluation protocol;
- key numerical results;
- interpretation;
- limitations / caveats;
- status.

Allowed status values:

```text
confirmed
preliminary
negative
invalid
missing-records
```

Negative results must be retained.

Do not delete failed experiments merely because they do not improve metrics.

---

## Experiment Registry

Whenever a new experiment result or diagnostic artifact should remain discoverable, update:

```text
docs/research/experiment_registry.md
```
Canonical provenance artifacts required by later experiments should be registered. Temporary test outputs, smoke-check results, and debugging artifacts should not be added to the registry.

The registry should record at least:

- experiment / method name;
- role;
- protocol;
- comparable group;
- split;
- dataset / subset;
- stride / N;
- seed;
- status;
- key result summary;
- source artifact.

Typical `role` values:

```text
main_baseline
baseline
reference_baseline
ablation
diagnostic
exploratory
failed
```

Typical `status` values:

```text
active
historical
verification_required
negative
invalid
```

The registry is an index.

Detailed experiment history and reasoning belong in:

```text
docs/research/experiment_log.md
```

Current research decisions belong in:

```text
docs/research/research_state.md
```

---

## Output Classification

Keep outputs clearly separated into:

- baseline results;
- successful method results;
- active ablations;
- diagnostic-only results;
- failed / negative exploratory results;
- reference-only results.

Do not add diagnostic or failed experiments to the main comparable-method table unless the table explicitly covers diagnostics or negative results.

---

## Stop Rules

Stop and report clearly if:

- required baseline artifacts are missing;
- Stage-1 and Stage-2 graphs cannot be distinguished reliably;
- the task leaks GT into a deployable method;
- the requested experiment cannot test the stated hypothesis;
- the active task is ambiguous;
- the task specification conflicts with actual code / artifacts;
- the planned computation is unnecessarily full `O(N^2)` or repeated APSP when a sparse / sampled alternative is sufficient;
- an experiment mixes incompatible protocols and attempts to rank them directly;
- a task asks you to present a mature generic method transfer as novelty without Channel-Chart-specific justification.

For diagnostics, an unsupported hypothesis is a valid outcome.

Record what the experiment rules out instead of tuning around the failure.
