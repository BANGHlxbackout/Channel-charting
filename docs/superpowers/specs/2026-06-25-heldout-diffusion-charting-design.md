# Held-Out Diffusion Charting Design

## 1. Goal

Add a seed-controlled 80/20 held-out experiment for the existing ADP/time
Siamese and graph-harmonic diffusion pipeline without changing the three
already working reproduction scripts.

All three scripts must support arbitrary integer seeds through `--seed`.
This acceptance run executes only seed 10; running multiple seeds is outside
the current execution scope.

For seed 10:

- training samples: 16,798
- test samples: 4,199
- the test set is excluded from graph construction and model training
- the trained FCF produces a held-out test embedding `Y_test`
- diffusion refinement produces only a training embedding `Z_train`
- producing a refined test embedding `Z_test` is deferred to the future
  out-of-sample extension experiment

This experiment must not use P2 model logic or treat P2 as an alignment target.
The only reused P2-compatible element is the deterministic split function
already implemented in `common_dichasus.py`.

## 2. Non-Goals

This change does not:

- modify the existing full-data reproduction scripts
- run or evaluate MDS
- construct a graph containing test samples
- refine test samples
- implement Nyström, distillation, interpolation, or another out-of-sample
  extension
- tune `graph_k` or `lambda_smooth`; the accepted values remain 128 and 0.15
- implement the five-paper baseline comparison

## 3. Existing Code Reuse

The held-out implementation will add three entry-point scripts and import the
existing computational functions directly.

Seed 10 has already appeared in the completed full-data transductive
experiments. This run is therefore an acceptance test of the held-out
implementation and an internally consistent comparison, not a previously
untouched final test set. For a final paper claim, parameters must be frozen
before evaluating additional held-out seeds.

### Data and split

Reuse from `common_dichasus.py`:

- `load_data1(data_path, seed, test_ratio)`
- `p2_split_indices(n_samples, seed, test_ratio)`
- `DichasusData.train_idx`
- `DichasusData.test_idx`

The held-out scripts must not duplicate random split code.

### ADP/time distance

Reuse from `run_adp_fused.py`:

- `scaling_factor_from_time`
- `fused_geodesic`

The full `adp.npy` may be reused by extracting:

```python
adp_train = adp_full[np.ix_(train_idx, train_idx)]
```

This is valid because each ADP entry depends only on the CSI pair `(i, j)`.
Before extraction, the script must verify that the cached full ADP corresponds
to the same dataset order:

```python
np.array_equal(saved_indices, np.arange(num_samples))
```

The full cached `fused.npy` and `geodesic.npy` must not be extracted. The
temporal scale and graph shortest paths depend on the participating sample set.
They must be recomputed from `adp_train` and `timestamps[train_idx]`.

### Siamese FCF

Reuse from `run_adp_siamese.py`:

- `PairDataset`
- `ForwardChartingFunction`
- `siamese_loss`
- `train_schedule`
- `predict`

`PairDataset` receives only local training arrays:

```text
h = H[train_idx]
distances = geodesic_train
```

Therefore its internally generated pair indices can only refer to training
samples.

### Diffusion refinement

Reuse from `run_diffusion_refined_charting.py`:

- `compute_neighbor_indices`
- `build_local_scaled_graph`
- `solve_graph_harmonic_refinement`

These functions receive only `geodesic_train` and `Y_train`.

## 4. Architecture

Three new scripts preserve the existing pipeline boundaries:

```text
run_adp_fused_heldout.py
    full adp.npy + train_idx
        -> adp_train
        -> gamma_train
        -> fused_train
        -> geodesic_train

run_adp_siamese_heldout.py
    H_train + geodesic_train
        -> trained FCF
        -> Y_train
    H_test + trained FCF
        -> Y_test

run_diffusion_refined_heldout.py
    geodesic_train + Y_train
        -> training graph
        -> Z_train
```

The scripts use seed-specific output directories:

```text
results/heldout/seed10/
    adp_fused_train/
    adp_siamese/
    diffusion_train/
```

Later seeds use the same scripts and only change `--seed`.

Each stage derives its split, model initialization, sampled training pairs,
DataLoader shuffling, and metric subset from the requested seed. Outputs and
neighbor caches are stored below that seed's directory, so runs for different
seeds cannot overwrite or reuse incompatible artifacts.

## 5. Stage 1: Training-Only ADP/Time Graph

### Inputs

- `datasets/adp_old_shard4`
- the completed full-data distance run containing:
  - `adp.npy`
  - `distance_data.npz`
- `--seed`, initially 10
- initial fused-graph neighbor count 20, matching the completed full-data run

### Processing

1. Load the complete dataset with `load_data1`.
2. Read `train_idx` and `test_idx` from the returned object.
3. Verify:
   - split sizes are 16,798 and 4,199 for seed 10
   - indices are disjoint
   - their union covers all 20,997 samples
4. Load the full distance run's saved indices.
5. Require the saved indices to equal `0..20996` in the same order.
6. Memory-map `adp.npy` and require shape `(20997, 20997)`.
7. Extract `adp_train`.
8. Estimate `gamma_train` from `adp_train` and training timestamps using the
   same robust gamma mode as the completed run
   (`exact_notebook_gamma = false`).
9. Recompute `fused_train` and `geodesic_train` with initial neighbor count 20.
10. Save global training and test indices with the result metadata.

`adp_train` and `fused_train` are temporary arrays. They are not saved because
each would require approximately 1.13 GB and can be reconstructed from the
full ADP cache. `geodesic_train` is saved because shortest-path computation is
the expensive stage.

### Outputs

`results/heldout/seed10/adp_fused_train/` contains:

- `geodesic_train.npy`
- `distance_data.npz`
  - `indices`: global training indices, preserving the existing distance-run
    convention
  - `positions`: training positions in the same order
  - `test_indices`: global test indices in deterministic sampled order
- `result.json`
  - seed and split sizes
  - source full-distance run
  - `gamma_train`
  - final neighbor count
  - elapsed time

No test ADP, fused, or geodesic matrix is produced.

## 6. Stage 2: Held-Out Siamese FCF

### Inputs

- complete CSI dataset
- stage 1 `geodesic_train.npy`
- stage 1 `distance_data.npz`
- `--seed`, initially 10

### Processing

1. Load the same deterministic split.
2. Verify stage 1 indices exactly match the current split.
3. Construct:
   - `H_train = H[train_idx]`
   - `H_test = H[test_idx]`
4. Train the existing FCF using only `H_train` and `geodesic_train`.
5. Predict:
   - `Y_train = FCF(H_train)`
   - `Y_test = FCF(H_test)`
6. Compute training CT/TW/KS on a deterministic 20% subset for diagnostics.
7. Compute held-out CT/TW/KS on all 4,199 test samples.
8. Fit the affine coordinate transform using training positions and `Y_train`.
9. Apply that fixed transform to `Y_test`.
10. Compute held-out localization errors without refitting on test.

The existing `localization_error_summary(reference, embedding)` cannot be used
directly for held-out localization because it fits and evaluates on the same
samples. A minimal reusable fit/apply helper will be added while preserving the
existing API.

### Outputs

`results/heldout/seed10/adp_siamese/` contains:

- `model.pt`
- `embedding_train.npz`
  - `indices`
  - `positions`
  - `embedding`
- `embedding_test.npz`
  - `indices`
  - `positions`
  - `embedding`
- `result.json`
  - training history
  - train diagnostic metrics
  - held-out test metrics
  - held-out localization errors using train-fitted affine transform
- separate train and test chart images

The existing `plot_chart` function fits an affine transform internally, so it
must not be used for the held-out test chart. The stage 2 script will use a
small plotting helper that receives an already train-aligned test embedding.
It must not fit a new transform on test positions.

## 7. Stage 3: Training-Only Diffusion Refinement

### Inputs

- stage 1 `geodesic_train.npy`
- stage 2 `embedding_train.npz`
- `graph_k = 128`
- `lambda_smooth = 0.15`
- `iterations = 200`

### Processing

1. Verify the stage 1 and stage 2 global training indices are identical.
2. Compute geodesic neighbors using only `geodesic_train`.
3. Build the graph using only training nodes.
4. Refine `Y_train` to `Z_train`.
5. Generate one deterministic 20% training subset from the seed.
6. Compute before/after CT/TW/KS on that identical subset.
7. Compute train-only localization summaries as diagnostics.

Stage 2 `embedding_test.npz` is not an input to this script.

### Outputs

`results/heldout/seed10/diffusion_train/` contains:

- `embedding_train.npz`
  - `indices`
  - `positions`
  - refined `embedding`
- `geodesic_top512.npy`, cached in the stage 1 training-distance directory,
  matching the existing diffusion script convention
- `result.json`
  - graph parameters
  - graph size
  - baseline train metrics for `Y_train`
  - refined train metrics for `Z_train`
  - an explicit protocol label: `train_graph_diagnostic`
- `chart.png`

It does not contain `embedding_test.npz` or any held-out diffusion metrics.

## 8. Metric Semantics

The result reports must keep the following claims separate:

| Result | Meaning |
| --- | --- |
| `Y_train` | FCF performance on samples used to train the FCF |
| `Y_test` | held-out generalization of the trained FCF |
| `Z_train` | graph-refinement behavior on the training graph |
| `Z_test` | unavailable until out-of-sample extension is implemented |

The current experiment can establish FCF performance on samples excluded from
this run's graph construction and model training. Because seed 10 samples were
visible in earlier full-data experiments, this is not an untouched final-test
claim. It also cannot establish held-out performance of the diffusion
refinement itself.

## 9. Validation and Failure Conditions

The scripts fail before expensive computation when:

- cached full-data indices do not equal the complete dataset order
- ADP matrix dimensions do not match the dataset
- split indices overlap or do not cover the full dataset
- saved stage indices differ from the split for the requested seed
- geodesic dimensions differ from the training sample count
- any geodesic distance is non-finite
- embedding dimensions or indices differ from the training set
- `graph_k > top_k`

Tests will verify:

- seed 10 produces 16,798/4,199 disjoint indices
- two different seeds produce different splits and seed-specific output paths
- ADP extraction uses exactly `np.ix_(train_idx, train_idx)`
- stage 1 calls `fused_geodesic` with the extracted training ADP and training
  timestamps and never loads the full cached geodesic
- Siamese pair generation cannot address test samples
- train-fitted affine transformation is applied unchanged to test data
- stage 3 accepts only training indices and produces no test embedding
- all output embeddings preserve global dataset indices

## 10. Acceptance Criteria

The seed-10 held-out experiment is accepted when:

1. Stage 1 produces a finite `(16798, 16798)` training geodesic matrix and no
   test graph artifact.
2. Stage 2 trains only on 16,798 samples and saves `Y_train` with 16,798 rows
   and `Y_test` with 4,199 rows.
3. Stage 2 reports test CT/TW/KS and localization errors computed with a
   train-fitted affine transform.
4. Stage 3 refines only the 16,798-row training embedding with
   `graph_k=128`, `lambda_smooth=0.15`, and 200 iterations.
5. Stage 3 produces `Z_train` but no `Z_test`.
6. The original full-data scripts and their completed results remain
   unchanged.
7. Result metadata states that seed 10 is an implementation-acceptance
   held-out run whose samples were present in earlier full-data experiments.
8. The same commands accept another `--seed` without code changes and write to
   a separate seed-specific directory, although no additional seed is run for
   this acceptance task.
