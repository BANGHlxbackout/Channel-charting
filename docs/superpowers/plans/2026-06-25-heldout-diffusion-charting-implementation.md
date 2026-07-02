# Held-Out Diffusion Charting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a multi-seed-capable held-out ADP/time Siamese pipeline that trains and refines only the training graph, evaluates the FCF on held-out samples, and executes seed 10 for acceptance.

**Architecture:** Keep the three existing full-data scripts unchanged. Add three held-out entry points that import their computational functions, plus minimal affine fit/apply helpers in `common_metrics.py`. Every stage writes below `results/heldout/seed{seed}/`, validates global indices before expensive work, and never constructs or refines a test graph.

**Tech Stack:** Python 3.11, NumPy, SciPy, scikit-learn, PyTorch, Matplotlib, `unittest`, conda environment `diffusion_charting`.

---

## File Map

- Modify `common_metrics.py`: separate affine fitting from affine application and add aligned-error/subset helpers without changing existing behavior.
- Create `run_adp_fused_heldout.py`: validate the full ADP cache, extract the train ADP submatrix, and recompute train-only fused/geodesic distances.
- Create `run_adp_siamese_heldout.py`: train the existing FCF on train-only pairs and predict separate train/test embeddings.
- Create `run_diffusion_refined_heldout.py`: refine only the train embedding using the train geodesic graph.
- Create `tests/test_heldout_metrics.py`: affine and deterministic-subset tests.
- Create `tests/test_adp_fused_heldout.py`: split, cache, ADP extraction, and train-only geodesic tests.
- Create `tests/test_adp_siamese_heldout.py`: train/test array and held-out evaluation tests.
- Create `tests/test_diffusion_refined_heldout.py`: train-index validation and train-only output tests.
- Modify `README.md`: document the three held-out commands and the interpretation boundary between `Y_test` and `Z_train`.

Do not modify `run_adp_fused.py`, `run_adp_siamese.py`, or
`run_diffusion_refined_charting.py`.

### Task 1: Add Leakage-Safe Metric Helpers

**Files:**
- Modify: `common_metrics.py`
- Create: `tests/test_heldout_metrics.py`

- [ ] **Step 1: Write failing affine and subset tests**

```python
import unittest

import numpy as np

from common_metrics import (
    apply_affine_transform,
    deterministic_subset_indices,
    fit_affine_transform,
    localization_error_summary_aligned,
)


class HeldoutMetricTests(unittest.TestCase):
    def test_train_fitted_affine_is_applied_unchanged_to_test(self):
        train_embedding = np.array(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
            dtype=np.float32,
        )
        test_embedding = np.array(
            [[2.0, 0.5], [-1.0, 2.0]],
            dtype=np.float32,
        )
        linear = np.array([[2.0, -0.5], [0.25, 3.0]], dtype=np.float32)
        offset = np.array([4.0, -2.0], dtype=np.float32)
        train_reference = train_embedding @ linear + offset
        test_reference = test_embedding @ linear + offset

        transform = fit_affine_transform(train_reference, train_embedding)
        aligned_test = apply_affine_transform(test_embedding, transform)

        np.testing.assert_allclose(aligned_test, test_reference, atol=1e-5)
        errors = localization_error_summary_aligned(test_reference, aligned_test)
        self.assertLess(errors["max_m"], 1e-5)

    def test_subset_is_reproducible_and_seed_dependent(self):
        first = deterministic_subset_indices(100, fraction=0.2, seed=10)
        repeated = deterministic_subset_indices(100, fraction=0.2, seed=10)
        other = deterministic_subset_indices(100, fraction=0.2, seed=11)

        np.testing.assert_array_equal(first, repeated)
        self.assertEqual(len(first), 20)
        self.assertTrue(np.all(first[:-1] < first[1:]))
        self.assertFalse(np.array_equal(first, other))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the tests and verify the imports fail**

Run:

```powershell
conda run -n diffusion_charting python -m unittest tests.test_heldout_metrics -v
```

Expected: `ImportError` for the new helper names.

- [ ] **Step 3: Implement the minimal helpers and preserve the old API**

Add to `common_metrics.py`:

```python
def _pad_affine(values: np.ndarray) -> np.ndarray:
    return np.hstack(
        [values, np.ones((values.shape[0], 1), dtype=values.dtype)]
    )


def fit_affine_transform(reference: np.ndarray, embedding: np.ndarray) -> np.ndarray:
    transform, *_ = np.linalg.lstsq(
        _pad_affine(embedding),
        _pad_affine(reference),
        rcond=None,
    )
    return transform


def apply_affine_transform(
    embedding: np.ndarray,
    transform: np.ndarray,
) -> np.ndarray:
    return (_pad_affine(embedding) @ transform)[:, :-1]


def localization_error_summary_aligned(
    reference: np.ndarray,
    aligned_embedding: np.ndarray,
) -> dict[str, float]:
    errors = np.linalg.norm(aligned_embedding - reference, axis=1)
    return {
        "mean_m": float(np.mean(errors)),
        "median_m": float(np.median(errors)),
        "p90_m": float(np.percentile(errors, 90)),
        "p95_m": float(np.percentile(errors, 95)),
        "max_m": float(np.max(errors)),
    }


def deterministic_subset_indices(
    n_samples: int,
    fraction: float,
    seed: int,
) -> np.ndarray:
    subset_size = max(1, int(n_samples * fraction))
    rng = np.random.default_rng(seed)
    return np.sort(
        rng.choice(n_samples, size=subset_size, replace=False).astype(np.int64)
    )
```

Replace the bodies of the existing functions without changing their signatures:

```python
def affine_transform(reference: np.ndarray, embedding: np.ndarray) -> np.ndarray:
    transform = fit_affine_transform(reference, embedding)
    return apply_affine_transform(embedding, transform)


def localization_error_summary(
    reference: np.ndarray,
    embedding: np.ndarray,
) -> dict[str, float]:
    aligned = affine_transform(reference, embedding)
    return localization_error_summary_aligned(reference, aligned)
```

- [ ] **Step 4: Run metric and existing tests**

Run:

```powershell
conda run -n diffusion_charting python -m unittest tests.test_heldout_metrics tests.test_euclidean_knn_refinement -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit only the metric change**

```powershell
git add common_metrics.py tests/test_heldout_metrics.py
git commit -m "test: add held-out affine metric helpers"
```

### Task 2: Implement Stage 1 Train-Only Geodesic Construction

**Files:**
- Create: `run_adp_fused_heldout.py`
- Create: `tests/test_adp_fused_heldout.py`

- [ ] **Step 1: Write failing split, cache, and extraction tests**

```python
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from common_dichasus import p2_split_indices
from run_adp_fused_heldout import (
    compute_train_geodesic,
    heldout_stage_dir,
    validate_full_adp_cache,
    validate_split,
)


class AdpFusedHeldoutTests(unittest.TestCase):
    def test_seed10_split_is_complete_and_disjoint(self):
        train_idx, test_idx = p2_split_indices(20997, seed=10)
        validate_split(20997, train_idx, test_idx)
        self.assertEqual(len(train_idx), 16798)
        self.assertEqual(len(test_idx), 4199)

    def test_different_seeds_use_different_directories(self):
        root = Path("results/heldout")
        self.assertNotEqual(
            heldout_stage_dir(root, 10, "adp_fused_train"),
            heldout_stage_dir(root, 11, "adp_fused_train"),
        )

    def test_full_cache_requires_complete_ordered_indices(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            run = Path(temp_dir)
            np.save(run / "adp.npy", np.zeros((4, 4), dtype=np.float32))
            np.savez(
                run / "distance_data.npz",
                indices=np.array([0, 2, 1, 3], dtype=np.int64),
                positions=np.zeros((4, 2), dtype=np.float32),
            )
            with self.assertRaisesRegex(ValueError, "complete dataset order"):
                validate_full_adp_cache(run, num_samples=4)

    def test_train_geodesic_uses_only_extracted_train_inputs(self):
        adp_full = np.arange(25, dtype=np.float32).reshape(5, 5)
        timestamps = np.arange(5, dtype=np.float32)
        train_idx = np.array([0, 2, 4], dtype=np.int64)
        expected_adp = adp_full[np.ix_(train_idx, train_idx)]
        expected_times = timestamps[train_idx]
        fake_geodesic = np.full((3, 3), 7.0, dtype=np.float32)

        with patch(
            "run_adp_fused_heldout.scaling_factor_from_time",
            return_value=2.5,
        ) as gamma_fn, patch(
            "run_adp_fused_heldout.fused_geodesic",
            return_value=(
                np.zeros((3, 3), dtype=np.float32),
                fake_geodesic,
                20,
            ),
        ) as geodesic_fn:
            gamma, geodesic, used_neighbors = compute_train_geodesic(
                adp_full,
                timestamps,
                train_idx,
                n_neighbors=20,
            )

        np.testing.assert_array_equal(gamma_fn.call_args.args[0], expected_adp)
        np.testing.assert_array_equal(gamma_fn.call_args.args[1], expected_times)
        np.testing.assert_array_equal(geodesic_fn.call_args.args[0], expected_adp)
        np.testing.assert_array_equal(geodesic_fn.call_args.args[1], expected_times)
        self.assertEqual(gamma, 2.5)
        self.assertEqual(used_neighbors, 20)
        np.testing.assert_array_equal(geodesic, fake_geodesic)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the tests and verify the module is missing**

Run:

```powershell
conda run -n diffusion_charting python -m unittest tests.test_adp_fused_heldout -v
```

Expected: `ModuleNotFoundError: run_adp_fused_heldout`.

- [ ] **Step 3: Add the Stage 1 helper functions**

Create `run_adp_fused_heldout.py` with these testable functions:

```python
def heldout_stage_dir(root: Path, seed: int, stage: str) -> Path:
    return root / f"seed{seed}" / stage


def validate_split(
    n_samples: int,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> None:
    if len(np.intersect1d(train_idx, test_idx)) != 0:
        raise ValueError("train/test split overlaps")
    union = np.union1d(train_idx, test_idx)
    if not np.array_equal(union, np.arange(n_samples)):
        raise ValueError("train/test split does not cover the dataset")


def validate_full_adp_cache(
    full_distance_run: Path,
    num_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    saved = np.load(full_distance_run / "distance_data.npz")
    indices = saved["indices"]
    expected = np.arange(num_samples, dtype=indices.dtype)
    if not np.array_equal(indices, expected):
        raise ValueError("full ADP cache indices do not match complete dataset order")
    adp = np.load(full_distance_run / "adp.npy", mmap_mode="r")
    if adp.shape != (num_samples, num_samples):
        raise ValueError(
            f"expected ADP shape {(num_samples, num_samples)}, got {adp.shape}"
        )
    return adp, indices


def compute_train_geodesic(
    adp_full: np.ndarray,
    timestamps: np.ndarray,
    train_idx: np.ndarray,
    n_neighbors: int,
) -> tuple[float, np.ndarray, int]:
    adp_train = np.asarray(
        adp_full[np.ix_(train_idx, train_idx)],
        dtype=np.float32,
    )
    train_timestamps = timestamps[train_idx]
    gamma = scaling_factor_from_time(
        adp_train,
        train_timestamps,
        exact_notebook=False,
    )
    fused_train, geodesic_train, used_neighbors = fused_geodesic(
        adp_train,
        train_timestamps,
        gamma=gamma,
        n_neighbors=n_neighbors,
    )
    del fused_train
    return gamma, geodesic_train, used_neighbors
```

Import `load_data1`, `scaling_factor_from_time`, `fused_geodesic`, and
`write_json` from the existing modules.

- [ ] **Step 4: Run Stage 1 unit tests**

Run:

```powershell
conda run -n diffusion_charting python -m unittest tests.test_adp_fused_heldout -v
```

Expected: all four tests pass.

- [ ] **Step 5: Add the Stage 1 CLI**

Use these arguments:

```python
parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA1)
parser.add_argument("--full-distance-run", type=Path, required=True)
parser.add_argument("--out-dir", type=Path, default=Path("results/heldout"))
parser.add_argument("--seed", type=int, default=10)
parser.add_argument("--neighbors", type=int, default=20)
parser.add_argument("--force", action="store_true")
```

The CLI must:

```python
data = load_data1(args.data_path, seed=args.seed)
validate_split(len(data.h_flat), data.train_idx, data.test_idx)
out_dir = heldout_stage_dir(args.out_dir, args.seed, "adp_fused_train")
geodesic_path = out_dir / "geodesic_train.npy"
```

Before computing, validate the full ADP cache. If all three Stage 1 output
files exist and `--force` is absent, validate the saved indices and reuse the
cached result. Otherwise compute and save:

```python
np.save(geodesic_path, geodesic_train)
np.savez(
    out_dir / "distance_data.npz",
    indices=data.train_idx,
    positions=data.ue_pos[data.train_idx],
    test_indices=data.test_idx,
)
```

Write `result.json` with:

```python
{
    "protocol": "heldout_train_graph",
    "dataset": str(args.data_path),
    "source_full_distance_run": str(args.full_distance_run),
    "seed": int(args.seed),
    "num_samples": int(len(data.h_flat)),
    "num_train": int(len(data.train_idx)),
    "num_test": int(len(data.test_idx)),
    "gamma_train": float(gamma),
    "neighbors": int(used_neighbors),
    "geodesic_shape": list(geodesic_train.shape),
    "seed10_historical_visibility": bool(args.seed == 10),
    "elapsed_s": float(time.time() - started),
}
```

Do not load `full_distance_run/geodesic.npy`. Do not save train ADP or fused
matrices.

- [ ] **Step 6: Add a cheap real-data validation mode test**

Add a test that calls `validate_full_adp_cache` on:

```text
results/adp_old_fused/stride1_n20997_seed10
```

Guard it with `unittest.skipUnless` when the local cache does not exist. Assert:

```python
self.assertEqual(adp.shape, (20997, 20997))
np.testing.assert_array_equal(indices, np.arange(20997))
```

- [ ] **Step 7: Run Stage 1 and full unit suites**

Run:

```powershell
conda run -n diffusion_charting python -m unittest tests.test_adp_fused_heldout -v
conda run -n diffusion_charting python -m unittest discover -s tests -v
```

Expected: all tests pass without computing the 16,798-sample geodesic.

- [ ] **Step 8: Commit Stage 1**

```powershell
git add run_adp_fused_heldout.py tests/test_adp_fused_heldout.py
git commit -m "feat: add held-out train geodesic stage"
```

### Task 3: Implement Stage 2 Held-Out Siamese FCF

**Files:**
- Create: `run_adp_siamese_heldout.py`
- Create: `tests/test_adp_siamese_heldout.py`

- [ ] **Step 1: Write failing train/test and evaluation tests**

```python
import unittest

import numpy as np

from run_adp_siamese_heldout import (
    evaluate_heldout_embeddings,
    prepare_heldout_arrays,
)


class AdpSiameseHeldoutTests(unittest.TestCase):
    def test_prepare_arrays_preserves_global_split_order(self):
        h = np.arange(6 * 2, dtype=np.float32).reshape(6, 2)
        positions = np.arange(6 * 2, dtype=np.float32).reshape(6, 2)
        train_idx = np.array([0, 2, 5], dtype=np.int64)
        test_idx = np.array([4, 1, 3], dtype=np.int64)

        h_train, h_test, pos_train, pos_test = prepare_heldout_arrays(
            h,
            positions,
            train_idx,
            test_idx,
        )

        np.testing.assert_array_equal(h_train, h[train_idx])
        np.testing.assert_array_equal(h_test, h[test_idx])
        np.testing.assert_array_equal(pos_train, positions[train_idx])
        np.testing.assert_array_equal(pos_test, positions[test_idx])

    def test_test_localization_uses_train_fitted_affine(self):
        y_train = np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
                [2.0, 0.0],
                [0.0, 2.0],
                [2.0, 2.0],
                [1.0, 2.0],
            ],
            dtype=np.float32,
        )
        y_test = np.array(
            [
                [2.0, 0.5],
                [-1.0, 2.0],
                [3.0, 1.0],
                [0.5, 3.0],
                [-2.0, -1.0],
                [4.0, 2.0],
            ],
            dtype=np.float32,
        )
        linear = np.array([[1.5, 0.25], [-0.5, 2.0]], dtype=np.float32)
        offset = np.array([3.0, -4.0], dtype=np.float32)
        pos_train = y_train @ linear + offset
        pos_test = y_test @ linear + offset

        result = evaluate_heldout_embeddings(
            pos_train,
            y_train,
            pos_test,
            y_test,
            seed=10,
            metric_subset_fraction=0.75,
        )

        self.assertLess(result["test_localization_errors"]["max_m"], 1e-5)
        np.testing.assert_allclose(result["aligned_test"], pos_test, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the tests and verify the module is missing**

Run:

```powershell
conda run -n diffusion_charting python -m unittest tests.test_adp_siamese_heldout -v
```

Expected: `ModuleNotFoundError: run_adp_siamese_heldout`.

- [ ] **Step 3: Implement preparation and evaluation helpers**

Create:

```python
def prepare_heldout_arrays(
    h: np.ndarray,
    positions: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return (
        h[train_idx],
        h[test_idx],
        positions[train_idx],
        positions[test_idx],
    )
```

Implement:

```python
def evaluate_heldout_embeddings(
    pos_train: np.ndarray,
    y_train: np.ndarray,
    pos_test: np.ndarray,
    y_test: np.ndarray,
    seed: int,
    metric_subset_fraction: float,
) -> dict:
    train_subset = deterministic_subset_indices(
        len(pos_train),
        metric_subset_fraction,
        seed,
    )
    transform = fit_affine_transform(pos_train, y_train)
    aligned_train = apply_affine_transform(y_train, transform)
    aligned_test = apply_affine_transform(y_test, transform)
    return {
        "train_metric_subset": train_subset,
        "train_metrics": chart_metrics(
            pos_train[train_subset],
            y_train[train_subset],
        ),
        "train_localization_errors": localization_error_summary_aligned(
            pos_train,
            aligned_train,
        ),
        "test_metrics": chart_metrics(pos_test, y_test),
        "test_localization_errors": localization_error_summary_aligned(
            pos_test,
            aligned_test,
        ),
        "aligned_train": aligned_train,
        "aligned_test": aligned_test,
        "affine_transform": transform,
    }
```

- [ ] **Step 4: Run Stage 2 helper tests**

Run:

```powershell
conda run -n diffusion_charting python -m unittest tests.test_adp_siamese_heldout -v
```

Expected: both tests pass.

- [ ] **Step 5: Implement the reused FCF training loop**

Import:

```python
from run_adp_siamese import (
    ForwardChartingFunction,
    PairDataset,
    predict,
    siamese_loss,
    train_schedule,
)
```

Add a `train_fcf(...)` function that follows the existing training loop
exactly:

```python
model = ForwardChartingFunction(input_shape=h_train.shape[1:]).to(device)
optimizer = torch.optim.Adam(model.parameters(), eps=1e-7)
schedule = train_schedule()
if session_limit is not None:
    schedule = schedule[:session_limit]
```

For every session, use:

```python
effective_samples = (requested_samples // batch_size) * batch_size
if effective_samples == 0:
    raise ValueError(
        f"samples_per_session must be at least batch size {batch_size}"
    )
dataset = PairDataset(
    h_train,
    geodesic_train,
    samples=effective_samples,
    seed=seed + session_idx,
)
```

Return `model` and `history`. This preserves the existing architecture,
schedule, loss, optimizer, pair generation, and session seed behavior.

- [ ] **Step 6: Add the Stage 2 CLI**

Use:

```python
parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA1)
parser.add_argument("--distance-run", type=Path, required=True)
parser.add_argument("--out-dir", type=Path, default=Path("results/heldout"))
parser.add_argument("--seed", type=int, default=10)
parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
parser.add_argument("--session-limit", type=int, default=None)
parser.add_argument("--samples-per-session", type=int, default=None)
parser.add_argument("--metric-subset-fraction", type=float, default=0.2)
```

Set all random seeds before model creation:

```python
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)
```

Validate that Stage 1 `distance_data.npz` has:

```python
np.array_equal(saved["indices"], data.train_idx)
np.array_equal(saved["test_indices"], data.test_idx)
```

Load `geodesic_train.npy`, require shape `(num_train, num_train)`, and reject
non-finite entries. Train on `H_train`, then call:

```python
out_dir = args.out_dir / f"seed{args.seed}" / "adp_siamese"
out_dir.mkdir(parents=True, exist_ok=True)
y_train = predict(model, h_train, device)
y_test = predict(model, h_test, device)
evaluation = evaluate_heldout_embeddings(...)
```

Save raw embeddings with global indices:

```python
np.savez(
    out_dir / "embedding_train.npz",
    indices=data.train_idx,
    positions=pos_train,
    embedding=y_train,
)
np.savez(
    out_dir / "embedding_test.npz",
    indices=data.test_idx,
    positions=pos_test,
    embedding=y_test,
)
torch.save(model.state_dict(), out_dir / "model.pt")
```

Write `result.json` with protocol `heldout_fcf`, train/test counts, training
history, train subset metrics, all-test metrics, train-fitted test localization
errors, distance-run path, and historical-visibility flag.

- [ ] **Step 7: Implement plotting without test refitting**

Add:

```python
def plot_aligned_chart(
    path: Path,
    reference: np.ndarray,
    aligned_embedding: np.ndarray,
    title: str,
) -> None:
```

Copy only the Matplotlib scatter/color logic from the existing `plot_chart`;
do not call `affine_transform`. Produce:

```text
chart_train.png
chart_test.png
```

using `evaluation["aligned_train"]` and `evaluation["aligned_test"]`.

- [ ] **Step 8: Add a pair-isolation test**

Add:

```python
def test_pair_dataset_indices_are_local_to_training_arrays(self):
    h_train = np.arange(3 * 2, dtype=np.float32).reshape(3, 2)
    distances = np.zeros((3, 3), dtype=np.float32)
    dataset = PairDataset(h_train, distances, samples=100, seed=10)
    self.assertLess(int(dataset.left.max()), len(h_train))
    self.assertLess(int(dataset.right.max()), len(h_train))
```

This verifies the reused pair generator has no path to test samples after
receiving train-only arrays.

- [ ] **Step 9: Run Stage 2 and full tests**

Run:

```powershell
conda run -n diffusion_charting python -m unittest tests.test_adp_siamese_heldout -v
conda run -n diffusion_charting python -m unittest discover -s tests -v
```

Expected: all tests pass without starting full FCF training.

- [ ] **Step 10: Commit Stage 2**

```powershell
git add run_adp_siamese_heldout.py tests/test_adp_siamese_heldout.py
git commit -m "feat: add held-out Siamese FCF stage"
```

### Task 4: Implement Stage 3 Train-Only Diffusion Refinement

**Files:**
- Create: `run_diffusion_refined_heldout.py`
- Create: `tests/test_diffusion_refined_heldout.py`

- [ ] **Step 1: Write failing input-validation and output tests**

```python
import tempfile
import unittest
from pathlib import Path

import numpy as np

from run_diffusion_refined_heldout import (
    load_train_inputs,
    save_train_refinement,
)
from common_dichasus import p2_split_indices


class DiffusionRefinedHeldoutTests(unittest.TestCase):
    def test_mismatched_training_indices_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            distance_run = root / "distance"
            distance_run.mkdir()
            train_idx, test_idx = p2_split_indices(5, seed=10)
            np.save(
                distance_run / "geodesic_train.npy",
                np.zeros((len(train_idx), len(train_idx)), dtype=np.float32),
            )
            np.savez(
                distance_run / "distance_data.npz",
                indices=train_idx,
                positions=np.zeros((len(train_idx), 2), dtype=np.float32),
                test_indices=test_idx,
            )
            init_path = root / "embedding_train.npz"
            wrong_indices = train_idx.copy()
            wrong_indices[-1] = test_idx[0]
            np.savez(
                init_path,
                indices=wrong_indices,
                positions=np.zeros((len(train_idx), 2), dtype=np.float32),
                embedding=np.zeros((len(train_idx), 2), dtype=np.float32),
            )
            with self.assertRaisesRegex(ValueError, "training indices"):
                load_train_inputs(distance_run, init_path, seed=10)

    def test_requested_seed_must_match_saved_split(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            distance_run = root / "distance"
            distance_run.mkdir()
            train_idx, test_idx = p2_split_indices(10, seed=10)
            np.save(
                distance_run / "geodesic_train.npy",
                np.zeros((len(train_idx), len(train_idx)), dtype=np.float32),
            )
            np.savez(
                distance_run / "distance_data.npz",
                indices=train_idx,
                positions=np.zeros((len(train_idx), 2), dtype=np.float32),
                test_indices=test_idx,
            )
            init_path = root / "embedding_train.npz"
            np.savez(
                init_path,
                indices=train_idx,
                positions=np.zeros((len(train_idx), 2), dtype=np.float32),
                embedding=np.zeros((len(train_idx), 2), dtype=np.float32),
            )
            with self.assertRaisesRegex(ValueError, "requested seed"):
                load_train_inputs(distance_run, init_path, seed=11)

    def test_save_refinement_creates_only_train_embedding(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            out_dir = Path(temp_dir)
            save_train_refinement(
                out_dir,
                indices=np.array([0, 2, 4]),
                positions=np.zeros((3, 2), dtype=np.float32),
                embedding=np.ones((3, 2), dtype=np.float32),
            )
            self.assertTrue((out_dir / "embedding_train.npz").exists())
            self.assertFalse((out_dir / "embedding_test.npz").exists())


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the tests and verify the module is missing**

Run:

```powershell
conda run -n diffusion_charting python -m unittest tests.test_diffusion_refined_heldout -v
```

Expected: `ModuleNotFoundError: run_diffusion_refined_heldout`.

- [ ] **Step 3: Implement Stage 3 input and output helpers**

```python
def load_train_inputs(
    distance_run: Path,
    init_embedding_path: Path,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    distance_data = np.load(distance_run / "distance_data.npz")
    init_data = np.load(init_embedding_path)
    distance_indices = distance_data["indices"]
    test_indices = distance_data["test_indices"]
    num_samples = len(distance_indices) + len(test_indices)
    expected_train, expected_test = p2_split_indices(num_samples, seed=seed)
    if not np.array_equal(distance_indices, expected_train) or not np.array_equal(
        test_indices,
        expected_test,
    ):
        raise ValueError("saved split does not match requested seed")
    init_indices = init_data["indices"]
    if not np.array_equal(distance_indices, init_indices):
        raise ValueError("distance and embedding training indices do not match")
    geodesic = np.load(distance_run / "geodesic_train.npy").astype(np.float32)
    if geodesic.shape != (len(init_indices), len(init_indices)):
        raise ValueError("training geodesic shape does not match training indices")
    if not np.all(np.isfinite(geodesic)):
        raise ValueError("training geodesic contains non-finite values")
    return (
        init_indices,
        init_data["positions"].astype(np.float32),
        init_data["embedding"].astype(np.float32),
        geodesic,
    )


def save_train_refinement(
    out_dir: Path,
    indices: np.ndarray,
    positions: np.ndarray,
    embedding: np.ndarray,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_dir / "embedding_train.npz",
        indices=indices,
        positions=positions,
        embedding=embedding,
    )
```

- [ ] **Step 4: Run Stage 3 helper tests**

Run:

```powershell
conda run -n diffusion_charting python -m unittest tests.test_diffusion_refined_heldout -v
```

Expected: both tests pass.

- [ ] **Step 5: Add the Stage 3 CLI and reuse diffusion functions**

Import:

```python
from run_diffusion_refined_charting import (
    build_local_scaled_graph,
    compute_neighbor_indices,
    solve_graph_harmonic_refinement,
)
from run_adp_siamese import plot_chart
```

Use:

```python
parser.add_argument("--distance-run", type=Path, required=True)
parser.add_argument("--init-embedding", type=Path, required=True)
parser.add_argument("--out-dir", type=Path, default=Path("results/heldout"))
parser.add_argument("--seed", type=int, default=10)
parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
parser.add_argument("--graph-k", type=int, default=128)
parser.add_argument("--top-k", type=int, default=512)
parser.add_argument("--lambda-smooth", type=float, default=0.15)
parser.add_argument("--iterations", type=int, default=200)
parser.add_argument("--metric-subset-fraction", type=float, default=0.2)
```

Require:

```python
if args.graph_k > args.top_k:
    raise ValueError("--graph-k must be <= --top-k")
```

Set the output to:

```python
out_dir = args.out_dir / f"seed{args.seed}" / "diffusion_train"
neighbor_cache = args.distance_run / f"geodesic_top{args.top_k}.npy"
```

Load inputs with the requested seed:

```python
train_indices, train_positions, y_train, geodesic_train = load_train_inputs(
    args.distance_run,
    args.init_embedding,
    seed=args.seed,
)
```

Run only:

```python
neighbors = compute_neighbor_indices(
    geodesic_train,
    args.top_k,
    device,
    neighbor_cache,
)
graph = build_local_scaled_graph(
    geodesic_train,
    neighbors,
    args.graph_k,
)
z_train = solve_graph_harmonic_refinement(
    y_train,
    graph,
    lambda_smooth=args.lambda_smooth,
    iterations=args.iterations,
    device=device,
)
```

Evaluate `Y_train` and `Z_train` on the same deterministic subset:

```python
subset = deterministic_subset_indices(
    len(train_positions),
    args.metric_subset_fraction,
    args.seed,
)
baseline_metrics = chart_metrics(
    train_positions[subset],
    y_train[subset],
)
metrics = chart_metrics(
    train_positions[subset],
    z_train[subset],
)
```

Train localization summaries may use the existing same-set
`localization_error_summary`, because they are explicitly labeled diagnostics.

Write `result.json` with:

```python
{
    "protocol": "train_graph_diagnostic",
    "seed": int(args.seed),
    "num_train": int(len(train_indices)),
    "graph_k": int(args.graph_k),
    "top_k": int(args.top_k),
    "lambda_smooth": float(args.lambda_smooth),
    "iterations": int(args.iterations),
    "graph_edges": int(graph.nnz),
    "metric_subset_size": int(len(subset)),
    "baseline_train_metrics": baseline_metrics,
    "refined_train_metrics": metrics,
    "baseline_train_localization_errors": baseline_errors,
    "refined_train_localization_errors": errors,
    "heldout_diffusion_metrics_available": False,
    "elapsed_s": float(time.time() - started),
}
```

Save only `embedding_train.npz` and `chart.png`.

- [ ] **Step 6: Add a test that test embeddings are not inputs**

In a temporary Stage 2 directory, create both `embedding_train.npz` and an
invalid `embedding_test.npz`. Call `load_train_inputs` with the explicit train
path and matching seed and assert it succeeds. This verifies Stage 3 neither
reads nor validates test embeddings.

- [ ] **Step 7: Run Stage 3 and full tests**

Run:

```powershell
conda run -n diffusion_charting python -m unittest tests.test_diffusion_refined_heldout -v
conda run -n diffusion_charting python -m unittest discover -s tests -v
```

Expected: all tests pass.

- [ ] **Step 8: Commit Stage 3**

```powershell
git add run_diffusion_refined_heldout.py tests/test_diffusion_refined_heldout.py
git commit -m "feat: add train-only diffusion held-out stage"
```

### Task 5: Document Commands and Protocol Boundaries

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Add the held-out experiment section**

Add commands:

```powershell
conda activate diffusion_charting

python run_adp_fused_heldout.py `
  --data-path datasets/adp_old_shard4 `
  --full-distance-run results/adp_old_fused/stride1_n20997_seed10 `
  --out-dir results/heldout `
  --seed 10 `
  --neighbors 20

python run_adp_siamese_heldout.py `
  --data-path datasets/adp_old_shard4 `
  --distance-run results/heldout/seed10/adp_fused_train `
  --out-dir results/heldout `
  --seed 10

python run_diffusion_refined_heldout.py `
  --distance-run results/heldout/seed10/adp_fused_train `
  --init-embedding results/heldout/seed10/adp_siamese/embedding_train.npz `
  --out-dir results/heldout `
  --seed 10 `
  --graph-k 128 `
  --lambda-smooth 0.15 `
  --iterations 200
```

State explicitly:

```text
Y_test is the held-out FCF result.
Z_train is a training-graph refinement diagnostic.
There is no Z_test until out-of-sample extension is implemented.
The scripts support other seeds, but this acceptance run executes seed 10 only.
```

- [ ] **Step 2: Check command paths against parser definitions**

Run:

```powershell
conda run -n diffusion_charting python run_adp_fused_heldout.py --help
conda run -n diffusion_charting python run_adp_siamese_heldout.py --help
conda run -n diffusion_charting python run_diffusion_refined_heldout.py --help
```

Expected: all commands exit 0 and list the documented arguments.

- [ ] **Step 3: Commit documentation**

```powershell
git add README.md
git commit -m "docs: add held-out experiment workflow"
```

### Task 6: Static Verification Before Expensive Execution

**Files:**
- Verify only; no new file required.

- [ ] **Step 1: Run the complete test suite**

```powershell
conda run -n diffusion_charting python -m unittest discover -s tests -v
```

Expected: all tests pass.

- [ ] **Step 2: Compile all changed Python files**

```powershell
conda run -n diffusion_charting python -m py_compile `
  common_metrics.py `
  run_adp_fused_heldout.py `
  run_adp_siamese_heldout.py `
  run_diffusion_refined_heldout.py
```

Expected: exit code 0 with no output.

- [ ] **Step 3: Confirm original script hashes are unchanged**

```powershell
Get-FileHash `
  run_adp_fused.py,`
  run_adp_siamese.py,`
  run_diffusion_refined_charting.py `
  -Algorithm SHA256 | ForEach-Object { "$($_.Path)|$($_.Hash)" }
```

Expected hashes from the approved pre-implementation workspace:

```text
run_adp_fused.py|54F2F7D6C839F819005603D45D19D9A88987E2D7BADB29F10AC76751461C4BFE
run_adp_siamese.py|DDBA370003A4F2DA6D1617D2E5A82D53CECB8AA24712299643386976CC44B25B
run_diffusion_refined_charting.py|F255B92CADF4DA770EA3986D1589BFC54C4C24F9FB82F7019A3C07B3987AAB0F
```

The full printed paths may differ, but the filename/hash pairs must match.

- [ ] **Step 4: Confirm no code path loads the full cached geodesic in Stage 1**

```powershell
rg -n "geodesic\\.npy|fused\\.npy" run_adp_fused_heldout.py
```

Expected: no reference to the full run's `geodesic.npy` or `fused.npy`;
`geodesic_train.npy` is allowed.

- [ ] **Step 5: Confirm Stage 3 has no test embedding output**

```powershell
rg -n "embedding_test|Z_test|test_embedding" run_diffusion_refined_heldout.py
```

Expected: no executable reference; a result metadata statement that held-out
diffusion is unavailable is allowed.

### Task 7: Execute Seed 10 Acceptance Run

**Files:**
- Generate: `results/heldout/seed10/adp_fused_train/*`
- Generate: `results/heldout/seed10/adp_siamese/*`
- Generate: `results/heldout/seed10/diffusion_train/*`

- [ ] **Step 1: Run Stage 1**

```powershell
conda run -n diffusion_charting python run_adp_fused_heldout.py `
  --data-path datasets/adp_old_shard4 `
  --full-distance-run results/adp_old_fused/stride1_n20997_seed10 `
  --out-dir results/heldout `
  --seed 10 `
  --neighbors 20
```

Expected:

- `geodesic_train.npy` shape `(16798, 16798)`
- all values finite
- `distance_data.npz["indices"]` equals `load_data1(..., seed=10).train_idx`
- no test graph file

- [ ] **Step 2: Verify Stage 1 artifacts**

```powershell
conda run -n diffusion_charting python -c "import numpy as np; from common_dichasus import load_data1; d=load_data1(r'datasets/adp_old_shard4', seed=10); r=r'results/heldout/seed10/adp_fused_train'; g=np.load(r+'\\geodesic_train.npy', mmap_mode='r'); s=np.load(r+'\\distance_data.npz'); print(g.shape, np.isfinite(g).all(), np.array_equal(s['indices'], d.train_idx), np.array_equal(s['test_indices'], d.test_idx))"
```

Expected:

```text
(16798, 16798) True True True
```

- [ ] **Step 3: Run Stage 2**

```powershell
conda run -n diffusion_charting python run_adp_siamese_heldout.py `
  --data-path datasets/adp_old_shard4 `
  --distance-run results/heldout/seed10/adp_fused_train `
  --out-dir results/heldout `
  --seed 10
```

Expected:

- eight training sessions complete
- `embedding_train.npz` contains 16,798 rows
- `embedding_test.npz` contains 4,199 rows
- result contains held-out test metrics and train-fitted affine errors

- [ ] **Step 4: Verify Stage 2 artifacts**

```powershell
conda run -n diffusion_charting python -c "import json,numpy as np; r=r'results/heldout/seed10/adp_siamese'; tr=np.load(r+'\\embedding_train.npz'); te=np.load(r+'\\embedding_test.npz'); j=json.load(open(r+'\\result.json')); print(tr['embedding'].shape, te['embedding'].shape, j['protocol'], sorted(j['test_metrics']))"
```

Expected:

```text
(16798, 2) (4199, 2) heldout_fcf ['ct', 'ks', 'n_neighbors', 'tw']
```

- [ ] **Step 5: Run Stage 3**

```powershell
conda run -n diffusion_charting python run_diffusion_refined_heldout.py `
  --distance-run results/heldout/seed10/adp_fused_train `
  --init-embedding results/heldout/seed10/adp_siamese/embedding_train.npz `
  --out-dir results/heldout `
  --seed 10 `
  --graph-k 128 `
  --lambda-smooth 0.15 `
  --iterations 200
```

Expected:

- train-only top-512 neighbors are computed or loaded
- Jacobi iterations reach 200
- `embedding_train.npz` contains 16,798 rows
- no `embedding_test.npz` is created

- [ ] **Step 6: Verify Stage 3 artifacts**

```powershell
conda run -n diffusion_charting python -c "import json,numpy as np; from pathlib import Path; r=Path(r'results/heldout/seed10/diffusion_train'); e=np.load(r/'embedding_train.npz'); j=json.load(open(r/'result.json')); print(e['embedding'].shape, j['protocol'], j['graph_k'], j['lambda_smooth'], j['heldout_diffusion_metrics_available'], (r/'embedding_test.npz').exists())"
```

Expected:

```text
(16798, 2) train_graph_diagnostic 128 0.15 False False
```

- [ ] **Step 7: Run final verification**

```powershell
conda run -n diffusion_charting python -m unittest discover -s tests -v
git status --short
```

Expected: all tests pass. Generated results may appear as ignored or untracked
large artifacts; do not commit them unless the repository's existing policy
explicitly tracks experiment outputs.

### Task 8: Report Acceptance Results

**Files:**
- Read: `results/heldout/seed10/*/result.json`

- [ ] **Step 1: Report protocol correctness before metric quality**

The completion report must state:

```text
Stage 1: train-only geodesic, 16,798 nodes.
Stage 2: FCF trained on train only; Y_test contains 4,199 held-out samples.
Stage 3: diffusion applied only to Y_train; no Z_test exists.
```

- [ ] **Step 2: Report metrics using separate labels**

Report:

- `Y_train`: training diagnostic CT/TW/KS and localization errors
- `Y_test`: held-out FCF CT/TW/KS and train-fitted affine localization errors
- `Z_train`: train-graph diagnostic before/after refinement

Do not place `Z_train` metrics in a column labeled held-out or test.

- [ ] **Step 3: Record the scientific limitation**

State:

```text
Seed 10 was used in earlier full-data experiments, so this run validates the
held-out implementation but is not a historically untouched final test.
Held-out diffusion performance requires the later out-of-sample extension.
```
