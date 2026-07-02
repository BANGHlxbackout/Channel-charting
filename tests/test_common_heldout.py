import unittest
from pathlib import Path

import numpy as np

from common_heldout import (
    apply_affine_transform,
    deterministic_subset_indices,
    fit_affine_transform,
    heldout_stage_dir,
    localization_error_summary_aligned,
    validate_split,
)


class CommonHeldoutTests(unittest.TestCase):
    def test_train_fitted_affine_is_applied_unchanged_to_test(self):
        train_embedding = np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
                [2.0, 0.0],
                [0.0, 2.0],
            ],
            dtype=np.float32,
        )
        test_embedding = np.array(
            [[2.0, 0.5], [-1.0, 2.0], [3.0, 1.0]],
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

    def test_subset_is_reproducible_sorted_and_seed_dependent(self):
        first = deterministic_subset_indices(100, fraction=0.2, seed=10)
        repeated = deterministic_subset_indices(100, fraction=0.2, seed=10)
        other = deterministic_subset_indices(100, fraction=0.2, seed=11)

        np.testing.assert_array_equal(first, repeated)
        self.assertEqual(len(first), 20)
        self.assertTrue(np.all(first[:-1] < first[1:]))
        self.assertFalse(np.array_equal(first, other))

    def test_validate_split_rejects_overlap_and_missing_samples(self):
        validate_split(
            5,
            np.array([0, 2, 4], dtype=np.int64),
            np.array([1, 3], dtype=np.int64),
        )
        with self.assertRaisesRegex(ValueError, "overlaps"):
            validate_split(
                5,
                np.array([0, 2, 4], dtype=np.int64),
                np.array([1, 2], dtype=np.int64),
            )
        with self.assertRaisesRegex(ValueError, "cover"):
            validate_split(
                5,
                np.array([0, 2], dtype=np.int64),
                np.array([1, 3], dtype=np.int64),
            )

    def test_heldout_stage_dir_is_seed_specific(self):
        root = Path("results/heldout")
        self.assertEqual(
            heldout_stage_dir(root, 10, "adp_fused_train"),
            Path("results/heldout/seed10/adp_fused_train"),
        )
        self.assertNotEqual(
            heldout_stage_dir(root, 10, "adp_fused_train"),
            heldout_stage_dir(root, 11, "adp_fused_train"),
        )


if __name__ == "__main__":
    unittest.main()
