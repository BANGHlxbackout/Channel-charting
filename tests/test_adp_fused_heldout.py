import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from common_dichasus import p2_split_indices
from common_heldout import heldout_stage_dir, validate_split
from run_adp_fused_heldout import (
    compute_train_geodesic,
    validate_full_adp_cache,
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

    def test_real_full_cache_indices_match_complete_dataset_when_available(self):
        run = Path("results/adp_old_fused/stride1_n20997_seed10")
        if not (run / "adp.npy").exists():
            self.skipTest("full ADP cache is not available")

        adp, indices = validate_full_adp_cache(run, num_samples=20997)

        self.assertEqual(adp.shape, (20997, 20997))
        np.testing.assert_array_equal(indices, np.arange(20997))


if __name__ == "__main__":
    unittest.main()
