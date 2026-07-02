import tempfile
import unittest
from pathlib import Path

import numpy as np

from run_adp_siamese import PairDataset
from run_adp_siamese_heldout import (
    evaluate_heldout_embeddings,
    prepare_heldout_arrays,
    validate_distance_run_split,
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

    def test_pair_dataset_indices_are_local_to_training_arrays(self):
        h_train = np.arange(3 * 2, dtype=np.float32).reshape(3, 2)
        distances = np.zeros((3, 3), dtype=np.float32)
        dataset = PairDataset(h_train, distances, samples=100, seed=10)

        self.assertLess(int(dataset.left.max()), len(h_train))
        self.assertLess(int(dataset.right.max()), len(h_train))

    def test_distance_run_split_must_match_requested_split(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            distance_run = Path(temp_dir)
            np.savez(
                distance_run / "distance_data.npz",
                indices=np.array([0, 2, 4], dtype=np.int64),
                positions=np.zeros((3, 2), dtype=np.float32),
                test_indices=np.array([1, 3], dtype=np.int64),
            )
            with self.assertRaisesRegex(ValueError, "split"):
                validate_distance_run_split(
                    distance_run,
                    train_idx=np.array([0, 2, 3], dtype=np.int64),
                    test_idx=np.array([1, 4], dtype=np.int64),
                )


if __name__ == "__main__":
    unittest.main()
