import tempfile
import unittest
from pathlib import Path

import numpy as np

from common_dichasus import p2_split_indices
from run_diffusion_refined_heldout import load_train_inputs, save_train_refinement


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
            wrong_indices = train_idx.copy()
            wrong_indices[-1] = test_idx[0]
            init_path = root / "embedding_train.npz"
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
                indices=np.array([0, 2, 4], dtype=np.int64),
                positions=np.zeros((3, 2), dtype=np.float32),
                embedding=np.ones((3, 2), dtype=np.float32),
            )

            self.assertTrue((out_dir / "embedding_train.npz").exists())
            self.assertFalse((out_dir / "embedding_test.npz").exists())

    def test_test_embedding_file_is_not_read(self):
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
            train_path = root / "embedding_train.npz"
            np.savez(
                train_path,
                indices=train_idx,
                positions=np.zeros((len(train_idx), 2), dtype=np.float32),
                embedding=np.zeros((len(train_idx), 2), dtype=np.float32),
            )
            np.savez(
                root / "embedding_test.npz",
                indices=np.array([999], dtype=np.int64),
                positions=np.full((1, 2), np.nan, dtype=np.float32),
                embedding=np.full((1, 2), np.nan, dtype=np.float32),
            )

            indices, positions, embedding, geodesic = load_train_inputs(
                distance_run,
                train_path,
                seed=10,
            )

            np.testing.assert_array_equal(indices, train_idx)
            self.assertEqual(positions.shape, (len(train_idx), 2))
            self.assertEqual(embedding.shape, (len(train_idx), 2))
            self.assertEqual(geodesic.shape, (len(train_idx), len(train_idx)))


if __name__ == "__main__":
    unittest.main()
