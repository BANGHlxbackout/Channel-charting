import unittest

import numpy as np

from archive.repair_gated_residual_correction.run_repair_gated_residual_correction import (
    apply_residual_gate,
    floor_top_gate,
    hard_top_gate,
    make_global_gate,
    make_random_gate,
    retained_repair_ratio,
)


class RepairGatedResidualCorrectionTests(unittest.TestCase):
    def test_hard_top_gate_selects_highest_scores(self):
        scores = np.array([0.1, 0.9, 0.4, 0.8, 0.2], dtype=np.float32)

        gate = hard_top_gate(scores, top_fraction=0.4)

        np.testing.assert_array_equal(gate, np.array([0.0, 1.0, 0.0, 1.0, 0.0], dtype=np.float32))

    def test_floor_top_gate_keeps_floor_for_unselected_nodes(self):
        scores = np.array([1.0, 3.0, 2.0, 0.0], dtype=np.float32)

        gate = floor_top_gate(scores, top_fraction=0.25, floor=0.3)

        np.testing.assert_allclose(gate, np.array([0.3, 1.0, 0.3, 0.3], dtype=np.float32))

    def test_apply_residual_gate_interpolates_between_y0_and_z_base(self):
        y0 = np.array([[0.0, 0.0], [10.0, 0.0]], dtype=np.float32)
        z_base = np.array([[2.0, 0.0], [14.0, 0.0]], dtype=np.float32)
        gate = np.array([0.25, 0.5], dtype=np.float32)

        corrected = apply_residual_gate(y0, z_base, gate)

        np.testing.assert_allclose(corrected, np.array([[0.5, 0.0], [12.0, 0.0]], dtype=np.float32))

    def test_make_global_gate_returns_constant_gate(self):
        gate = make_global_gate(n=4, gamma=0.6)

        np.testing.assert_allclose(gate, np.full(4, 0.6, dtype=np.float32))

    def test_make_random_gate_preserves_distribution(self):
        gate = np.array([0.0, 1.0, 0.0, 0.5, 1.0], dtype=np.float32)

        shuffled = make_random_gate(gate, seed=123)

        np.testing.assert_allclose(np.sort(shuffled), np.sort(gate))
        self.assertFalse(np.array_equal(shuffled, gate))

    def test_retained_repair_ratio_compares_outlier_mean_improvement(self):
        init_error = np.array([2.0, 3.0, 0.1], dtype=np.float32)
        baseline_error = np.array([1.0, 1.5, 0.2], dtype=np.float32)
        candidate_error = np.array([1.2, 1.8, 0.2], dtype=np.float32)
        outlier = np.array([True, True, False])

        ratio = retained_repair_ratio(init_error, baseline_error, candidate_error, outlier)

        self.assertAlmostEqual(ratio, 0.8)


if __name__ == "__main__":
    unittest.main()
