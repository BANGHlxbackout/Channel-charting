import unittest

import numpy as np

from diagnostics.sc2_stage1_multiview_check.run_pilot import rank_bins, rank_score, shuffle_within_strata


class MultiViewDiagnosticTests(unittest.TestCase):
    def test_higher_input_gets_higher_combined_score(self):
        full_score = rank_score(np.array([-1.0, -2.0, -3.0]))
        agreement_score = rank_score(np.array([0.9, 0.5, 0.1]))
        combined = full_score + agreement_score
        self.assertGreater(combined[0], combined[1])
        self.assertGreater(combined[1], combined[2])

    def test_shuffle_preserves_each_fixed_stratum_multiset(self):
        values = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        strata = np.array([0, 0, 0, 1, 1, 1])
        shuffled = shuffle_within_strata(values, strata)
        for label in (0, 1):
            self.assertEqual(sorted(shuffled[strata == label].tolist()), sorted(values[strata == label].tolist()))

    def test_rank_bins_are_deterministic_and_cover_all_items(self):
        labels = rank_bins(np.array([4.0, 1.0, 3.0, 2.0]), 2)
        np.testing.assert_array_equal(labels, np.array([1, 0, 1, 0]))


if __name__ == "__main__":
    unittest.main()