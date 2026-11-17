"""Training weights must depend only on eligible past dates."""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from paper_models import recency_date_weights


class PaperWeightTests(unittest.TestCase):
    def test_equal_date_mass_and_half_life(self):
        sample = pd.DataFrame({"date": pd.to_datetime(["2020-01-01", "2020-01-01", "2024-01-01"])})
        weights = recency_date_weights(sample, pd.Timestamp("2024-01-01"))
        self.assertAlmostEqual(weights.mean(), 1)
        self.assertAlmostEqual(weights[0], weights[1])
        # Four years exactly 1461 days = 365.25 * 4.
        self.assertAlmostEqual(weights[2] / weights[:2].sum(), 2)

    def test_future_observation_is_rejected(self):
        with self.assertRaises(ValueError):
            recency_date_weights(pd.DataFrame({"date": ["2024-01-02"]}), pd.Timestamp("2024-01-01"))

    def test_order_does_not_change_weight_membership(self):
        sample = pd.DataFrame({"date": pd.to_datetime(["2010-01-01", "2012-01-01", "2012-01-01", "2020-01-01"])})
        expected = recency_date_weights(sample, pd.Timestamp("2020-01-01"))
        permutation = [3, 1, 0, 2]
        actual = recency_date_weights(sample.iloc[permutation], pd.Timestamp("2020-01-01"))
        np.testing.assert_allclose(actual, expected[permutation])


if __name__ == "__main__":
    unittest.main()
