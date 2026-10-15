import unittest

import numpy as np

from data.scaler import StandardScaler
from data.split import chronological_split


class ChronologicalSplitTest(unittest.TestCase):
    def setUp(self):
        self.X = np.arange(20, dtype=float).reshape(10, 2)

    def test_train_is_the_past_and_test_the_future(self):
        train, test = chronological_split(self.X, train_fraction=0.7)
        np.testing.assert_array_equal(train, self.X[:7])
        np.testing.assert_array_equal(test, self.X[7:])

    def test_split_point_rounds_down(self):
        train, test = chronological_split(self.X, train_fraction=0.75)
        self.assertEqual((len(train), len(test)), (7, 3))

    def test_rejects_fraction_outside_open_unit_interval(self):
        for f in (0.0, 1.0, -0.1, 1.5):
            with self.assertRaises(ValueError):
                chronological_split(self.X, train_fraction=f)

    def test_rejects_empty_train_part(self):
        with self.assertRaises(ValueError):
            chronological_split(self.X, train_fraction=0.05)  # floor(0.5) = 0 rows

    def test_test_part_is_never_empty(self):
        train, test = chronological_split(self.X, train_fraction=0.99)
        self.assertEqual((len(train), len(test)), (9, 1))

    def test_scaler_fitted_on_train_ignores_test(self):
        X = np.r_[np.arange(8.0), 100.0, 100.0][:, None]  # jump in the last 20%
        train, test = chronological_split(X, train_fraction=0.8)
        scaler = StandardScaler().fit(train)
        np.testing.assert_allclose(scaler.mean_, [3.5])
        self.assertGreater(scaler.transform(test).min(), 30.0)


if __name__ == "__main__":
    unittest.main()
