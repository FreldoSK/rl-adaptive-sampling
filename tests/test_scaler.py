import unittest

import numpy as np

from data.scaler import StandardScaler


class StandardScalerTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.X = rng.normal(size=(200, 3)) * [1, 10, 100] + [0, 5, -3]

    def test_transform_gives_zero_mean_unit_std_per_feature(self):
        Z = StandardScaler().fit(self.X).transform(self.X)
        np.testing.assert_allclose(Z.mean(axis=0), 0.0, atol=1e-12)
        np.testing.assert_allclose(Z.std(axis=0), 1.0, atol=1e-12)

    def test_inverse_transform_round_trip(self):
        scaler = StandardScaler().fit(self.X)
        np.testing.assert_allclose(scaler.inverse_transform(scaler.transform(self.X)), self.X,
                                   rtol=1e-12, atol=1e-12)

    def test_constant_feature_is_left_unscaled(self):
        X = self.X.copy()
        X[:, 1] = 7.0
        scaler = StandardScaler().fit(X)
        Z = scaler.transform(X)
        self.assertEqual(scaler.std_[1], 1.0)
        self.assertTrue(np.all(np.isfinite(Z)))
        np.testing.assert_array_equal(Z[:, 1], 0.0)  # centred, not divided
        np.testing.assert_allclose(scaler.inverse_transform(Z), X, atol=1e-12)

    def test_statistics_come_from_fit_data_only(self):
        train, test = self.X[:150], self.X[150:] + 1000.0
        scaler = StandardScaler().fit(train)
        np.testing.assert_allclose(scaler.mean_, train.mean(axis=0))
        np.testing.assert_allclose(scaler.std_, train.std(axis=0))
        np.testing.assert_allclose(scaler.transform(test), (test - train.mean(axis=0)) / train.std(axis=0))

    def test_transform_before_fit_raises(self):
        with self.assertRaises(RuntimeError):
            StandardScaler().transform(self.X)

    def test_rejects_wrong_number_of_features(self):
        scaler = StandardScaler().fit(self.X)
        with self.assertRaises(ValueError):
            scaler.transform(self.X[:, :2])

    def test_rejects_non_2d_input(self):
        with self.assertRaises(ValueError):
            StandardScaler().fit(self.X[:, 0])

    def test_does_not_modify_input(self):
        X = self.X.copy()
        scaler = StandardScaler().fit(X)
        scaler.inverse_transform(scaler.transform(X))
        np.testing.assert_array_equal(X, self.X)


if __name__ == "__main__":
    unittest.main()
