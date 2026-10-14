import unittest

import numpy as np

from baselines.hankel_pca import HankelPCABaseline, grid_search


def naive_hankel(X, L):
    """Reference: explicit loop building [x(n), x(n-1), ..., x(n-L+1)]."""
    return np.array([np.concatenate([X[n - j] for j in range(L)])
                     for n in range(L - 1, len(X))])


class HankelPCABaselineTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(1)
        self.X = rng.normal(size=(503, 4)) * [1, 10, 100, 0.1] + [0, 5, -3, 1]

    def test_basis_matches_naive_pca_of_full_hankel_matrix(self):
        L = 3
        model = HankelPCABaseline(L, K=2, chunk_size=50).fit(self.X)
        Xs = (self.X - self.X.mean(0)) / self.X.std(0)
        cov = np.cov(naive_hankel(Xs, L), rowvar=False)
        expected = np.sort(np.linalg.eigvalsh(cov))[::-1]
        np.testing.assert_allclose(model.eigenvalues_, expected, atol=1e-10)

    def test_periodic_rows_are_non_overlapping_hankel_rows(self):
        L = 4
        model = HankelPCABaseline(L, K=1).fit(self.X)
        rows = model._periodic_rows(self.X)
        np.testing.assert_allclose(rows, naive_hankel(self.X, L)[::L][: len(rows)])

    def test_full_rank_reconstruction_is_lossless(self):
        L = 5
        model = HankelPCABaseline(L, K=4 * L).fit(self.X)
        X_true, X_hat = model.transform_reconstruct(self.X)
        self.assertEqual(X_true.shape, ((503 // L) * L, 4))
        np.testing.assert_allclose(X_hat, X_true, atol=1e-8)
        self.assertLess(model.evaluate(self.X)["rmse"], 1e-9)

    def test_error_decreases_with_more_components(self):
        model = HankelPCABaseline(3, K=1).fit(self.X)
        rmse = [model.evaluate(self.X, K)["rmse"] for K in (1, 4, 8, 12)]
        self.assertEqual(rmse, sorted(rmse, reverse=True))

    def test_worst_metric_is_mean_of_top_fraction(self):
        model = HankelPCABaseline(2, K=2).fit(self.X)
        X_true, X_hat = model.transform_reconstruct(self.X, original_units=False)
        e = np.mean((X_true - X_hat) ** 2, axis=1)
        top = np.sort(e)[::-1][: int(np.ceil(0.05 * e.size))]
        r = model.evaluate(self.X)
        self.assertAlmostEqual(r["worst_mse"], top.mean())
        self.assertGreaterEqual(r["worst_rmse"], r["rmse"])

    def test_grid_search_skips_infeasible_K(self):
        res = grid_search(self.X, L_values=[1, 2], K_values=[1, 4, 8])
        self.assertEqual([(r["L"], r["K"]) for r in res], [(1, 1), (1, 4), (2, 1), (2, 4), (2, 8)])


if __name__ == "__main__":
    unittest.main()
