"""Periodic-sampling baseline with Hankel-matrix PCA compression.

Pipeline
--------
The sensor sends data every ``L`` steps. Each transmission covers the last
``L`` time steps, stacked into one Hankel row::

    h(n) = [x(n), x(n-1), ..., x(n-L+1)]        shape (N*L,)

The row is projected onto a fixed PCA basis ``B`` (eigenvectors of the
covariance of all Hankel rows, computed once for the whole dataset) and only
the top ``K`` coefficients ``z(n) = (h(n) - mu) @ B[:, :K]`` are sent. The
receiver knows ``B`` and ``mu`` and reconstructs the whole window::

    h_hat(n) = z(n) @ B[:, :K].T + mu

which yields estimates of all ``L`` steps ``x(n), ..., x(n-L+1)``. Consecutive
transmissions use non-overlapping windows, so every time step is
reconstructed exactly once and the sender transmits ``K`` numbers per ``L``
steps instead of ``N*L`` raw values.
"""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


class HankelPCABaseline:
    """Fixed periodic sampling + truncated PCA projection of Hankel rows.

    Parameters
    ----------
    L : int
        Hankel order = sampling period (one transmission every ``L`` steps).
    K : int
        Number of principal components sent per transmission (1 <= K <= N*L).
    standardize : bool
        Z-score each feature before building Hankel rows. Without it, PCA is
        dominated by features with large physical units. The unit of the
        reported metrics is chosen by ``evaluate(..., original_units=...)``.
    chunk_size : int
        Number of Hankel rows processed at once while accumulating the
        covariance, which bounds memory for long series (M ~ 1e6).
    """

    def __init__(self, L, K, standardize=True, chunk_size=100_000):
        if L < 1 or K < 1:
            raise ValueError("L and K must be positive integers")
        self.L = int(L)
        self.K = int(K)
        self.standardize = standardize
        self.chunk_size = int(chunk_size)

    # ------------------------------------------------------------------ fit

    def fit(self, X):
        """Compute the constant basis ``B`` from all Hankel rows of ``X``.

        Uses every overlapping row (M-L+1 of them), not only the transmitted
        ones, so the basis reflects the full temporal statistics of the data.
        """
        X = self._check_X(X)
        M, N = X.shape
        if M < self.L:
            raise ValueError(f"need at least L={self.L} time steps, got {M}")
        if self.K > N * self.L:
            raise ValueError(f"K={self.K} exceeds row dimension N*L={N * self.L}")
        self.n_features_ = N

        # Per-feature scaling (identity when standardize=False).
        self.feature_mean_ = X.mean(axis=0) if self.standardize else np.zeros(N)
        std = X.std(axis=0) if self.standardize else np.ones(N)
        self.feature_std_ = np.where(std > 0, std, 1.0)  # constant features stay as-is
        Xs = self._scale(X)

        # Accumulate sum(h) and sum(h h^T) over all overlapping Hankel rows in
        # chunks, so the (M-L+1, N*L) matrix is never materialised at once.
        D = N * self.L
        windows = sliding_window_view(Xs, self.L, axis=0)  # (M-L+1, N, L), view
        n_rows = windows.shape[0]
        row_sum = np.zeros(D)
        gram = np.zeros((D, D))
        for start in range(0, n_rows, self.chunk_size):
            H = self._windows_to_rows(windows[start:start + self.chunk_size])
            row_sum += H.sum(axis=0)
            gram += H.T @ H

        # Covariance of Hankel rows, then its eigendecomposition. eigh returns
        # ascending eigenvalues for a symmetric matrix; flip to descending.
        self.mean_ = row_sum / n_rows
        cov = (gram - n_rows * np.outer(self.mean_, self.mean_)) / (n_rows - 1)
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = np.argsort(eigvals)[::-1]
        self.eigenvalues_ = np.clip(eigvals[order], 0.0, None)  # remove -1e-17 noise
        self.B_ = eigvecs[:, order]  # columns = eigenvectors, shape (N*L, N*L)
        return self

    # ------------------------------------------------- sender / receiver

    def transform(self, X, K=None):
        """Sender side: one truncated projection per non-overlapping window.

        Returns ``Z`` of shape ``(M // L, K)``. Trailing ``M % L`` steps that
        do not fill a full window are not transmitted.
        """
        K = self._resolve_K(K)
        H = self._periodic_rows(self._scale(self._check_X(X)))
        return (H - self.mean_) @ self.B_[:, :K]

    def reconstruct(self, Z):
        """Receiver side: map ``Z`` back to time steps, shape ``(rows*L, N)``.

        Each reconstructed row ``[x(n), ..., x(n-L+1)]`` is reversed back into
        chronological order, so the output aligns with ``X[:rows*L]``.
        """
        K = Z.shape[1]
        H_hat = Z @ self.B_[:, :K].T + self.mean_
        X_hat = H_hat.reshape(-1, self.L, self.n_features_)[:, ::-1, :]
        return X_hat.reshape(-1, self.n_features_)  # still in scaled space

    def transform_reconstruct(self, X, K=None, original_units=True):
        """Run sender + receiver and return ``(X_true, X_hat)``, aligned.

        Both arrays have shape ``((M // L) * L, N)``. With
        ``original_units=False`` both are returned in the standardized space.
        """
        X = self._check_X(X)
        X_hat = self.reconstruct(self.transform(X, K))
        X_true = X[: X_hat.shape[0]]
        if original_units:
            return X_true, self._unscale(X_hat)
        return self._scale(X_true), X_hat

    # ----------------------------------------------------------- metrics

    def evaluate(self, X, K=None, worst_fraction=0.05, original_units=False):
        """Reconstruction quality on ``X``.

        A *sample* is one time step ``x(n)`` (all N features). Its error is
        ``e(n) = mean_f (x_f(n) - x_hat_f(n))^2``. The worst-tail metric
        averages ``e(n)`` over the ``worst_fraction`` of steps with the largest
        ``e(n)``.

        ``original_units=False`` (default) measures error on z-scored
        features, so every feature contributes equally regardless of its
        physical unit. Set it to True for errors in raw units.
        """
        K = self._resolve_K(K)
        X_true, X_hat = self.transform_reconstruct(X, K, original_units)
        sample_mse = np.mean((X_true - X_hat) ** 2, axis=1)  # e(n), shape (T,)

        n_worst = max(1, int(np.ceil(worst_fraction * sample_mse.size)))
        # np.partition finds the top n_worst in O(T), no full sort needed.
        worst = np.partition(sample_mse, -n_worst)[-n_worst:]
        mse, worst_mse = sample_mse.mean(), worst.mean()

        D = self.n_features_ * self.L
        return {
            "L": self.L,
            "K": K,
            "rmse": float(np.sqrt(mse)),
            "mse": float(mse),
            "worst_rmse": float(np.sqrt(worst_mse)),
            "worst_mse": float(worst_mse),
            "worst_threshold_mse": float(worst.min()),  # e(n) cut-off of the tail
            "worst_fraction": worst_fraction,
            "n_samples": int(sample_mse.size),
            # numbers sent per time step relative to sending all N raw values
            "compression_ratio": K / D,
            "explained_variance": float(self.eigenvalues_[:K].sum() / self.eigenvalues_.sum()),
        }

    # ----------------------------------------------------------- helpers

    def _windows_to_rows(self, windows):
        """(rows, N, L) ascending-time windows -> (rows, N*L) Hankel rows.

        The window's last time index is the newest step ``x(n)``; reversing it
        and putting time before features gives ``[x(n), x(n-1), ...]`` with
        each ``x(.)`` a contiguous block of N values.
        """
        rows = windows.shape[0]
        return windows[:, :, ::-1].transpose(0, 2, 1).reshape(rows, -1)

    def _periodic_rows(self, Xs):
        """Non-overlapping Hankel rows ending at n = L-1, 2L-1, ..."""
        m = Xs.shape[0] // self.L
        if m == 0:
            raise ValueError(f"need at least L={self.L} time steps")
        blocks = Xs[: m * self.L].reshape(m, self.L, self.n_features_)
        return blocks[:, ::-1, :].reshape(m, -1)

    def _resolve_K(self, K):
        K = self.K if K is None else int(K)
        if not 1 <= K <= self.B_.shape[1]:
            raise ValueError(f"K={K} must be in [1, N*L={self.B_.shape[1]}]")
        return K

    def _check_X(self, X):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError(f"X must have shape (M, N), got {X.shape}")
        if hasattr(self, "n_features_") and X.shape[1] != self.n_features_:
            raise ValueError(f"expected {self.n_features_} features, got {X.shape[1]}")
        return X

    def _scale(self, X):
        return (X - self.feature_mean_) / self.feature_std_

    def _unscale(self, Xs):
        return Xs * self.feature_std_ + self.feature_mean_


def grid_search(X, L_values, K_values, worst_fraction=0.05, X_eval=None,
                standardize=True, original_units=False):
    """Evaluate every (L, K) pair; return a list of metric dicts.

    The basis is fitted once per ``L`` on ``X`` and reused for all ``K``
    (truncating ``B`` is free). Pairs with ``K > N*L`` are skipped. Metrics
    are computed on ``X_eval`` when given, otherwise on ``X``.
    """
    X = np.asarray(X, dtype=np.float64)
    X_eval = X if X_eval is None else X_eval
    results = []
    for L in L_values:
        model = HankelPCABaseline(L, K=1, standardize=standardize).fit(X)
        for K in K_values:
            if K > X.shape[1] * L:
                continue
            results.append(model.evaluate(X_eval, K, worst_fraction, original_units))
    return results


if __name__ == "__main__":
    # Smoke demo on synthetic correlated signals: N=30 features driven by
    # a few latent oscillators plus noise.
    rng = np.random.default_rng(0)
    M, N = 50_000, 30
    t = np.arange(M)[:, None]
    latent = np.hstack([np.sin(2 * np.pi * t / p) for p in (50, 120, 400)])
    X = latent @ rng.normal(size=(3, N)) + 0.1 * rng.normal(size=(M, N))

    rows = grid_search(X, L_values=[1, 5, 10], K_values=[1, 3, 5, 10, 30])
    print(f"{'L':>3} {'K':>3} {'ratio':>7} {'expl.var':>8} {'RMSE':>8} {'worst5%':>8}")
    for r in rows:
        print(f"{r['L']:>3} {r['K']:>3} {r['compression_ratio']:>7.3f} "
              f"{r['explained_variance']:>8.4f} {r['rmse']:>8.4f} {r['worst_rmse']:>8.4f}")
