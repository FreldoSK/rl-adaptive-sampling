"""Per-feature z-score normalization, as a separate step after the loader.

Datasets stay in physical units; the scaler maps an ``(M, N)`` array to
zero mean / unit std per feature and back. Fit it on the training part only
(see ``data.split.chronological_split``): fitting on the whole series leaks
test statistics into training.
"""

import numpy as np


class StandardScaler:
    """``Z = (X - mean_) / std_`` with ``mean_`` and ``std_`` of shape ``(N,)``.

    A constant feature (std = 0) gets ``std_ = 1``, so it is only centred and
    never divided by zero.
    """

    def fit(self, X):
        """Learn per-feature ``mean_`` and ``std_`` from ``X`` (the training part)."""
        X = self._check_X(X)
        self.mean_ = X.mean(axis=0)
        std = X.std(axis=0)
        self.std_ = np.where(std > 0, std, 1.0)
        return self

    def transform(self, X):
        """Physical units -> z-scores, shape ``(M, N)``."""
        return (self._check_fitted_X(X) - self.mean_) / self.std_

    def inverse_transform(self, Z):
        """Z-scores -> physical units, shape ``(M, N)``."""
        return self._check_fitted_X(Z) * self.std_ + self.mean_

    @staticmethod
    def _check_X(X):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError(f"expected a 2-D (M, N) array, got shape {X.shape}")
        return X

    def _check_fitted_X(self, X):
        if not hasattr(self, "mean_"):
            raise RuntimeError("StandardScaler is not fitted; call fit(X_train) first")
        X = self._check_X(X)
        if X.shape[1] != self.mean_.shape[0]:
            raise ValueError(f"fitted on {self.mean_.shape[0]} features, got {X.shape[1]}")
        return X
