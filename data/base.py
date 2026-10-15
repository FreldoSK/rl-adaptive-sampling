"""Common interface in front of every dataset.

A dataset is a multivariate time series of M time steps x N features, in
physical units (no normalization). Subclasses only implement ``_load``,
which returns a DataFrame indexed by time with one column per feature; the
rest of the pipeline reads ``values`` / ``ds[idx]`` and never sees how the
dataset is stored.
"""

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class Dataset(ABC):
    """Subclasses store whatever ``_load`` needs, then call ``super().__init__()``."""

    def __init__(self):
        frame = self._load()
        if not isinstance(frame.index, pd.DatetimeIndex):
            raise TypeError(f"{type(self).__name__}._load must return a DatetimeIndex")
        if not frame.index.is_monotonic_increasing:
            raise ValueError(f"{type(self).__name__}._load must return rows sorted by time")
        self._features = list(frame.columns)
        self._timestamps = frame.index
        self._values = frame.to_numpy(dtype=np.float64, copy=True)
        self._values.flags.writeable = False

    @abstractmethod
    def _load(self):
        """Return the ``(M, N)`` data as a time-indexed DataFrame, physical units."""

    @property
    def features(self):
        return list(self._features)

    @property
    def timestamps(self):
        return self._timestamps

    @property
    def values(self):
        """The full ``(M, N)`` float64 array (read-only)."""
        return self._values

    def __len__(self):
        return self._values.shape[0]

    def __getitem__(self, idx):
        """Time step ``idx`` as ``(N,)``, or several steps as ``(k, N)``."""
        return self._values[idx, :]
