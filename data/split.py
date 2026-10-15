"""Chronological train/test split of a time series.

The first ``train_fraction`` of the time steps is the training part and the
rest the test part, with no shuffling: the model only ever sees the past,
and Hankel rows inside each part stay made of consecutive steps.
"""


def chronological_split(X, train_fraction=0.8):
    """Split ``X`` along its first (time) axis into ``(train, test)``.

    The split index is ``floor(train_fraction * len(X))``. Both parts must
    be non-empty; the test part always is, since ``train_fraction < 1``.
    ``X`` can be an array or anything sliceable with a length (e.g. a
    ``Dataset``); for an array, the parts are views, not copies.
    """
    if not 0.0 < train_fraction < 1.0:
        raise ValueError(f"train_fraction must be in (0, 1), got {train_fraction}")
    n_train = int(train_fraction * len(X))
    if n_train == 0:
        raise ValueError(f"train_fraction={train_fraction} leaves no training steps "
                         f"in a series of {len(X)}")
    return X[:n_train], X[n_train:]
