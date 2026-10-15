"""Validate results/hankel_pca_SOLETE_<res>_v4.csv.

1. Internal consistency of every row (identities, bounds, monotonicity).
2. Independent re-implementation (explicit lag stacking + eigh, no code
   shared with baselines/hankel_pca.py) recomputes (L, K) points from the
   raw data and compares them to the CSV: every L, or L in {1, 10, 60} for
   series longer than 1M steps.

Usage:
    python -m experiments.validate_results [RES ...]   (default: all resolutions)
"""

import sys

import numpy as np
import pandas as pd

from data.solete import FEATURES, SHA256, load_solete

N = len(FEATURES)
fails = []


def check(cond, msg):
    if not cond:
        fails.append(msg)


def independent(X, L, Ks):
    """Reference implementation written from the definition."""
    M = X.shape[0]
    Xs = (X - X.mean(0)) / X.std(0)
    # all overlapping rows h(n) = [x(n), x(n-1), ..., x(n-L+1)], n = L-1..M-1,
    # in blocks of n so the full matrix never exists (1 s data: ~170 GB)
    D, rows, step = N * L, M - L + 1, 200_000
    s, G = np.zeros(D), np.zeros((D, D))
    for a in range(L - 1, M, step):
        n = np.arange(a, min(a + step, M))
        H = np.hstack([Xs[n - l] for l in range(L)])
        s += H.sum(0)
        G += H.T @ H
    mu = s / rows
    cov = (G - rows * np.outer(mu, mu)) / (rows - 1)
    w, V = np.linalg.eigh(cov)
    idx = np.argsort(w)[::-1]
    w, V = np.clip(w[idx], 0, None), V[:, idx]
    # transmitted rows end at n = L-1, 2L-1, ...
    ends = np.arange(L - 1, (M // L) * L, L)
    out = {}
    for K in Ks:
        parts = []
        for a in range(0, len(ends), step):
            n = ends[a:a + step]
            P = np.hstack([Xs[n - l] for l in range(L)])
            P_hat = (P - mu) @ V[:, :K] @ V[:, :K].T + mu
            parts.append(((P - P_hat) ** 2).reshape(len(n), L, N).mean(2).ravel())
        e = np.concatenate(parts)  # per-time-step error
        k = max(1, int(np.ceil(0.05 * e.size)))
        worst = np.sort(e)[-k:]
        out[K] = dict(rmse=np.sqrt(e.mean()), worst_rmse=np.sqrt(worst.mean()),
                      explained_variance=w[:K].sum() / w.sum(), n_samples=e.size)
    return out


for res in sys.argv[1:] or list(SHA256):
    f = f"results/hankel_pca_SOLETE_{res}_v4.csv"
    r = pd.read_csv(f)
    df, _ = load_solete(res)
    X = df.to_numpy(float)
    M = X.shape[0]
    print(f"\n== {res}: {len(r)} rows, M={M}")
    n0 = len(fails)

    check(not r.isna().any().any(), f"{res}: NaN in CSV")
    check(np.isfinite(X).all(), f"{res}: non-finite values in data")
    for _, row in r.iterrows():
        L, K = int(row.L), int(row.K)
        tag = f"{res} L={L} K={K}"
        check(1 <= K <= N * L, f"{tag}: K out of range")
        check(np.isclose(row.compression_ratio, K / (N * L)), f"{tag}: compression_ratio")
        check(row.n_samples == (M // L) * L, f"{tag}: n_samples {row.n_samples}")
        check(np.isclose(row.rmse ** 2, row.mse), f"{tag}: rmse^2 != mse")
        check(np.isclose(row.worst_rmse ** 2, row.worst_mse), f"{tag}: worst_rmse^2 != worst_mse")
        check(row.worst_mse >= row.mse - 1e-12, f"{tag}: worst < mean")
        check(row.worst_threshold_mse <= row.worst_mse + 1e-12, f"{tag}: threshold > worst")
        check(0 <= row.explained_variance <= 1 + 1e-12, f"{tag}: EV out of [0,1]")
        # on standardized data, per-element MSE ~ unexplained variance share
        # (only a statistical identity, so skip it when there are few windows)
        check(M // L < 1000 or abs(row.mse - (1 - row.explained_variance)) < 0.02,
              f"{tag}: mse {row.mse:.4f} vs 1-EV {1 - row.explained_variance:.4f}")
        if K == N * L:
            check(row.rmse < 1e-6, f"{tag}: full basis but rmse={row.rmse:.2e}")
    for L, g in r.groupby("L"):
        g = g.sort_values("K")
        check(g.rmse.is_monotonic_decreasing, f"{res} L={L}: rmse not decreasing in K")
        check(g.explained_variance.is_monotonic_increasing, f"{res} L={L}: EV not increasing in K")
    print(f"   consistency checks: {len(fails) - n0} failures")

    # independent recomputation (all points for small data, a sample otherwise)
    Ls = sorted(r.L.unique()) if M < 1_000_000 else [1, 10, 60]
    max_err = 0.0
    for L in Ls:
        g = r[r.L == L]
        ref = independent(X, L, g.K.astype(int).tolist())
        for _, row in g.iterrows():
            o = ref[int(row.K)]
            check(o["n_samples"] == row.n_samples, f"{res} L={L} K={row.K}: ref n_samples")
            for key in ("rmse", "worst_rmse", "explained_variance"):
                d = abs(o[key] - row[key])
                max_err = max(max_err, d)
                check(d < 1e-6, f"{res} L={L} K={row.K}: {key} csv={row[key]:.8f} ref={o[key]:.8f}")
    print(f"   independent recompute L={Ls}: max abs diff {max_err:.2e}")
    print(r[r.K.isin([1, 3]) & r.L.isin([1, 10, 60])][["L", "K", "compression_ratio", "rmse", "worst_rmse", "explained_variance"]].to_string(index=False))

print(f"\nTOTAL FAILURES: {len(fails)}")
for m in fails[:40]:
    print("  ", m)
sys.exit(1 if fails else 0)
