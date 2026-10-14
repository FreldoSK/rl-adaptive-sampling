"""Run the Hankel-PCA baseline grid search on a SOLETE HDF5 file.

Usage:
    python -m experiments.grid_search_solete PATH_TO.h5 [--out results.csv]
"""

import argparse
import time
from pathlib import Path

import pandas as pd

from baselines.hankel_pca import grid_search

L_VALUES = [1, 2, 5, 10, 20, 30, 60]
# Fractions of the full row dimension N*L, so every L gets comparable K levels.
K_FRACTIONS = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5]
K_ABSOLUTE = [1, 2, 3, 5, 8, 11]


def k_values(n_features, L):
    D = n_features * L
    ks = set(K_ABSOLUTE) | {max(1, round(f * D)) for f in K_FRACTIONS}
    return sorted(k for k in ks if k <= D)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("h5_path")
    parser.add_argument("--key", default="DATA")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    df = pd.read_hdf(args.h5_path, key=args.key).sort_index()
    X = df.to_numpy(dtype=float)
    print(f"{Path(args.h5_path).name}: {X.shape[0]} steps x {X.shape[1]} features")

    rows = []
    for L in L_VALUES:
        t0 = time.time()
        rows += grid_search(X, [L], k_values(X.shape[1], L))
        print(f"  L={L:>3} done in {time.time() - t0:.1f}s")

    res = pd.DataFrame(rows)
    out = args.out or f"results/hankel_pca_{Path(args.h5_path).stem}.csv"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(out, index=False)
    print(f"saved {len(res)} rows to {out}")


if __name__ == "__main__":
    main()
