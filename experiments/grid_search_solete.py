"""Run the Hankel-PCA baseline grid search on the SOLETE v4 dataset.

Usage:
    python -m experiments.grid_search_solete [--resolution 1min] [--path FILE.h5] [--out results.csv]

The file is downloaded into data/raw/ on first use (see data/solete.py).
"""

import argparse
import time
from pathlib import Path

import pandas as pd

from baselines.hankel_pca import grid_search
from data.solete import SHA256, load_solete

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
    parser.add_argument("--resolution", default="1min", choices=list(SHA256))
    parser.add_argument("--path", default=None, help="already downloaded v4 .h5 file")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    df, interpolated = load_solete(args.resolution, args.path)
    X = df.to_numpy(dtype=float)
    print(f"SOLETE v4 {args.resolution}: {X.shape[0]} steps x {X.shape[1]} features")
    for col, n in interpolated.items():
        print(f"  interpolated {n:>5} values in {col}")

    rows = []
    for L in L_VALUES:
        t0 = time.time()
        rows += grid_search(X, [L], k_values(X.shape[1], L))
        print(f"  L={L:>3} done in {time.time() - t0:.1f}s")

    res = pd.DataFrame(rows)
    out = args.out or f"results/hankel_pca_SOLETE_{args.resolution}_v4.csv"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(out, index=False)
    print(f"saved {len(res)} rows to {out}")


if __name__ == "__main__":
    main()
