# PCA Compression Baseline

## 1. Objective

- Build a simple, reproducible compression baseline.
- Input: multivariate time series.
- Method: Hankel embedding → PCA → retain top K components → inverse projection → overlap-add reconstruction.
- Evaluate reconstruction quality, encoded storage cost, and runtime.

## 2. Model configuration

```python
config = {
    "hankel_order": 20,
    "max_components": 100,
    "pca_solver": "auto",
}
```

- `hankel_order`: temporal embedding length L.
- `max_components`: maximum number of retained PCA components to evaluate.
- Fit one maximum-size PCA basis per L configuration and evaluate prefixes of that basis for different K values.
- Validate K against the available training rank.

## 3. Model interface

```python
class CompressionModel:
    def fit(self, trainer):
        ...

    def compress(self, X, **kwargs):
        ...

    def decompress(self, Z, **kwargs):
        ...

    def save(self, model_path):
        ...

    @classmethod
    def load(cls, model_path):
        ...
```

## 4. PCA model

```python
class HankelPCAModel(CompressionModel):
    def __init__(self, config):
        ...

    def fit(self, trainer):
        ...

    def compress(self, X, top_k=None):
        ...

    def decompress(self, Z, top_k=None):
        ...

    def save(self, model_path):
        ...

    @classmethod
    def load(cls, model_path):
        ...
```

The implementation should define whether `compress()` accepts a full time series or a precomputed embedding. Prefer a full-series public interface, with embedding operations implemented as separate testable helpers.

## 5. Training

1. Sample training windows from the training split.
2. Construct Hankel embeddings.
3. Center embeddings using the fitted training mean.
4. Fit PCA.
5. Store the mean and maximum required component basis.
6. Validate candidate values of L and K.

For large datasets, use an appropriate incremental or randomized PCA implementation. Do not fit independent PCA bases on batches and combine them as if they were one shared basis.

## 6. Compression

Given input X:

1. Construct H with shape `(N, L * C)`.
2. Center: `H_centered = H - mean`.
3. Select `P_K` with shape `(L * C, K)`.
4. Project: `Z = H_centered @ P_K`.
5. Return Z and any required side information.

Expected shape:

- H: `(N, L * C)`.
- P_K: `(L * C, K)`.
- Z: `(N, K)`.

## 7. Decompression

1. Reconstruct the embedding: `H_reconstructed = Z @ P_K.T + mean`.
2. Reshape rows into windows of shape `(L, C)`.
3. Map each window back to the original time indices.
4. Average overlapping estimates.
5. Return the reconstructed series with shape `(T, C)`.

## 8. Hyperparameter search

- Search over candidate values of L and K.
- Use FFT or spectral analysis to suggest candidate temporal scales.
- Constrain L by temporal context, training data volume, memory, and compute.
- Evaluate K using both absolute values and fractions of the embedded dimension `D = L * C`.
- Select hyperparameters using validation data.
- Keep the test split untouched until final evaluation.

## 9. Evaluation

- R², MSE, MAPE, and MAE.
- Mean, standard deviation, histograms, and tail quantiles.
- Per-channel and per-dataset results.
- Compression ratio and bits per original scalar.
- Model size, training time, compression throughput, and decompression throughput.

Report both idealized PCA coefficient savings and actual encoded storage. Overlapping Hankel windows can make independent storage of all coefficients larger than the original data.

## 10. Persistence

Save:

- PCA mean.
- PCA component basis.
- Normalization parameters.
- Hankel order and embedding configuration.
- Supported component counts.
- Model format version.

Test that saving and loading the model preserves compression and decompression behavior.
