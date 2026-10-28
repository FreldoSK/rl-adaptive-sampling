# Metrics

## 1. Reconstruction metrics

- **R²:** coefficient of determination; higher is better. Can be negative.
- **MSE:** mean squared error; lower is better.
- **MAPE:** mean absolute percentage error; lower is better, but unstable around zero.
- **MAE:** mean absolute error; lower is better and relatively interpretable in original units.

Calculate metrics per window and channel where practical, then aggregate across datasets.

Define how constant-valued ground truth is handled for R² and how zero or near-zero values are handled for MAPE.

## 2. Distribution reporting

For each metric, report:

- Mean and standard deviation.
- Median and selected quantiles.
- Histogram or empirical distribution.
- Per-channel and per-dataset results.
- Number of evaluated windows.

For R², lower-tail quantiles describe poor reconstructions. For error metrics, upper-tail quantiles describe poor reconstructions.

If reporting the worst 68%, 95%, and 99.7% of windows, define the statistic explicitly. For example, use the 32nd, 5th, and 0.15th percentiles of R², and corresponding upper-tail percentiles for error metrics.

## 3. Compression metrics

- Compression ratio = original encoded bytes / compressed encoded bytes.
- Bits per original scalar = compressed bits / number of original scalar values.
- Model storage size.
- Report idealized coefficient-count ratios separately from actual encoded storage.

## 4. Computational metrics

- Training time.
- Compression throughput.
- Decompression throughput.
- Peak memory usage.
- Model size.

## 5. Relative metrics

- Absolute difference from the reference model.
- Relative MSE reduction.
- Relative R² change.
- Compression ratio improvement.
- Reconstruction quality at a fixed storage budget.
- Storage cost at a fixed quality target.

Define policies for zero or near-zero reference values.

## 6. Python interfaces

```python
class ReconstructionMetrics:
    def clear(self):
        ...

    def add(self, X_ground_truth, X_prediction):
        ...

    def eval(self) -> dict:
        ...


class CompressionMetrics:
    def clear(self):
        ...

    def add(self, original_size_bytes,
             compressed_size_bytes, num_values):
        ...

    def eval(self) -> dict:
        ...


class ModelEvaluator:
    def clear(self):
        ...

    def add(self, X_ground_truth, X_prediction,
            compression_info=None, metadata=None):
        ...

    def eval(self) -> dict:
        ...
```

- `ReconstructionMetrics` computes reconstruction statistics.
- `CompressionMetrics` measures encoded storage efficiency.
- `ModelEvaluator` coordinates evaluation and aggregates results.
- All returned dictionaries should be JSON-compatible.
- Use streaming statistics when possible; retain per-window scores or use approximate quantile algorithms when distribution reporting requires them.
- Evaluate in normalized space and, where possible, in original physical units.
