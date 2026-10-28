# Multimodel time-series compression — revised design

The goal is to keep the baseline simple while establishing a framework that also supports self-supervised learning and reinforcement learning later.

I'll preserve your numbered, bullet-point structure and focus on correcting the mathematics, defining every shape, improving class names and interfaces, and filling gaps in the metrics. This is a design revision only; no files are being modified.

# Task

- Multimodel time-series data compression.
- Baseline: Hankel embedding → PCA → retain the top \\(K\\) principal components → reconstruct the original time series.
- Future models:
  - Self-supervised learning.
  - Reinforcement learning.
  - Other learned compression methods.
- All models must share the same training, sampling, evaluation, compression, and decompression framework.

Terminology: use top \\(K\\) principal components rather than top \\(K\\) columns. The columns are just how the components are represented in our chosen matrix convention.

# 1. Dataset structure

- Bronze: original downloaded data, immutable where possible, with original formats and metadata.
- Silver: cleaned, validated, normalized data with consistent channel ordering and an explicit missing-value policy.
- Gold: deployment-ready artifacts and production outputs. Not required for the initial experiments.

## Convert all datasets into a common format

- Small datasets:
  - NumPy arrays (`.npy`).
  - JSON metadata and schema.
- Large datasets:
  - Arrow/Parquet for columnar storage.
  - Polars for efficient data processing.
  - NumPy memory mapping when appropriate for large array-oriented workloads.
- Each dataset must define:
  - Number of time steps \\(T\\).
  - Number of channels \\(C\\).
  - Channel names and units.
  - Sampling interval or timestamps.
  - Data type and shape.
  - Normalization parameters.
  - Dataset identifier and version.
  - Train/validation/test split references.
- Train/validation/test splits:
  - Define the split in the data pipeline.
  - Store split assignments in a manifest.
  - Split before generating overlapping windows.
  - Fit normalization parameters on training data only.
  - Never use validation or test data to fit PCA or other learned parameters.
- All datasets must satisfy a documented data contract:
  - Shape: `(num_samples, num_channels)`.
  - Fixed channel order within each dataset.
  - No unresolved NaN or infinite values.
  - Explicit handling of constant-valued channels.
  - Consistent timestamp and sampling semantics.

# 2. Dataset interface

- One common dataset interface.
- Responsibilities:
  - Load data and metadata.
  - Provide indexed access to rows and windows.
  - Expose channel names and dimensions.
  - Apply the configured preprocessing pipeline.
  - Support efficient iteration and sampling without requiring all data to fit in RAM.

Suggested interface:

```
class TimeSeriesDataset:    def __len__(self) -> int: ...    def get_row(self, row_idx: int): ...    def get_window(self, row_idx: int, window_size: int): ...    @property    def column_names(self): ...    @property    def num_channels(self) -> int: ...
```

- `get_row(row_idx)`:
  - Returns shape `(C,)`, where \\(C\\) is the number of channels.
- `get_window(row_idx, window_size)`:
  - Returns shape `(W, C)`, where \\(W\\) is the requested number of time steps.
- Define `row_idx` as the starting index of a window.
- Reject windows that exceed the dataset boundary, unless an explicit padding policy is configured.
- The dataset returns values according to a configured preprocessing policy. For this experiment, normalized values are the default.
- Preserve access to original units and normalization metadata for evaluation and decompression.

# 3. Data augmentation

- One composable augmentation interface.
- Base class: `IdentityAugmentation`.
- Each augmentation wraps a dataset or another augmentation.
- Augmentations can be chained:

```
dataset = TimeSeriesDataset(...)a = AugmentationA(dataset)b = AugmentationB(a)c = AugmentationC(b)
```

- Requirements:
  - All augmentations expose a compatible dataset interface.
  - Random augmentations support explicit seeds or random generators.
  - Channel count and channel order remain consistent unless an explicit transformation changes them.
  - Augmentation order is part of the experiment configuration.
  - Start with identity augmentation for the PCA baseline.
- Important distinction:
  - Augmentation transforms samples.
  - Sampling selects which samples or windows the trainer receives.
  - These must be separate abstractions. Self-supervised learning may need multiple correlated views of a window; reinforcement learning may need sequential transitions rather than independent windows.

# 4. Sampling and training framework

- The framework must support different training algorithms without requiring every model to use the same optimization procedure.
- The trainer owns the training loop and controls what data the model receives.
- The sampler owns the selection and delivery of training samples.
- The model owns its representation, compression, decompression, and learned parameters.

## 4.1 Sampling interface

```
class SampleProvider:    def sample(self, batch_size: int, **kwargs):        ...class WindowSampler(SampleProvider):    ...class SequenceSampler(SampleProvider):    ...
```

- `WindowSampler`:
  - Samples windows from one or more time-series datasets.
  - Useful for PCA, self-supervised learning, and many neural models.
- `SequenceSampler`:
  - Provides temporally ordered sequences.
  - Useful for recurrent models, predictive objectives, and reinforcement learning.
- Future samplers can provide pairs of views, positive/negative pairs, or state/action/reward transitions.
- Sampling must respect recording boundaries and data splits.

## 4.2 Trainer interface

```
class Trainer:    def __init__(self, model, sampler, config):        ...    def fit(self, num_steps: int):        ...    def validate(self, evaluator):        ...
```

- `num_steps` means the number of training iterations, not necessarily epochs.
- Each iteration requests samples from the sampler and passes them to the model's training procedure.
- The model may implement:
  - `train_step(batch)` for gradient-based training.
  - `fit_from_samples(...)` for algorithms such as PCA.
  - A specialized update method for reinforcement learning.
- The trainer tracks progress, random seeds, checkpoints, and experiment metadata.

Important: do not force PCA, self-supervised learning, and reinforcement learning into one identical optimization algorithm. Standardize the lifecycle and interfaces, while allowing each trainer/model combination to define its own learning procedure.

For the initial PCA baseline, the trainer can collect or stream the training windows and fit PCA. It does not need a neural-network-style gradient update.

# 5. Metrics

Your existing metrics are a good foundation. Keep them, but distinguish reconstruction quality from compression efficiency and computational cost.

## 5.1 Reconstruction metrics

- R² coefficient
  - Measures how much variation is explained by the reconstruction.
  - Higher is better.
  - Can be negative.
  - Define how scores are aggregated across channels and windows.
- MSE
  - Mean squared reconstruction error.
  - Lower is better.
  - Calculate in normalized space and, where possible, original physical units.
- MAPE
  - Mean absolute percentage error.
  - Lower is better.
  - Undefined or unstable when ground-truth values are zero or close to zero.
  - Define an explicit policy for such values; do not silently discard them.
- MAE — recommended addition
  - Mean absolute error.
  - Lower is better.
  - Easier to interpret in original units and less sensitive to large errors than MSE.

## 5.2 Distribution reporting

- Every metric should support:
  - Mean.
  - Standard deviation.
  - Histogram or empirical distribution.
  - Median and selected quantiles.
  - Per-channel and per-dataset breakdowns.
  - Number of evaluated windows.
- For a single summary value:
  - Mean R².
  - Lower-tail R² quantiles to describe poor reconstructions.
  - Upper-tail error quantiles for MSE, MAE, and MAPE.
- If your intention is to report the worst 68%, 95%, and 99.7% of windows, define that precisely. For example, the 32nd, 5th, and 0.15th percentiles of R² are increasingly conservative lower-tail summaries. For error metrics, use the corresponding upper-tail percentiles.
- For relative metrics:
  - Compare against the baseline.
  - Report absolute differences as well as relative changes.
  - Handle zero or near-zero denominators explicitly.

## 5.3 Compression metrics — required additions

\- Compression ratio:

\\[ \text{CR} = \frac{\text{original data size in bytes}} {\text{compressed representation size in bytes}} \\]

\- Bits per original scalar:

\\[ \text{BPS} = \frac{8 \times \text{compressed size in bytes}} {T C} \\]

- Include:
  - Encoded coefficient values.
  - PCA basis and mean, when measuring standalone storage.
  - Quantization parameters.
  - Metadata and other required side information.
- Report both:
  - Idealized coefficient-count compression.
  - Actual storage compression.

## 5.4 Computational metrics — recommended additions

- Training time.
- Compression throughput.
- Decompression throughput.
- Peak memory usage.
- Model storage size.

## 5.5 Metric interfaces

```
class ReconstructionMetrics:    def clear(self): ...    def add(self, X_ground_truth, X_prediction): ...    def eval(self) -> dict: ...class CompressionMetrics:    def clear(self): ...    def add(self, original_size, compressed_size, num_values): ...    def eval(self) -> dict: ...class ModelEvaluator:    def clear(self): ...    def add(self, X_ground_truth, X_prediction, **metadata): ...    def eval(self) -> dict: ...
```

- `ReconstructionMetrics` calculates R², MSE, MAPE, MAE, and their distributions.
- `CompressionMetrics` calculates storage efficiency and compression ratios.
- `ModelEvaluator` combines results and groups them by dataset, channel, configuration, and other relevant metadata.
- Use dictionaries containing ordinary JSON-compatible values for reports.

# 6. Baseline model: Hankel embedding → PCA → top K components

## 6.1 Input time series

\- Input window:

\\[ X\in\mathbb{R}^{T\times C} \\]

where:

- \\(T\\): number of time steps in the input window.
- \\(C\\): number of channels.
- \\(X\_{t,c}\\): value of channel \\(c\\) at time step \\(t\\).

Example:

```
X.shape == (1000, 10)
```

This means 1,000 time steps and 10 channels.

## 6.2 Hankel embedding

- Configuration parameter: `hankel_order` or `window_length`, denoted by \\(L\\).
- Create one flattened observation for every sliding window of \\(L\\) time steps.

\\[ N=T-L+1 \\]

\\[ D=L C \\]

\\[ \boxed{H\in\mathbb{R}^{N\times D}} \\]

- \\(N\\): number of sliding windows.
- \\(D\\): number of features in each flattened window.

The embedding is:

\\[ H\_{i,:} = [x_i,x\_{i+1},\ldots,x\_{i+L-1}] \\]

where each \\(x_i\in\mathbb{R}^{C}\\).

- The rows of \\(H\\) are overlapping temporal windows.
- Each row contains \\(L C\\) values.
- Do not construct windows across separate recordings or dataset boundaries.

Example:

```
T = 1000C = 10L = 20N = T - L + 1  # 981D = L * C      # 200H.shape == (981, 200)
```

The name `Hankel embedding` is preferable to `Hankel matrix construction` in the model interface because the operation is specifically a sliding-window transformation. Strictly speaking, a multivariate embedding matrix formed this way has Hankel-like overlapping structure.

## 6.3 PCA fitting

- Fit PCA using training embeddings only.
- Compute the training mean:

\\[ \mu\in\mathbb{R}^{D} \\]

\- Center the training embeddings:

\\[ H_c=H-\mu \\]

Here, \\(\mu\\) is broadcast across all \\(N\\) rows.

\- Compute the singular value decomposition:

\\[ H_c=U\Sigma V^\top \\]

Shapes:

\\[ U\in\mathbb{R}^{N\times r} \\]

\\[ \Sigma\in\mathbb{R}^{r\times r} \\]

\\[ V^\top\in\mathbb{R}^{r\times D} \\]

where \\(r=\min(N,D)\\) for a compact SVD, with additional rank restrictions possible for degenerate data.

- PCA components are the right singular vectors.
- Define the component matrix:

\\[ P_K=V\_{[:,1:K]} \\]

Using one-based mathematical indexing:

\\[ \boxed{P_K\in\mathbb{R}^{D\times K}} \\]

- The \\(K\\) columns of \\(P_K\\) are the retained principal components, ordered by decreasing explained variance.
- Store the mean, components, and any normalization information needed to reproduce the transformation.

For centered PCA, the maximum number of nonzero components is bounded by \\(\min(N\_{\text{train}}-1,D)\\). The implementation should validate the requested \\(K\\).

## 6.4 Compression: project onto top K components

For an input embedding matrix:

\\[ H\in\mathbb{R}^{N\times D} \\]

\- Center it using the fitted training mean:

\\[ H_c=H-\mu \\]

\- Project onto the first \\(K\\) components:

\\[ \boxed{Z_K=H_cP_K} \\]

Shapes:

\\[ (N\times D)(D\times K)=(N\times K) \\]

Therefore:

\\[ \boxed{Z_K\in\mathbb{R}^{N\times K}} \\]

- \\(Z_K\\) is the compressed coefficient matrix.
- `top_K` means the number of retained principal components.
- `K` is a configurable compression parameter.
- Use the fitted PCA basis; do not refit PCA during compression or validation.

Example:

```
H.shape   == (981, 200)P_K.shape == (200, 20)Z_K = (H - mean) @ P_KZ_K.shape == (981, 20)
```

In this example, each embedded window is represented by 20 coefficients rather than 200 values.

## 6.5 Decompression: reconstruct the embedding

\- Use the same component matrix and training mean:

\\[ \boxed{\hat H=Z_KP_K^\top+\mu} \\]

Shapes:

\\[ (N\times K)(K\times D)=(N\times D) \\]

Therefore:

\\[ \hat H\in\mathbb{R}^{N\times D} \\]

- This reconstructs the flattened sliding windows.
- It does not yet reconstruct the original time series.
- PCA reconstruction is generally approximate when \\(K\<D\\).

Example:

```
Z_K.shape == (981, 20)P_K.T.shape == (20, 200)H_reconstructed = Z_K @ P_K.T + meanH_reconstructed.shape == (981, 200)
```

## 6.6 Reconstruct the original time series

\- Reshape each reconstructed row into a window:

\\[ \hat H_i\rightarrow \hat X_i\in\mathbb{R}^{L\times C} \\]

- Place each window back at its original time indices.
- Average all estimates for each original time step.

For original time index \\(t\\), define the set of windows that cover it as \\(I_t\\). Then:

\\[ \boxed{ \hat X_t= \frac{1}{|I_t|} \sum\_{i\in I_t} \hat X_i[t-i] } \\]

where the indexing \\(t-i\\) identifies the offset within window \\(i\\).

\- The reconstructed time series has shape:

\\[ \boxed{\hat X\in\mathbb{R}^{T\times C}} \\]

\- Evaluate \\(X\\) against \\(\hat X\\), not only \\(H\\) against \\(\hat H\\).

Important compression caveat: the Hankel representation contains overlapping copies of the same original samples. Storing every row of \\(Z_K\\) independently may produce more data than the original series. This is a valid first representation-learning baseline, but actual compression efficiency must be measured after accounting for this redundancy. A later version can use non-overlapping blocks, a stride greater than one, or a streaming representation to reduce encoding overhead.

## 6.7 Selecting L and K

- `L` controls temporal context and the embedded dimension \\(D=LC\\).
- `K` controls the number of retained principal components.
- Evaluate combinations of \\(L\\) and \\(K\\) on validation data.
- Use FFT or power spectral density to propose candidate temporal scales, not to dictate a universal `L_max`.
- Include a range of small, medium, and large temporal contexts, constrained by memory and the amount of training data.
- Fit one maximum-size PCA basis per configuration of \\(L\\), then evaluate prefixes of that basis for different \\(K\\) values.
- Select the final configuration using validation reconstruction quality and compression cost.
- Keep the test set untouched until final evaluation.

# 7. Model interfaces

The key design decision is to separate the model interface from the training algorithm.

- A model defines how to fit its parameters, compress data, and reconstruct data.
- A trainer controls the training process and obtains samples from a sampler.
- An evaluator measures reconstruction quality, compression efficiency, and computational cost.
- A sampler determines which data the trainer receives.

## 7.1 Base model

Suggested name: `CompressionModel`.

```
class CompressionModel:    def fit(self, trainer):        ...    def compress(self, X, **kwargs):        ...    def decompress(self, Z, **kwargs):        ...    def save(self, model_path):        ...    @classmethod    def load(cls, model_path):        ...
```

- `fit(trainer)` delegates training to the configured training procedure, or calls the appropriate algorithm-specific fitting routine.
- `compress(X)` converts input data into a representation that can be stored or transmitted.
- `decompress(Z)` reconstructs the input from that representation.
- `save()` and `load()` persist and restore the model parameters and configuration.
- The model configuration should be a readable dictionary or JSON-compatible structure.

Interface contract: `compress()` must produce a representation that `decompress()` can interpret using the saved model and any explicitly supplied metadata. The representation does not have to be a NumPy array for every future algorithm; it may contain codes, quantized values, or structured data.

## 7.2 Dummy model

Suggested name: `IdentityCompressionModel`.

```
class IdentityCompressionModel(CompressionModel):    def compress(self, X, **kwargs):        return X    def decompress(self, Z, **kwargs):        return Z
```

- Acts as an identity reference.
- Useful for testing the framework, evaluator, and data pipeline.
- Has no learned parameters.
- Does not provide compression; it establishes the zero-distortion reference.

## 7.3 PCA baseline

Suggested name: `HankelPCAModel`.

```
class HankelPCAModel(CompressionModel):    def __init__(self, config):        ...    def fit(self, trainer):        ...    def compress(self, X, top_k=None):        ...    def decompress(self, Z, top_k=None):        ...    def save(self, model_path):        ...    @classmethod    def load(cls, model_path):        ...
```

- `config` contains:
  - `hankel_order`
  - `max_components`
  - Normalization configuration.
  - PCA solver and numerical settings.
  - Any windowing or stride configuration.
- The model stores:
  - Training mean in the embedded feature space.
  - PCA components.
  - Explained variances or singular values, as needed.
  - Normalization parameters.
  - Embedding configuration and model version.
- `top_k` selects a prefix of the fitted component basis.
- The implementation must define whether `compress()` accepts an entire series or an already-created embedding. My recommendation is to expose the full-series API and keep embedding/inverse-embedding as internal or separately testable operations.

## 7.4 Training and sampling

```
class Trainer:    def __init__(self, model, sampler, config):        ...    def fit(self, num_steps):        ...    def validate(self, evaluator):        ...
```

- The PCA implementation may use a specialized trainer, such as `PCATrainer`, to stream embeddings and fit PCA.
- Future implementations can include:
  - `SelfSupervisedTrainer`
  - `ReinforcementLearningTrainer`
- These trainers share the same high-level lifecycle but may have different training steps, objectives, and sampling requirements.

This is preferable to placing all algorithm-specific training logic inside `CompressionModel`.

# 8. Evaluation and experiment orchestration

Suggested names:

```
class ExperimentRunner:    def run(self, config):        ...class ModelEvaluator:    def evaluate(self, model, dataset):        ...class ExperimentResult:    ...
```

- `ExperimentRunner`:
  - Loads experiment configuration.
  - Builds datasets, augmentations, samplers, model, and trainer.
  - Runs training and validation.
  - Saves results and artifacts.
- `ModelEvaluator`:
  - Runs compression and decompression.
  - Compares reconstructed data against ground truth.
  - Calculates reconstruction metrics, compression metrics, and runtime metrics.
  - Aggregates results across windows, channels, and datasets.
- `ExperimentResult`:
  - Stores the configuration, metric summaries, and artifact references.
  - Exports JSON-compatible dictionaries.

Example configuration:

```
config = {    "model": {        "name": "hankel_pca",        "hankel_order": 20,        "max_components": 100,    },    "training": {        "num_steps": 1000,        "batch_size": 256,        "seed": 42,    },    "evaluation": {        "metrics": ["r2", "mse", "mape", "mae"],        "report_distributions": True,        "measure_compression_ratio": True,    },}
```

The values are illustrative. For PCA, `num_steps` and `batch_size` are meaningful only if the selected PCA fitting procedure uses a streaming or iterative algorithm. A full-data PCA fit may instead use a configuration describing the solver and memory budget.

# 9. Save/load contract

- Save the model separately from experiment results.
- The saved model must contain enough information to reproduce decompression.
- Record the preprocessing and normalization conventions.
- Record the selected number of components and the supported range of \\(K\\).
- Include a model format version.
- Validate the restored model against a known input using a save/load round-trip test.
- Experiment results should contain:
  - Dataset and split identifiers.
  - Model configuration.
  - Training configuration and seed.
  - Reconstruction metric distributions.
  - Compression ratio and bits per original scalar.
  - Runtime and memory measurements.
  - References to saved model artifacts.

# 10. Final architecture

The resulting design separates responsibilities cleanly:

```
Dataset
  └── Augmentation pipeline
        └── Sampler
              └── Trainer
                    └── CompressionModel
                          ├── fit / learned parameters
                          ├── compress
                          ├── decompress
                          └── save / load

ExperimentRunner
  ├── builds the components
  ├── runs training
  ├── runs validation
  └── stores results

ModelEvaluator
  ├── reconstruction metrics
  ├── metric distributions
  ├── compression metrics
  └── runtime and memory metrics
```
