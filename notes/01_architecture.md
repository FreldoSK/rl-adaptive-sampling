# Overall Architecture

## Task

- Multimodel time-series data compression.
- Baseline: Hankel embedding → PCA → retain top K principal components → reconstruction.
- Future models: self-supervised learning, reinforcement learning, and other learned compression methods.
- All models share the same data, sampling, evaluation, and compression interfaces.

## 1. Dataset pipeline

- **Bronze:** raw downloaded datasets in their original formats.
- **Silver:** cleaned, validated, normalized datasets in a consistent format.
- **Gold:** deployment-ready artifacts and production outputs; out of scope initially.

- Convert datasets to a common numerical representation.
- Small datasets: NumPy arrays plus JSON metadata.
- Large datasets: Arrow/Parquet, Polars, or memory-mapped NumPy arrays.
- Define train/validation/test splits before generating overlapping windows.
- Fit normalization parameters using training data only.

## 2. Dataset loader

- One common interface for loading and accessing time-series data.
- Supports row access and sliding-window access.
- Exposes channel names, dimensions, and preprocessing metadata.
- Guarantees that returned data follow the configured missing-value and normalization policies.

## 3. Augmentation

- Augmentations wrap a dataset and expose the same access interface.
- Augmentations can be chained.
- Identity augmentation is the default.
- Random augmentations support reproducible seeds.
- Sampling and augmentation are separate responsibilities.

## 4. Sampling

- Samplers determine which training samples are provided to the model.
- Window sampler: samples time-series windows.
- Sequence sampler: provides temporally ordered sequences.
- Future samplers may provide pairs of views, transitions, or other algorithm-specific inputs.

## 5. Training

- Trainer controls the training lifecycle and obtains samples from a sampler.
- PCA may use a full-data or streaming fitting procedure.
- Self-supervised models may use gradient-based optimization.
- Reinforcement-learning models may use sequential interactions and specialized update rules.
- Share the lifecycle, not necessarily the optimization algorithm.

## 6. Compression models

- Common interface: fit, compress, decompress, save, and load.
- `IdentityCompressionModel`: reference implementation without compression.
- `HankelPCAModel`: initial PCA baseline.
- Future models implement the same external contract.

## 7. Evaluation

- Reconstruction quality: R², MSE, MAPE, MAE.
- Distribution reporting: mean, standard deviation, histogram, and quantiles.
- Compression efficiency: compression ratio and bits per original scalar.
- Computational cost: training time, throughput, memory, and model size.
- Evaluate on held-out data and report per-dataset and per-channel results.

## 8. Experiment orchestration

- `ExperimentRunner`: builds components and executes experiments.
- `ModelEvaluator`: evaluates reconstruction quality, storage cost, and runtime.
- `ExperimentResult`: stores configurations, metrics, and artifact references.
- All experiments use reproducible configurations and seeds.

## Initial implementation order

1. Validate the data pipeline.
2. Implement the identity model and metrics.
3. Implement Hankel embedding and inverse embedding.
4. Implement the PCA baseline.
5. Evaluate K and L on validation data.
6. Add real coefficient encoding and storage accounting.
7. Introduce more advanced models.
