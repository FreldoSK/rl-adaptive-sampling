# Dataset Loader and Augmentation

## 1. Dataset contract

- Shape: `(num_samples, num_channels)`.
- Fixed channel ordering within each dataset.
- No unresolved NaN or infinite values.
- Explicit missing-value and constant-channel policies.
- Normalization parameters fitted on training data only.
- Split definitions established before creating overlapping windows.

## 2. Dataset interface

```python
class TimeSeriesDataset:
    def __len__(self) -> int:
        ...

    def get_row(self, row_idx: int):
        """Return shape (C,)."""
        ...

    def get_window(self, row_idx: int, window_size: int):
        """Return shape (window_size, C)."""
        ...

    @property
    def column_names(self):
        ...

    @property
    def num_channels(self) -> int:
        ...
```

- `row_idx` is the starting index of a window.
- Invalid window boundaries raise an explicit error unless padding is configured.
- Large datasets may use memory mapping or chunked storage.
- Preserve original units and preprocessing metadata for evaluation.

## 3. Augmentation interface

```python
class DatasetTransform:
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def get_row(self, row_idx):
        return self.dataset.get_row(row_idx)

    def get_window(self, row_idx, window_size):
        return self.dataset.get_window(row_idx, window_size)
```

This is a structural sketch. Concrete transformations override the methods they need to change and delegate all other access and metadata to the wrapped dataset.

## 4. Identity augmentation

```python
class IdentityAugmentation(DatasetTransform):
    pass
```

- Returns the underlying data unchanged.
- Used for baseline experiments and interface testing.

## 5. Composable augmentations

```python
dataset = TimeSeriesDataset(...)

a = AugmentationA(dataset)
b = AugmentationB(a)
c = AugmentationC(b)
```

- Every augmentation follows the same interface.
- Stochastic transformations use explicit random seeds.
- Transformations preserve temporal alignment and channel semantics unless configured otherwise.
- Augmentation order is recorded in the experiment configuration.

## 6. Sampling is separate

```python
class WindowSampler:
    def __init__(self, dataset, seed=None):
        ...

    def sample(self, batch_size, window_size):
        ...


class SequenceSampler:
    def __init__(self, dataset, seed=None):
        ...

    def sample(self, batch_size, sequence_length):
        ...
```

- Window samplers provide batches of windows.
- Sequence samplers provide ordered sequences.
- Future self-supervised methods may require paired views.
- Future reinforcement-learning methods may require transitions or trajectories.
- Samplers must respect dataset splits and recording boundaries.

## 7. Data preparation

1. Load the silver dataset.
2. Select the appropriate split.
3. Apply fitted preprocessing.
4. Apply configured augmentations.
5. Sample windows or sequences.
6. Pass samples to the trainer.

Fit normalization statistics on training data only. Do not randomly split overlapping windows into train and validation partitions.
