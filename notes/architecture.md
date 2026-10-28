# task 

- multimodel time series data compression
- baseline : hankel martrix -> PCA -> select top K, then use for reconstruction

# 1, dataset structure 

- bronze - raw unsorted data, different formats, usually as downloaded
- silver - cleaned up, normalized, same format, fixed missing values, NaN, ready for training
- gold   - ready for testing, deployment, production (not needed)

## convert all datasets into same, rigid format first

- store as numpy array if small + json with schema
- if huge dataset, use arrow format (amazon) or oss polars 

- json example, each json paired with one numpy dataset

```json
{
  "file_name"           : "data.npy",
  "num_samples"         : 1234,
  "num_columns"         : 10,
  "columns_names"       : ["A", "B", "C"],
  "normalization_mean"  : [1, 2, 3, 4],
  "normalization_std"   : [1, 2, 3, 4],
}
```

- provide split train / val here ? 


# 2, dataset loader class

- one dataset loader
- methods : 
    load(file_path)
    x = get_row(row_idx),                 x.shape = (1, num_columns)
    x = get_window(row_idx, window_size), x.shape = (window_size, num_columns)
    __len__()   : return number of rows (num_samples)
    get column_names() : return column names

- this class apply mean and std for normalization, all returned data are normalized
- all returned data are valid, no NaN, no missing values


# 3, data augmentation

- class obtain dataset loader in constructor
- class inerits from base class DummyAugmentation (which returns identity)
- augmentation can ba chained : 
a = AugA(dataset)
b = AugB(a)
c = AugB(b)
...


# metrics 

- R2 coeff
- MSE, MAPE
- all include distribution
- if single number must be presented : R2 mean, R2  wortst 68%, 95%, and 99.7%
- always mean, std, histogram
- as soon baseline trained, include relative metrics

- TODO, validate, suggest




# main approach 



## data preparation

- load dataset
- create augmentation path

## 1, baseline - TODO check algorithm

- 1, train on all datasets
- 2, validate for different hankel order L and K features send
- 3, K : traverse relative with respect to num_columns (0.1%, 1%, 10% ... )
- 4, L : traverse from 1 .. L_max (how to estimate ? from FFT ??? and lowest periodic freq in data ?)

returns json (python dictionary)


```python
class BasicMetric:

    def __init__(self)

    def clear()
    def add(X_ground_truth, X_prediction)
    def eval()
```

```python
class ModelEvaluator:

    def __init__(self)

    def clear()
    def add(X_ground_truth, X_prediction)
    def eval()
```



### training

- sample long window X = (window_size, num_columns)
- if dataset small, one big window, otherwise batched iterative approach 
- construct hankel matrix of order L : H = (window_size-L, L*num_columns) (TODO check shape)
- compute PCA, eigen vectors called P (projection matrix),
P shape : (L*num_columns, L*num_columns)

### inference / validation 

- sample data : X = (window_size, num_columns)
- construct hankel matrix H as in training 
- data compression : Z = H@P  (or PˆT ???? !!! validate), prefered format is H first
- from Z selected top K columns, get Z_sel

- decompression : 
P_strip : select only top K columns
X_decoded = Z_sel @ P #reconstruction

- evaluate metrics
- save all model

class sketch - inrirts basic structure from DummyModel class,
args shall be in dictionary, json-like, so they are clear readable passed for many models
```python
class EigenBaseline:

    def __init__(self, dataset_aug, num_steps, hankel_order_L, top_K_values)


    def fit(dataset_iterator ??, num_steps)
    
    def compress(X, top_K)

    def decompress(Z)

    def save(model_path)
    def load(model_path)
```

DummyModel class : 
```python
class DummyModel:

    def __init__(self, dataset_aug)


    def fit(dataset_iterator ??, num_steps)
    
    def compress(X)

    def decompress(Z)

    def save(model_path)
    def load(model_path)
```