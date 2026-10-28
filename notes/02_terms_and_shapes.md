# Terms, Shapes, and Equations

## 1. Input time series

Let:

- T = number of time steps.
- C = number of channels.
- L = Hankel embedding order.
- K = number of retained principal components.

The input window is:

$
X\in\mathbb{R}^{T\times C}
$

Example: `X.shape == (1000, 10)`.

## 2. Hankel embedding

Create a flattened sliding window of L consecutive time steps for each valid starting index.

$
N=T-L+1,\qquad D=LC
$

$
H\in\mathbb{R}^{N\times D}
$

Each row contains L consecutive observations, flattened across C channels.

Example: T = 1000, C = 10, L = 20.

- N = 981 sliding windows.
- D = 200 features per window.
- H shape = `(981, 200)`.

The embedding creates overlapping windows and therefore introduces redundancy.

## 3. PCA fitting

Fit PCA on centered training embeddings only.

$
H_c=H-\mu
$

where:

$
\mu\in\mathbb{R}^{D}
$

Using singular value decomposition:

$
H_c=U\Sigma V^\top
$

Store the first K principal components as columns:

$
P_K\in\mathbb{R}^{D\times K}
$

The maximum useful K is bounded by the available training samples and embedded dimension. For centered data, the rank is at most `min(N_train - 1, D)`.

## 4. Compression

Project the centered embedding onto the retained components:

$
Z_K=(H-\mu)P_K
$

Shapes:

$
(N\times D)(D\times K)=(N\times K)
$

Therefore:

$
Z_K\in\mathbb{R}^{N\times K}
$

K is the number of retained principal components, not a fraction of the original channel count.

## 5. Decompression

Reconstruct the embedded windows:

$
\hat H=Z_KP_K^\top+\mu
$

Shapes:

$
(N\times K)(K\times D)=(N\times D)
$

Therefore:

$
\hat H\in\mathbb{R}^{N\times D}
$

The PCA mean must be added back after inverse projection.

## 6. Reconstructing the original series

- Reshape each reconstructed row into a window of shape `(L, C)`.
- Place each window at its original time indices.
- Average overlapping estimates for each time step.

The final reconstruction has shape:

$
\hat X\in\mathbb{R}^{T\times C}
$

Evaluate the original series against this reconstructed series, not only the embedded matrices.

## 7. Compression accounting

The Hankel matrix contains overlapping copies of the original observations. Storing all projected windows may therefore cost more than storing the original series.

$
C_{ratio} = \frac{original_{size}}{compressed_{size}}
$


Include coefficients, model parameters, quantization information, and required metadata in actual storage measurements.

## 8. Naming conventions

- `T`: original time steps.
- `C`: channels.
- `L`: Hankel order or embedding length.
- `N`: number of embedded windows.
- `D`: embedded feature dimension, `L * C`.
- `K`: retained principal components.
- `X`: original time series.
- `H`: embedded window matrix.
- `mu`: PCA mean in embedded feature space.
- `P_K`: retained PCA components, shape `(D, K)`.
- `Z_K`: compressed coefficients, shape `(N, K)`.
- `H_reconstructed`: reconstructed embedding.
- `X_reconstructed`: reconstructed original series.
