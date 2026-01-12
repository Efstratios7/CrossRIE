# 

[![PyPI version](https://img.shields.io/pypi/v/compact-rienet.svg)](https://pypi.org/project/compact-rienet/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**This library implements the neural estimators introduced in:**
- **Manolakis, E., Bongiorno, C., & Mantegna, R. N. (2025). Physics-Informed Singular-Value Learning for Cross-Covariances Forecasting in Financial Markets. Working Paper.**


## Key Features
- **Generalized Cross-Correlation Correction (CCC)**: Uses deep learning to denoise cross-correlation matrices by leveraging empirical marginal correlations and singular value decomposition.
- **Deep Spectral Denoising**: Implements a shared encoder (MLP) and a bidirectional LSTM aggregator to clean singular values based on global spectral context.
- **Flexible Correction Mechanisms**: Supports both additive (default) and bounded multiplicative corrections for singular values.
- **Dimension Awareness**: explicitly incorporates aspect ratios and system dimensions ($n_x, n_y, \Delta t_{in}$) into the correction logic.
- **Practical Outputs**: Returns naturally denoised cross-correlation matrices ($\mathbf{C}_{XY}$) and/or cleaned singular values ($\widetilde{s}_k$).
- **TensorFlow/Keras Implementation**: Built as a standard Keras Layer for easy integration into larger deep learning models.

## Installation
Install from source:

```bash
git clone https://github.com/Efstratios7/CrossRIE.git
cd CrossRIE
pip install -e .
```

## Quick Start

### Basic Usage
The core component is the `CrossRIELayer`. It expects four inputs: the two marginal covariance matrices ($\mathbf{C}_{XX}, \mathbf{C}_{YY}$), the cross-correlation matrix ($\mathbf{C}_{XY}$), and the number of samples ($n$).

```python
import tensorflow as tf
from crossrie.layer import CrossRIELayer

# Initialize the layer
# By default, it returns the cleaned Cross-Correlation matrix 'Cxy'
cross_rie = CrossRIELayer(
    encoding_units=[16, 2],
    lstm_units=[128, 64],
    outputs=['Cxy', 'Sxy']
)

# Generate dummy data (Batch, N, M)
B, N, M, T = 32, 10, 12, 100
Cxx = tf.random.normal((B, N, N))
Cyy = tf.random.normal((B, M, M))
Cxy = tf.random.normal((B, N, M))
n_samples = tf.constant([T] * B) # Number of samples which are used to compute the covariance matrices Cxx and Cyy

# Forward pass
outputs = cross_rie([Cxx, Cyy, Cxy, n_samples])

Cxy_clean = outputs['Cxy']      # Denoised Cross-Correlation
Sxy_clean = outputs['Sxy']      # Cleaned Singular Values

print("Cleaned Cxy shape:", Cxy_clean.shape)
print("Cleaned Sxy shape:", Sxy_clean.shape)
```

### Training
The layer is fully differentiable and can be trained using standard Keras optimization workflows.

```python
import tensorflow as tf
from keras import Model, Input
from crossrie.layer import CrossRIELayer

B, N, M, T = 32, 10, 12, 100

def create_model():
    # Shapes are (None, None) to allow variable sequence lengths
    input_cxx = Input(shape=(None, None), name='Cxx')
    input_cyy = Input(shape=(None, None), name='Cyy')
    input_cxy = Input(shape=(None, None), name='Cxy')
    input_n = Input(shape=(1,), name='n_samples')
    
    # Forward pass
    cxy_clean = CrossRIELayer(outputs=['Cxy'])([input_cxx, input_cyy, input_cxy, input_n])
    
    return Model(inputs=[input_cxx, input_cyy, input_cxy, input_n], outputs=cxy_clean)

model = create_model()
model.compile(optimizer='adam', loss='mse')

# Training Data
# In a real scenario, these would be computed from your data.
# Cxx: (Batch, N, N), Cyy: (Batch, M, M), Cxy: (Batch, N, M)
Cxx = tf.random.normal((B, N, N))
Cyy = tf.random.normal((B, M, M))
Cxy = tf.random.normal((B, N, M))

# n_samples must match the batch dimension. 
# It represents the number of time steps T used to compute the correlations/covariances.
# Shape: (Batch, 1) or (Batch,)
n_samples = tf.constant([[float(T)] for _ in range(B)]) 

# Target Variable (Cleaned Cxy)
# In supervised learning, this would be the "true" cross-correlation.
Y_target = tf.random.normal((B, N, M))

model.fit([Cxx, Cyy, Cxy, n_samples], Y_target, epochs=1, batch_size=32)
```

### Different Output Types
You can configure the layer to return different components by passing a list of keys to `outputs`.

- `'Cxy'`: The reconstructed, denoised cross-correlation matrix.
- `'Sxy'`: The vector of cleaned singular values.

```python
# Returns only the cleaned singular values
layer_s = CrossRIELayer(outputs=['Sxy'])
s_tilde = layer_s([Cxx, Cyy, Cxy, n_samples])
```

## Requirements
- Python >= 3.8
- TensorFlow >= 2.10.0
- Keras >=3.12.0
- NumPy >= 1.26.4

## Development
```bash
git clone https://github.com/Efstratios7/CrossRIE.git
cd CrossRIE
pip install -e ".[dev]"
pytest tests/
```

## Citation

## Support
For questions, issues, or contributions, please:

- Open an issue on [GitHub](https://github.com/bongiornoc/Compact-RIEnet/issues)
- Check the documentation
- Contact Efstratios Manolakis (<stratomanolaki@gmail.com>)
- Contact Prof. Christian Bongiorno (<christian.bongiorno@centralesupelec.fr>) for calibrated model weights or collaboration requests
