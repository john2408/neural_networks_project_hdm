# MLP Multivariate Forecasting Approach

## Overview

The **MLPMultivariate** model implements a fundamentally different approach to multivariate time series forecasting compared to our other models (LSTM, RNN, GRU, CNN, Transformer).

## Key Concept: Flattening vs One-Hot Encoding

### Traditional Approach (One-Hot Encoding)
Our existing models use **one-hot encoding** to represent time series identity:

```
Input for ONE time series:
Shape: (batch_size, seq_length, n_features)
Features: [value, additional_features, year, month, one_hot_vector]

Example with 1000 time series, 5 features:
(32, 6, 1 + 5 + 2 + 1000) = (32, 6, 1008)
```

**Characteristics:**
- Each sample contains ONE time series
- Batch mixes different time series
- Sparse representation (one-hot vectors mostly zeros)
- High dimensionality for many series
- Series identity via one-hot encoding

### Flattening Approach (MLPMultivariate)
The new approach **flattens ALL time series values** into a single input vector:

```
Input for ALL time series together:
Shape: (batch_size, seq_length, n_series + n_series*n_features + 2)
Features per timestep: [val₁, val₂, ..., valₙ, feat₁₁, ..., featₙₖ, year, month]

Example with 1000 time series, 5 features:
(32, 6, 1000 + 1000*5 + 2) = (32, 6, 6002)
```

**Characteristics:**
- Each sample contains ALL time series
- Batch represents different timesteps
- Dense representation (all actual values)
- Direct cross-series learning
- Series identity implicit in position

## Architecture Comparison

### Data Flow: One-Hot Encoding Models
```
Data Organization:
├─ Sample 1: Series A, timesteps [t-5, t-4, t-3, t-2, t-1, t] → Predict A(t+1)
├─ Sample 2: Series B, timesteps [t-5, t-4, t-3, t-2, t-1, t] → Predict B(t+1)
└─ Sample 3: Series C, timesteps [t-5, t-4, t-3, t-2, t-1, t] → Predict C(t+1)

Batch (size=32): Mix of different series at various timesteps
Model learns: Series-specific patterns via one-hot encoding
```

### Data Flow: Flattening Approach
```
Data Organization:
├─ Sample 1: ALL series, timesteps [t-5, t-4, t-3, t-2, t-1, t] → Predict ALL(t+1)
├─ Sample 2: ALL series, timesteps [t-4, t-3, t-2, t-1, t, t+1] → Predict ALL(t+2)
└─ Sample 3: ALL series, timesteps [t-3, t-2, t-1, t, t+1, t+2] → Predict ALL(t+3)

Batch (size=32): Same timestep windows across all series
Model learns: Cross-series relationships directly from values
```

## MLPMultivariate Model Architecture

```
Input: (batch_size, seq_length, n_series*(1+n_features) + 2)
                ↓
    Extract temporal features (year, month)
                ↓
    Flatten sequence: (batch_size, flattened_dim)
                ↓
    MLP Layer 1: Linear → ReLU → Dropout
                ↓
    MLP Layer 2: Linear → ReLU → Dropout
                ↓
             ...
                ↓
    MLP Layer N: Linear → ReLU → Dropout
                ↓
    Output Layer: Linear(hidden_size → n_series)
                ↓
Output: (batch_size, n_series) - predictions for all series
```

### Input Dimension Calculation
```python
input_dim = (
    n_series * seq_length +                      # All values across time
    n_series * n_features * seq_length +         # All features across time  
    2                                             # Temporal (year, month)
)
```

## Advantages of Flattening

1. **Cross-Series Learning**: Directly learns relationships between different time series
2. **Memory Efficiency**: Dense representation instead of sparse one-hot vectors
3. **Scalability**: Better for datasets with many time series (100s-1000s)
4. **Simplicity**: No need for embedding layers or one-hot encoding
5. **Direct Modeling**: Values themselves carry the information, not categorical encodings

## Disadvantages

1. **Synchronized Data Required**: All series must have observations at same timesteps
2. **Missing Data Handling**: More complex than one-hot approach
3. **Single Forecast**: Predicts all series simultaneously (can't predict single series in isolation)
4. **Memory for Few Series**: May be less efficient for very few time series (<10)

## When to Use Each Approach

### Use One-Hot Encoding (LSTM, RNN, GRU, etc.) when:
- Few time series (< 100)
- Irregular/unsynchronized observations
- Need to predict individual series independently
- Series have very different characteristics

### Use Flattening (MLPMultivariate) when:
- Many time series (100-1000+)
- Regular, synchronized observations
- Believe series are interdependent
- Want to capture cross-series patterns

## Implementation Details

### Dataset: TimeSeriesDatasetFlattened
- Organizes data by **timestep** instead of by **series**
- Each sample contains all series values at a sequence of timesteps
- No one-hot encoding
- Maintains temporal features (year, month)

### Training Modifications
- Different loss calculation (per-series or aggregate)
- Different prediction extraction (select specific series from output)
- Modified evaluation metrics
- Adjusted autoregressive forecasting

## Inspiration

This approach is inspired by Nixtla's `MLPMultivariate` model, which has shown effectiveness in multi-series forecasting competitions by directly modeling cross-series dependencies rather than treating series as independent.

## References
- [Nixtla NeuralForecast](https://github.com/Nixtla/neuralforecast/blob/main/neuralforecast/models/mlpmultivariate.py)
- Original implementation: `neuralforecast/models/mlpmultivariate.py`
