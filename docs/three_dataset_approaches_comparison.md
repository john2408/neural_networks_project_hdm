# Three Dataset Approaches: Complete Comparison

## Overview

This document compares three fundamentally different approaches to organizing time series data for neural network training:

1. **TimeSeriesDataset**: Traditional per-series samples with one-hot encoding
2. **TimeSeriesDatasetFlattened**: Time-first organization without one-hot encoding
3. **TimeSeriesDatasetVectorized**: Nixtla-style vectorized batching (univariate only)

---

## Concrete Example Setup

Throughout this document, we'll use a consistent example:

```python
# Dataset configuration
n_series = 1000          # Number of time series
n_timesteps = 60         # Values per series
seq_length = 6           # Lookback window
embargo = 1              # Gap before prediction
batch_size = 32          # For TimeSeriesDataset and TimeSeriesDatasetFlattened
batch_size_vec = 16      # For TimeSeriesDatasetVectorized
n_features = 3           # Additional features (GDP, CPI, Interest_Rate)

# For TimeSeriesDataset and TimeSeriesDatasetFlattened
# For TimeSeriesDatasetVectorized: no additional features (univariate)
```

---

## Part 1: Philosophical Differences

### TimeSeriesDataset (Traditional One-Hot)
```
Paradigm: One sample per series per time window
Philosophy: Treat each time series independently with identity encoding
Batching: Mix different series and time windows randomly
```

**Key characteristics:**
- Each sample represents ONE time series at ONE time window
- Series identity encoded as one-hot vector (1000 dimensions)
- Supports multivariate features
- Most flexible but least efficient

### TimeSeriesDatasetFlattened (Time-First)
```
Paradigm: One sample per time window, all series flattened
Philosophy: Organize by global timestep, concatenate all series
Batching: Mix different time windows randomly
```

**Key characteristics:**
- Each sample represents ALL series at ONE time window
- No one-hot encoding (series identity implicit in position)
- Supports multivariate features
- More efficient than TimeSeriesDataset

### TimeSeriesDatasetVectorized (Nixtla-Style)
```
Paradigm: One sample per time window, all series vectorized
Philosophy: Batch across TIME dimension only
Batching: Sequential time windows with all series together
```

**Key characteristics:**
- Each sample represents ALL series at ONE time window
- No one-hot encoding needed
- **UNIVARIATE ONLY** (no additional features)
- Most efficient but most restrictive

---

## Part 2: Dataset Size Comparison

Given: 1000 series × 60 timesteps, seq_length=6, embargo=1

### Number of Samples

**Available time windows per series:**
```
n_windows = n_timesteps - seq_length - embargo = 60 - 6 - 1 = 53
```

#### TimeSeriesDataset
```python
dataset_length = n_series × n_windows
               = 1000 × 53
               = 53,000 samples
```

Each sample: One series, one time window

#### TimeSeriesDatasetFlattened
```python
dataset_length = n_windows
               = 53 samples
```

Each sample: All series, one time window (flattened)

#### TimeSeriesDatasetVectorized
```python
dataset_length = n_windows
               = 53 samples
```

Each sample: All series, one time window (vectorized)

### Sample Reduction Factor

| Dataset Type | Samples | Reduction vs Traditional |
|--------------|---------|-------------------------|
| TimeSeriesDataset | 53,000 | Baseline (1x) |
| TimeSeriesDatasetFlattened | 53 | **1000x fewer** |
| TimeSeriesDatasetVectorized | 53 | **1000x fewer** |

---

## Part 3: Data Structure Comparison

### Example: Single Sample from Each Dataset

#### TimeSeriesDataset (Sample #25,000)

This could be series #471 at time window #26:

```python
X.shape = (seq_length, n_features_total)
        = (6, 1 + 3 + 2 + 1000)
        = (6, 1006)

# Feature composition per timestep:
# [Value, GDP, CPI, Interest_Rate, year, month, onehot_0, ..., onehot_999]

y.shape = (1,)

# Example X (timestep t-5 to t):
[
  [1234.5, 2.3, 105.2, 3.5, 2023, 1, 0, 0, ..., 1, ..., 0],  # t-5
  [1245.2, 2.3, 105.3, 3.5, 2023, 2, 0, 0, ..., 1, ..., 0],  # t-4
  [1256.8, 2.4, 105.4, 3.6, 2023, 3, 0, 0, ..., 1, ..., 0],  # t-3
  [1267.3, 2.4, 105.5, 3.6, 2023, 4, 0, 0, ..., 1, ..., 0],  # t-2
  [1278.9, 2.5, 105.6, 3.7, 2023, 5, 0, 0, ..., 1, ..., 0],  # t-1
  [1289.1, 2.5, 105.7, 3.7, 2023, 6, 0, 0, ..., 1, ..., 0]   # t
]

# Target y (timestep t+1+embargo):
[1302.4]
```

#### TimeSeriesDatasetFlattened (Sample #26)

This is time window #26 with ALL 1000 series:

```python
X.shape = (seq_length, n_features_total)
        = (6, 1000 + 1000*3 + 2)
        = (6, 4002)

# Feature composition per timestep:
# [series_0_val, series_1_val, ..., series_999_val,
#  series_0_GDP, series_1_GDP, ..., series_999_GDP,
#  series_0_CPI, series_1_CPI, ..., series_999_CPI,
#  series_0_Int, series_1_Int, ..., series_999_Int,
#  year, month]

y.shape = (1000,)

# Example X (conceptual, showing structure):
[
  [val_0, val_1, ..., val_999, gdp_0, ..., gdp_999, cpi_0, ..., cpi_999, int_0, ..., int_999, 2023, 1],  # t-5
  [val_0, val_1, ..., val_999, gdp_0, ..., gdp_999, cpi_0, ..., cpi_999, int_0, ..., int_999, 2023, 2],  # t-4
  [val_0, val_1, ..., val_999, gdp_0, ..., gdp_999, cpi_0, ..., cpi_999, int_0, ..., int_999, 2023, 3],  # t-3
  [val_0, val_1, ..., val_999, gdp_0, ..., gdp_999, cpi_0, ..., cpi_999, int_0, ..., int_999, 2023, 4],  # t-2
  [val_0, val_1, ..., val_999, gdp_0, ..., gdp_999, cpi_0, ..., cpi_999, int_0, ..., int_999, 2023, 5],  # t-1
  [val_0, val_1, ..., val_999, gdp_0, ..., gdp_999, cpi_0, ..., cpi_999, int_0, ..., int_999, 2023, 6]   # t
]

# Target y (all series at timestep t+1+embargo):
[y_0, y_1, y_2, ..., y_999]
```

#### TimeSeriesDatasetVectorized (Sample #26)

This is time window #26 with ALL 1000 series (vectorized):

```python
X.shape = (n_series, seq_length, n_features)
        = (1000, 6, 1)

# Feature: Value only (univariate)

y.shape = (1000,)

# Example X (series dimension first):
[
  [[1234.5], [1245.2], [1256.8], [1267.3], [1278.9], [1289.1]],  # Series 0: t-5 to t
  [[2341.2], [2356.7], [2372.1], [2387.4], [2402.8], [2418.3]],  # Series 1: t-5 to t
  [[3456.8], [3478.9], [3501.2], [3523.6], [3546.1], [3568.7]],  # Series 2: t-5 to t
  ...
  [[9876.5], [9912.3], [9948.7], [9985.2], [10022], [10059]]     # Series 999: t-5 to t
]
# Shape: (1000, 6, 1)

# Target y (all series at timestep t+1+embargo):
[1302.4, 2433.9, 3591.4, ..., 10096]
# Shape: (1000,)
```

---

## Part 4: Batch Structure Comparison

### Batch Shapes

#### TimeSeriesDataset (batch_size=32)

```python
X_batch.shape = (batch_size, seq_length, n_features_total)
              = (32, 6, 1006)

y_batch.shape = (batch_size, 1)
              = (32, 1)

# This batch contains 32 random samples
# Could be: series 42 window 10, series 721 window 33, series 5 window 18, ...
# Each sample is from ONE series at ONE time window
```

**Interpretation:**
- 32 independent predictions
- Mixed series and time windows
- Each row: one series' sequence with its one-hot encoding

#### TimeSeriesDatasetFlattened (batch_size=32)

```python
X_batch.shape = (batch_size, seq_length, n_features_total)
              = (32, 6, 4002)

y_batch.shape = (batch_size, n_series)
              = (32, 1000)

# This batch contains 32 random time windows
# Could be: window 5, window 22, window 41, window 3, ...
# Each window contains ALL 1000 series
```

**Interpretation:**
- 32 time windows
- Each window has all 1000 series flattened
- Total predictions per batch: 32 × 1000 = 32,000

#### TimeSeriesDatasetVectorized (batch_size=16)

```python
X_batch.shape = (batch_time, n_series, seq_length, n_features)
              = (16, 1000, 6, 1)

y_batch.shape = (batch_time, n_series)
              = (16, 1000)

# This batch contains 16 SEQUENTIAL time windows
# Windows: t, t+1, t+2, ..., t+15
# Each window contains ALL 1000 series (vectorized, not flattened)
```

**Interpretation:**
- 16 time windows (typically sequential)
- Each window has all 1000 series as separate vectors
- Total predictions per batch: 16 × 1000 = 16,000
- **Must reshape before model**: (16×1000, 6, 1) = (16000, 6, 1)

---

## Part 5: Model Input Reshaping

### TimeSeriesDataset
```python
# NO RESHAPING NEEDED
X_batch = X_batch.to(device)  # (32, 6, 1006)
predictions = model(X_batch)   # (32, 1)
```

### TimeSeriesDatasetFlattened
```python
# NO RESHAPING NEEDED
X_batch = X_batch.to(device)  # (32, 6, 4002)
predictions = model(X_batch)   # (32, 1000)
```

### TimeSeriesDatasetVectorized
```python
# RESHAPING REQUIRED
batch_time, n_series, seq_len, n_feats = X_batch.shape  # (16, 1000, 6, 1)

X_reshaped = X_batch.view(batch_time * n_series, seq_len, n_feats)  # (16000, 6, 1)
y_reshaped = y_batch.view(batch_time * n_series, 1)                 # (16000, 1)

X_reshaped = X_reshaped.to(device)
y_reshaped = y_reshaped.to(device)

predictions = model(X_reshaped)  # (16000, 1)
```

---

## Part 6: Feature Engineering Comparison

### TimeSeriesDataset (Most Flexible)

**Supports:**
- ✅ Value (target variable history)
- ✅ Exogenous features (economic indicators, weather, etc.)
- ✅ Temporal encoding (year, month, day, etc.)
- ✅ One-hot series encoding (series identity)
- ✅ Different features per series (if needed)

**Feature vector per timestep:**
```python
n_features = 1 + n_exog + n_temporal + n_series
           = 1 + 3 + 2 + 1000
           = 1006
```

### TimeSeriesDatasetFlattened (Flexible)

**Supports:**
- ✅ Value (all series concatenated)
- ✅ Exogenous features (all series concatenated)
- ✅ Temporal encoding (shared across series)
- ❌ No one-hot encoding (series identity implicit in position)

**Feature vector per timestep:**
```python
n_features = n_series * (1 + n_exog) + n_temporal
           = 1000 * (1 + 3) + 2
           = 4002
```

### TimeSeriesDatasetVectorized (Restrictive)

**Supports:**
- ✅ Value (vectorized across series)
- ❌ No exogenous features
- ❌ No temporal encoding
- ❌ No one-hot encoding (not needed)

**Feature vector per series per timestep:**
```python
n_features = 1  # Value only
```

---

## Part 7: Training Efficiency Comparison

### Forward Passes Per Epoch

With our example (1000 series, 53 windows):

#### TimeSeriesDataset
```
Total samples: 53,000
Batch size: 32
Forward passes per epoch: 53,000 / 32 = 1,656 passes

Each pass:
- Input: (32, 6, 1006)
- Predictions: 32
```

#### TimeSeriesDatasetFlattened
```
Total samples: 53
Batch size: 32
Forward passes per epoch: 53 / 32 = 2 passes (with 21 samples in last batch)

Each pass:
- Input: (32, 6, 4002) or (21, 6, 4002)
- Predictions: 32 × 1000 = 32,000 or 21 × 1000 = 21,000
```

#### TimeSeriesDatasetVectorized
```
Total samples: 53
Batch size: 16
Forward passes per epoch: 53 / 16 = 4 passes (with 5 samples in last batch)

Each pass (after reshaping):
- Input: (16000, 6, 1) or (5000, 6, 1)
- Predictions: 16,000 or 5,000
```

### Efficiency Summary

| Dataset Type | Forward Passes | Avg Predictions/Pass | Total Predictions | Speedup |
|--------------|----------------|---------------------|-------------------|---------|
| TimeSeriesDataset | 1,656 | 32 | 53,000 | 1x (baseline) |
| TimeSeriesDatasetFlattened | 2 | 26,500 | 53,000 | **828x faster** |
| TimeSeriesDatasetVectorized | 4 | 13,250 | 53,000 | **414x faster** |

Note: Actual training time also depends on:
- Matrix multiplication size (larger = better GPU utilization)
- Memory bandwidth
- Data loading overhead

---

## Part 8: Memory Comparison

### Per-Sample Memory

#### TimeSeriesDataset
```python
X: (6, 1006) × 4 bytes = 24,144 bytes ≈ 24 KB
y: (1,) × 4 bytes = 4 bytes
Total per sample: ≈ 24 KB
Total dataset: 53,000 × 24 KB = 1,272 MB
```

#### TimeSeriesDatasetFlattened
```python
X: (6, 4002) × 4 bytes = 96,048 bytes ≈ 96 KB
y: (1000,) × 4 bytes = 4,000 bytes ≈ 4 KB
Total per sample: ≈ 100 KB
Total dataset: 53 × 100 KB = 5.3 MB
```

#### TimeSeriesDatasetVectorized
```python
X: (1000, 6, 1) × 4 bytes = 24,000 bytes ≈ 24 KB
y: (1000,) × 4 bytes = 4,000 bytes ≈ 4 KB
Total per sample: ≈ 28 KB
Total dataset: 53 × 28 KB = 1.48 MB
```

### Memory Efficiency

| Dataset Type | Total Memory | Memory vs Traditional |
|--------------|--------------|----------------------|
| TimeSeriesDataset | 1,272 MB | Baseline (1x) |
| TimeSeriesDatasetFlattened | 5.3 MB | **240x less** |
| TimeSeriesDatasetVectorized | 1.48 MB | **860x less** |

---

## Part 9: GPU Utilization Comparison

### Batch Processing Characteristics

#### TimeSeriesDataset
```
Batch: (32, 6, 1006)
Matrix operations: Small-medium
GPU cores utilized: ~5-15%
Memory bandwidth: Underutilized
Bottleneck: Too many small operations
```

#### TimeSeriesDatasetFlattened
```
Batch: (32, 6, 4002)
Matrix operations: Medium-large
GPU cores utilized: ~30-60%
Memory bandwidth: Better utilized
Bottleneck: Can saturate memory with large feature vectors
```

#### TimeSeriesDatasetVectorized
```
Batch (reshaped): (16000, 6, 1)
Matrix operations: Very large (many samples)
GPU cores utilized: ~80-95%
Memory bandwidth: Optimally utilized
Bottleneck: Model computation (good thing!)
```

### Real-world Training Time (Example)

Assuming 30 epochs on our dataset:

| Dataset Type | Time per Epoch | Total Training Time | GPU Utilization |
|--------------|----------------|---------------------|-----------------|
| TimeSeriesDataset | ~90 seconds | ~45 minutes | 8-12% |
| TimeSeriesDatasetFlattened | ~5 seconds | ~2.5 minutes | 40-55% |
| TimeSeriesDatasetVectorized | ~3 seconds | ~1.5 minutes | 85-92% |

---

## Part 10: When to Use Each Approach

### Use TimeSeriesDataset When:

✅ **Best for:**
- Need maximum flexibility
- Different series have different feature sets
- Series have different lengths (with padding)
- Experimenting with various feature combinations
- Small datasets (< 100 series)
- Need to sample series randomly for regularization

❌ **Avoid when:**
- Training speed is critical
- Have many series (> 500)
- Features are simple or uniform

**Example use case:** 
Small research project with heterogeneous time series and complex feature engineering requirements.

### Use TimeSeriesDatasetFlattened When:

✅ **Best for:**
- Many time series (100-10,000)
- All series share same exogenous features
- All series aligned by date
- Need multivariate forecasting
- Want better efficiency than TimeSeriesDataset
- Can't sacrifice exogenous features

❌ **Avoid when:**
- Series have different lengths
- Features differ between series
- Need maximum speed (use Vectorized instead)

**Example use case:**
Economic forecasting for 1,000+ regions using shared economic indicators (GDP, inflation, rates).

### Use TimeSeriesDatasetVectorized When:

✅ **Best for:**
- **Univariate forecasting only**
- Many time series (1,000-100,000+)
- All series aligned by date
- Training speed is critical
- Have abundant GPU memory
- No exogenous features needed
- Following Nixtla's approach

❌ **Avoid when:**
- Need exogenous features
- Need temporal encoding
- Series have different lengths
- GPU memory is limited

**Example use case:**
Large-scale univariate forecasting like predicting thousands of product sales or sensor readings.

---

## Part 11: Similarities and Differences Table

### Similarities

| Aspect | All Three Approaches |
|--------|---------------------|
| Purpose | Time series forecasting with sliding windows |
| Embargo | Support gap between observation and prediction |
| Train-test split | Chronological splitting |
| Standardization | StandardScaler for continuous features |
| Output | Single value per series prediction |
| PyTorch | Compatible with PyTorch models |

### Differences

| Aspect | TimeSeriesDataset | TimeSeriesDatasetFlattened | TimeSeriesDatasetVectorized |
|--------|-------------------|---------------------------|----------------------------|
| **Samples** | N_series × N_windows | N_windows | N_windows |
| **Sample shape** | (seq, features) | (seq, features_flat) | (n_series, seq, features) |
| **Batch shape** | (batch, seq, feat) | (batch, seq, feat_flat) | (batch_t, n_series, seq, feat) |
| **Exogenous features** | ✅ Yes | ✅ Yes | ❌ No (univariate only) |
| **Temporal encoding** | ✅ Yes | ✅ Yes | ❌ No |
| **One-hot encoding** | ✅ Yes | ❌ No | ❌ No |
| **Requires reshaping** | ❌ No | ❌ No | ✅ Yes |
| **Feature per timestep** | 1 + F + 2 + N | N × (1 + F) + 2 | 1 |
| **Forward passes** | Many | Few | Few |
| **GPU efficiency** | Low (5-15%) | Medium (40-60%) | High (80-95%) |
| **Memory efficiency** | Baseline | 240x better | 860x better |
| **Training speed** | Baseline | 20-30x faster | 20-50x faster |
| **Flexibility** | Maximum | High | Low (univariate only) |
| **Series length req** | Flexible | Must be equal | Must be equal |
| **Use case** | General purpose | Multi-series multivariate | Large-scale univariate |

---

## Part 12: Visual Comparison

### Sample Organization

```
TimeSeriesDataset: Series-first organization
┌─────────────────────────────────────────────┐
│ Sample 1:    Series A, Window 1             │
│ Sample 2:    Series A, Window 2             │
│ Sample 3:    Series A, Window 3             │
│ Sample 4:    Series B, Window 1             │
│ Sample 5:    Series B, Window 2             │
│ Sample 6:    Series B, Window 3             │
│ ...                                         │
│ Sample 53000: Series ZZZ, Window 53         │
└─────────────────────────────────────────────┘

TimeSeriesDatasetFlattened: Time-first organization
┌─────────────────────────────────────────────┐
│ Sample 1:  Window 1  [A, B, C, ..., ZZZ]    │
│ Sample 2:  Window 2  [A, B, C, ..., ZZZ]    │
│ Sample 3:  Window 3  [A, B, C, ..., ZZZ]    │
│ ...                                         │
│ Sample 53: Window 53 [A, B, C, ..., ZZZ]    │
└─────────────────────────────────────────────┘

TimeSeriesDatasetVectorized: Time-first vectorized
┌─────────────────────────────────────────────┐
│ Sample 1:  Window 1  [[A], [B], ..., [ZZZ]] │
│ Sample 2:  Window 2  [[A], [B], ..., [ZZZ]] │
│ Sample 3:  Window 3  [[A], [B], ..., [ZZZ]] │
│ ...                                         │
│ Sample 53: Window 53 [[A], [B], ..., [ZZZ]] │
└─────────────────────────────────────────────┘
```

### Batching Visualization

```
TimeSeriesDataset Batch (batch_size=32):
┌────────────────────────────────────────┐
│ [Series 42,  Window 10] ─┐             │
│ [Series 721, Window 33] ─┤             │
│ [Series 5,   Window 18] ─┤  32 random │
│ [Series 891, Window 7]  ─┤  samples   │
│ ...                      ─┤             │
│ [Series 156, Window 45] ─┘             │
└────────────────────────────────────────┘
Shape: (32, 6, 1006)

TimeSeriesDatasetFlattened Batch (batch_size=32):
┌────────────────────────────────────────┐
│ Window 5:  [S0, S1, ..., S999] ─┐      │
│ Window 22: [S0, S1, ..., S999] ─┤      │
│ Window 41: [S0, S1, ..., S999] ─┤  32  │
│ Window 3:  [S0, S1, ..., S999] ─┤ time │
│ ...                             ─┤ wins │
│ Window 17: [S0, S1, ..., S999] ─┘      │
└────────────────────────────────────────┘
Shape: (32, 6, 4002)

TimeSeriesDatasetVectorized Batch (batch_size=16):
┌────────────────────────────────────────┐
│ Window 10: [[S0], [S1], ..., [S999]]   │
│ Window 11: [[S0], [S1], ..., [S999]]   │
│ Window 12: [[S0], [S1], ..., [S999]]   │
│ Window 13: [[S0], [S1], ..., [S999]]   │
│ ...                                    │
│ Window 25: [[S0], [S1], ..., [S999]]   │
└────────────────────────────────────────┘
Shape: (16, 1000, 6, 1)
After reshape: (16000, 6, 1)
```

---

## Part 13: Code Example - Creating Each Dataset

### Setup
```python
import pandas as pd
from neuralts.core.func import (
    TimeSeriesDataset,
    TimeSeriesDatasetFlattened,
    TimeSeriesDatasetVectorized
)

# Example data: 1000 series, 60 timesteps each
df = pd.DataFrame({
    'Date': pd.date_range('2020-01', periods=60, freq='M').tolist() * 1000,
    'ts_key': [f'series_{i}' for i in range(1000) for _ in range(60)],
    'Value': np.random.randn(60000).cumsum() + 1000,
    'GDP': np.random.randn(60000) * 10 + 100,
    'CPI': np.random.randn(60000) * 5 + 105,
    'Interest_Rate': np.random.randn(60000) * 0.5 + 3.5
})

seq_length = 6
embargo = 1
batch_size = 32
```

### TimeSeriesDataset
```python
dataset = TimeSeriesDataset(
    df=df,
    feature_cols=['GDP', 'CPI', 'Interest_Rate'],
    seq_length=seq_length,
    embargo=embargo,
    train=True,
    train_ratio=0.8
)

print(f"Dataset length: {len(dataset)}")  # 53,000
print(f"Sample shape: {dataset[0][0].shape}")  # (6, 1006)

loader = DataLoader(dataset, batch_size=32, shuffle=True)
X_batch, y_batch = next(iter(loader))
print(f"Batch shape: {X_batch.shape}")  # (32, 6, 1006)
```

### TimeSeriesDatasetFlattened
```python
dataset = TimeSeriesDatasetFlattened(
    df=df,
    feature_cols=['GDP', 'CPI', 'Interest_Rate'],
    seq_length=seq_length,
    embargo=embargo,
    train=True,
    train_ratio=0.8
)

print(f"Dataset length: {len(dataset)}")  # 53
print(f"Sample shape: {dataset[0][0].shape}")  # (6, 4002)

loader = DataLoader(dataset, batch_size=32, shuffle=True)
X_batch, y_batch = next(iter(loader))
print(f"Batch shape: {X_batch.shape}")  # (21, 6, 4002) - last batch
```

### TimeSeriesDatasetVectorized
```python
# Must use only Date, ts_key, Value
df_univariate = df[['Date', 'ts_key', 'Value']]

dataset = TimeSeriesDatasetVectorized(
    df=df_univariate,
    seq_length=seq_length,
    embargo=embargo,
    train=True,
    train_ratio=0.8
)

print(f"Dataset length: {len(dataset)}")  # 53
print(f"Sample shape: {dataset[0][0].shape}")  # (1000, 6, 1)

loader = DataLoader(dataset, batch_size=16, shuffle=False)
X_batch, y_batch = next(iter(loader))
print(f"Batch shape: {X_batch.shape}")  # (16, 1000, 6, 1)
```

---

## Part 14: Training Loop Differences

### TimeSeriesDataset (Standard)
```python
for epoch in range(epochs):
    for X_batch, y_batch in train_loader:
        # X_batch: (32, 6, 1006)
        # y_batch: (32, 1)
        
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        
        predictions = model(X_batch)  # Direct forward pass
        loss = criterion(predictions, y_batch)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### TimeSeriesDatasetFlattened (Standard)
```python
for epoch in range(epochs):
    for X_batch, y_batch in train_loader:
        # X_batch: (batch, 6, 4002)
        # y_batch: (batch, 1000)
        
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        
        predictions = model(X_batch)  # Direct forward pass
        loss = criterion(predictions, y_batch)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### TimeSeriesDatasetVectorized (Requires Reshaping)
```python
for epoch in range(epochs):
    for X_batch, y_batch in train_loader:
        # X_batch: (16, 1000, 6, 1)
        # y_batch: (16, 1000)
        
        # CRITICAL STEP: Reshape before model
        batch_time, n_series, seq_len, n_feats = X_batch.shape
        X_reshaped = X_batch.view(batch_time * n_series, seq_len, n_feats)
        y_reshaped = y_batch.view(batch_time * n_series, 1)
        
        # X_reshaped: (16000, 6, 1)
        # y_reshaped: (16000, 1)
        
        X_reshaped = X_reshaped.to(device)
        y_reshaped = y_reshaped.to(device)
        
        predictions = model(X_reshaped)  # Forward pass with reshaped data
        loss = criterion(predictions, y_reshaped)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

---

## Part 15: Summary Decision Matrix

### Quick Reference Guide

| Your Situation | Best Choice |
|----------------|-------------|
| Small dataset (< 100 series) | TimeSeriesDataset |
| Need different features per series | TimeSeriesDataset |
| Many series + need exogenous features | TimeSeriesDatasetFlattened |
| Large dataset + multivariate forecasting | TimeSeriesDatasetFlattened |
| **Large dataset + univariate only** | **TimeSeriesDatasetVectorized** |
| Need maximum training speed | TimeSeriesDatasetVectorized |
| Following Nixtla approach | TimeSeriesDatasetVectorized |
| Series have different lengths | TimeSeriesDataset (with padding) |
| Limited GPU memory | TimeSeriesDataset or Flattened |
| Abundant GPU memory + many series | TimeSeriesDatasetVectorized |

### Performance vs Flexibility Trade-off

```
High Flexibility ←──────────────────────→ High Performance
                                          
TimeSeriesDataset    TimeSeriesDatasetFlattened    TimeSeriesDatasetVectorized
        │                       │                              │
        │                       │                              │
   [Slowest but               [Balanced                   [Fastest but
    most flexible]            approach]                  most restrictive]
        │                       │                              │
   53,000 samples            53 samples                   53 samples
   1,656 forward passes      2 forward passes             4 forward passes
   Full feature support      No one-hot                   Univariate only
   5-15% GPU                 40-60% GPU                   80-95% GPU
```

---

## Conclusion

The three dataset approaches represent different points on the flexibility-performance spectrum:

- **TimeSeriesDataset**: Maximum flexibility, familiar patterns, but slowest
- **TimeSeriesDatasetFlattened**: Good balance, eliminates one-hot overhead, supports multivariate
- **TimeSeriesDatasetVectorized**: Maximum performance, Nixtla-style efficiency, univariate only

Choose based on your specific requirements: feature complexity vs. computational efficiency.
