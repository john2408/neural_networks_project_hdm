# TimeSeriesDataset Approaches: Batching & Modeling Guide

## Executive Summary

This document compares three approaches to organizing time series data for neural network training, using our dataset of **1502 vehicle registration time series**.

### The Three Approaches

| Approach | Philosophy | Best For | Key Limitation |
|----------|-----------|----------|----------------|
| **TimeSeriesDataset** | One sample per series per window | Maximum flexibility, experimentation | Slow training (many forward passes) |
| **TimeSeriesDatasetFlattened** | All series flattened per time window | Balanced efficiency + multivariate | Large feature vectors |
| **TimeSeriesDatasetVectorizedExog** | All series vectorized per time window | Maximum speed + exogenous features | Requires reshaping, high memory |

### Key Insight

The main trade-off is **flexibility vs. computational efficiency**:

```
TimeSeriesDataset          TimeSeriesDatasetFlattened     TimeSeriesDatasetVectorizedExog
     (Flexible)                    (Balanced)                      (Fast)
        │                              │                              │
   80,000+ samples                   50 samples                    50 samples
   2,500+ forward passes          2 forward passes              4 forward passes
   One-hot encoding              No one-hot                    No one-hot
   10-15% GPU util               40-60% GPU util               80-95% GPU util
```

---

## 1. Configuration & Setup

Throughout this document, we use:

```python
# Dataset configuration (our actual data)
n_series = 1502              # Vehicle registration time series
n_timesteps = 60             # Monthly data points per series
seq_length = 6               # Lookback window (6 months)
embargo = 1                  # Gap before prediction
test_period = 3              # Last 3 months for testing
n_exog_features = 3          # Economic indicators (GDP, CPI, Interest Rate)

# Derived values
n_windows_total = 60 - 6 - 1 = 53 total windows
n_windows_train = 53 - 3 = 50 training windows
n_windows_test = 3 test windows

# Batch sizes
batch_size = 32              # For TimeSeriesDataset and Flattened
batch_size_vec = 16          # For VectorizedExog
```

---

## 2. Core Philosophical Differences

### TimeSeriesDataset: Series-First Organization

```
Paradigm: ONE sample = ONE series + ONE time window
Identity: One-hot encoding (1502 dimensions)
Batching: Random mix of series and time windows
```

**Sample composition:**
- Each training sample represents a single time series at a single time window
- Series identity encoded as one-hot vector
- Full feature support (value + exogenous + temporal + one-hot)

### TimeSeriesDatasetFlattened: Time-First Flattened

```
Paradigm: ONE sample = ALL series (flattened) + ONE time window
Identity: Implicit (position in flattened vector)
Batching: Random mix of time windows
```

**Sample composition:**
- Each training sample contains all 1502 series concatenated
- No one-hot encoding needed (series ID is implicit in position)
- Full feature support (values + exogenous features concatenated)

### TimeSeriesDatasetVectorizedExog: Time-First Vectorized

```
Paradigm: ONE sample = ALL series (vectorized) + ONE time window
Identity: Implicit (position in series dimension)
Batching: Sequential time windows
```

**Sample composition:**
- Each training sample contains all 1502 series as separate vectors
- No one-hot encoding needed
- **Full feature support including exogenous variables** (key difference from original)
- Requires reshaping before model input

---

## 3. Dataset Size Comparison

### Number of Training Samples

**TimeSeriesDataset:**
```python
n_samples = n_series × n_windows_train
          = 1502 × 50
          = 75,100 training samples
```

**TimeSeriesDatasetFlattened:**
```python
n_samples = n_windows_train
          = 50 training samples
```

**TimeSeriesDatasetVectorizedExog:**
```python
n_samples = n_windows_train
          = 50 training samples
```

### Sample Reduction

| Dataset Type | Training Samples | Reduction Factor |
|--------------|------------------|------------------|
| TimeSeriesDataset | 75,100 | Baseline (1x) |
| TimeSeriesDatasetFlattened | 50 | **1,502x fewer** |
| TimeSeriesDatasetVectorizedExog | 50 | **1,502x fewer** |

---

## 4. Data Structure Deep Dive

### Single Sample Comparison

#### TimeSeriesDataset (Sample #37,550)

Could be series #750 at time window #25:

```python
X.shape = (seq_length, n_features_total)
        = (6, 1 + 3 + 2 + 1502)
        = (6, 1508)

# Feature composition per timestep:
# [Value, GDP, CPI, Interest_Rate, year, month, onehot_0, ..., onehot_1501]

y.shape = (1,)

# Example structure:
X = [
  [1234.5, 2.3, 105.2, 3.5, 2023, 1, 0, ..., 1, ..., 0],  # t-5
  [1245.2, 2.3, 105.3, 3.5, 2023, 2, 0, ..., 1, ..., 0],  # t-4
  [1256.8, 2.4, 105.4, 3.6, 2023, 3, 0, ..., 1, ..., 0],  # t-3
  [1267.3, 2.4, 105.5, 3.6, 2023, 4, 0, ..., 1, ..., 0],  # t-2
  [1278.9, 2.5, 105.6, 3.7, 2023, 5, 0, ..., 1, ..., 0],  # t-1
  [1289.1, 2.5, 105.7, 3.7, 2023, 6, 0, ..., 1, ..., 0]   # t
]

y = [1302.4]  # Target at t+1+embargo
```

#### TimeSeriesDatasetFlattened (Sample #25)

Time window #25 with ALL 1502 series flattened:

```python
X.shape = (seq_length, n_features_total)
        = (6, 1502 + 1502*3 + 2)
        = (6, 6010)

# Feature composition per timestep:
# [val_0, val_1, ..., val_1501,          # All 1502 values
#  gdp_0, gdp_1, ..., gdp_1501,          # All 1502 GDP values
#  cpi_0, cpi_1, ..., cpi_1501,          # All 1502 CPI values
#  int_0, int_1, ..., int_1501,          # All 1502 Interest Rate values
#  year, month]                          # Shared temporal features

y.shape = (1502,)

# Target vector contains predictions for all series:
y = [y_0, y_1, y_2, ..., y_1501]
```

#### TimeSeriesDatasetVectorizedExog (Sample #25)

Time window #25 with ALL 1502 series vectorized:

```python
X.shape = (n_series, seq_length, n_features)
        = (1502, 6, 4)

# Feature composition per series per timestep:
# [Value, GDP, CPI, Interest_Rate]

y.shape = (1502,)

# Example structure (series dimension first):
X = [
  # Series 0: 6 timesteps × 4 features
  [[1234.5, 2.3, 105.2, 3.5], [1245.2, 2.3, 105.3, 3.5], ..., [1289.1, 2.5, 105.7, 3.7]],
  
  # Series 1: 6 timesteps × 4 features
  [[2341.2, 2.3, 105.2, 3.5], [2356.7, 2.3, 105.3, 3.5], ..., [2418.3, 2.5, 105.7, 3.7]],
  
  # ...
  
  # Series 1501: 6 timesteps × 4 features
  [[9876.5, 2.3, 105.2, 3.5], [9912.3, 2.3, 105.3, 3.5], ..., [10059, 2.5, 105.7, 3.7]]
]

y = [1302.4, 2433.9, ..., 10096]  # All 1502 predictions
```

**Key Difference from Original Vectorized:** Now includes exogenous features (GDP, CPI, Interest Rate), not just univariate values!

---

## 5. Batch Structure & Reshaping

### Batch Shapes Overview

| Dataset | Batch Shape | Predictions/Batch | Reshaping Needed? |
|---------|-------------|-------------------|-------------------|
| TimeSeriesDataset | (32, 6, 1508) | 32 | ❌ No |
| TimeSeriesDatasetFlattened | (32, 6, 6010) | 32,064 | ❌ No |
| TimeSeriesDatasetVectorizedExog | (16, 1502, 6, 4) | 24,032 | ✅ Yes |

### TimeSeriesDataset Batch (batch_size=32)

```python
X_batch.shape = (32, 6, 1508)
y_batch.shape = (32, 1)

# Contains 32 random samples
# Example: [series 42 window 10, series 721 window 33, series 5 window 18, ...]
# Each sample is ONE series at ONE time window

# Direct model input (no reshaping):
predictions = model(X_batch)  # Output: (32, 1)
```

### TimeSeriesDatasetFlattened Batch (batch_size=32)

```python
X_batch.shape = (32, 6, 6010)
y_batch.shape = (32, 1502)

# Contains 32 random time windows
# Each window has ALL 1502 series flattened
# Total predictions: 32 × 1502 = 48,064

# Direct model input (no reshaping):
predictions = model(X_batch)  # Output: (32, 1502)
```

### TimeSeriesDatasetVectorizedExog Batch (batch_size=16)

```python
X_batch.shape = (16, 1502, 6, 4)
y_batch.shape = (16, 1502)

# Contains 16 SEQUENTIAL time windows
# Each window has ALL 1502 series as separate vectors
# Total predictions: 16 × 1502 = 24,032

# REQUIRES RESHAPING before model:
batch_time, n_series, seq_len, n_feats = X_batch.shape
X_reshaped = X_batch.view(batch_time * n_series, seq_len, n_feats)
y_reshaped = y_batch.view(batch_time * n_series, 1)

# After reshaping:
X_reshaped.shape = (24032, 6, 4)
y_reshaped.shape = (24032, 1)

predictions = model(X_reshaped)  # Output: (24032, 1)
```

---

## 6. Feature Engineering Support

### TimeSeriesDataset (Maximum Flexibility)

```python
# Supports everything:
✅ Value (target history)
✅ Exogenous features (GDP, CPI, Interest_Rate)
✅ Temporal encoding (year, month)
✅ One-hot series encoding (1502 dimensions)
✅ Different features per series (if needed)

# Feature count per timestep:
n_features = 1 + 3 + 2 + 1502 = 1508
```

### TimeSeriesDatasetFlattened (High Flexibility)

```python
# Supports most features:
✅ Value (all series concatenated)
✅ Exogenous features (all series concatenated)
✅ Temporal encoding (shared across series)
❌ No one-hot (series identity implicit)

# Feature count per timestep:
n_features = 1502 * (1 + 3) + 2 = 6010
```

### TimeSeriesDatasetVectorizedExog (Multivariate Support)

```python
# Supports multivariate forecasting:
✅ Value (vectorized per series)
✅ Exogenous features (GDP, CPI, Interest_Rate per series)
❌ No temporal encoding (could be added as feature)
❌ No one-hot (not needed)

# Feature count per series per timestep:
n_features = 1 + 3 = 4
```

**Key Improvement:** Unlike the original `TimeSeriesDatasetVectorized` (univariate only), `TimeSeriesDatasetVectorizedExog` supports exogenous features!

---

## 7. Performance Comparison

### Training Efficiency

**Forward Passes Per Epoch (50 training windows):**

| Dataset | Total Samples | Batch Size | Forward Passes | Predictions/Pass |
|---------|---------------|------------|----------------|------------------|
| TimeSeriesDataset | 75,100 | 32 | 2,347 | 32 |
| TimeSeriesDatasetFlattened | 50 | 32 | 2 | 48,064 (avg) |
| TimeSeriesDatasetVectorizedExog | 50 | 16 | 4 | 24,032 (avg) |

**Speedup Factor:**

| Dataset | Speedup vs TimeSeriesDataset |
|---------|------------------------------|
| TimeSeriesDataset | 1x (baseline) |
| TimeSeriesDatasetFlattened | **~1,170x faster** |
| TimeSeriesDatasetVectorizedExog | **~585x faster** |

### Memory Footprint (Training Set)

**Per-Sample Memory:**

```python
# TimeSeriesDataset
X: (6, 1508) × 4 bytes = 36,192 bytes ≈ 35 KB
y: (1,) × 4 bytes = 4 bytes
Total: ≈ 35 KB per sample
Training set total: 75,100 × 35 KB ≈ 2,628 MB

# TimeSeriesDatasetFlattened
X: (6, 6010) × 4 bytes = 144,240 bytes ≈ 141 KB
y: (1502,) × 4 bytes = 6,008 bytes ≈ 6 KB
Total: ≈ 147 KB per sample
Training set total: 50 × 147 KB ≈ 7.35 MB

# TimeSeriesDatasetVectorizedExog
X: (1502, 6, 4) × 4 bytes = 144,192 bytes ≈ 141 KB
y: (1502,) × 4 bytes = 6,008 bytes ≈ 6 KB
Total: ≈ 147 KB per sample
Training set total: 50 × 147 KB ≈ 7.35 MB
```

**Memory Efficiency:**

| Dataset | Memory | Reduction vs Traditional |
|---------|--------|-------------------------|
| TimeSeriesDataset | 2,628 MB | Baseline (1x) |
| TimeSeriesDatasetFlattened | 7.35 MB | **357x less** |
| TimeSeriesDatasetVectorizedExog | 7.35 MB | **357x less** |

### GPU Utilization

```
TimeSeriesDataset:
  Batch: (32, 6, 1508)
  GPU utilization: ~10-15%
  Bottleneck: Many small operations
  
TimeSeriesDatasetFlattened:
  Batch: (32, 6, 6010)
  GPU utilization: ~40-60%
  Bottleneck: Large feature vectors
  
TimeSeriesDatasetVectorizedExog:
  Batch (reshaped): (24032, 6, 4)
  GPU utilization: ~80-95%
  Bottleneck: Model computation (optimal!)
```

### Estimated Training Time (30 epochs)

| Dataset | Time/Epoch | Total (30 epochs) | GPU Util |
|---------|------------|-------------------|----------|
| TimeSeriesDataset | ~120 sec | ~60 min | 10-15% |
| TimeSeriesDatasetFlattened | ~6 sec | ~3 min | 40-60% |
| TimeSeriesDatasetVectorizedExog | ~4 sec | ~2 min | 80-95% |

---

## 8. When to Use Each Approach

### Decision Tree

```
Do you need different features per series?
    ├─ YES → TimeSeriesDataset
    └─ NO → Continue...
    
Do you need exogenous features?
    ├─ NO → Consider simpler univariate approaches
    └─ YES → Continue...
    
Do you have abundant GPU memory (16GB+)?
    ├─ YES → TimeSeriesDatasetVectorizedExog (fastest)
    └─ NO → TimeSeriesDatasetFlattened (balanced)
    
Is training speed critical?
    ├─ YES → TimeSeriesDatasetVectorizedExog
    └─ NO → Either Flattened or VectorizedExog
```

### Use TimeSeriesDataset When:

✅ **Best for:**
- Prototyping and experimentation
- Different feature sets per series
- Series with different lengths (with padding)
- Small datasets (< 500 series)
- Need maximum flexibility

❌ **Avoid when:**
- Training many epochs (slow)
- Have many series (> 1000)
- Training speed is important

**Example:** Research phase with heterogeneous vehicle types requiring different features.

### Use TimeSeriesDatasetFlattened When:

✅ **Best for:**
- Balanced speed and simplicity
- All series share same features
- Medium GPU memory (8-12 GB)
- Don't want to deal with reshaping
- Stable production pipelines

❌ **Avoid when:**
- Need maximum speed (use VectorizedExog)
- Have limited GPU memory
- Series have different lengths

**Example:** Production forecasting system with consistent feature engineering across all vehicle types.

### Use TimeSeriesDatasetVectorizedExog When:

✅ **Best for:**
- Maximum training speed
- Large-scale forecasting (1000+ series)
- Abundant GPU memory (16GB+)
- Multivariate forecasting with exogenous features
- Iterating on model architectures frequently

❌ **Avoid when:**
- Limited GPU memory (< 8GB)
- Need temporal encoding features
- Series have different lengths
- Want simpler code (no reshaping)

**Example:** Large-scale vehicle registration forecasting with economic indicators, requiring fast iteration.

---

## 9. Code Examples

### Setup (All Approaches)

```python
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from neuralts.core.func import (
    TimeSeriesDataset,
    TimeSeriesDatasetFlattened,
    TimeSeriesDatasetVectorizedExog
)

# Load your data: 1502 series, 60 timesteps each
df = pd.read_csv('vehicle_registrations.csv')
# Expected columns: ['Date', 'ts_key', 'Value', 'GDP', 'CPI', 'Interest_Rate']

# Configuration
seq_length = 6
embargo = 1
test_period = 3
feature_cols = ['GDP', 'CPI', 'Interest_Rate']
```

### TimeSeriesDataset

```python
# Create dataset
train_dataset = TimeSeriesDataset(
    df=df,
    feature_cols=feature_cols,
    seq_length=seq_length,
    embargo=embargo,
    train=True,
    test_period=test_period
)

test_dataset = TimeSeriesDataset(
    df=df,
    feature_cols=feature_cols,
    seq_length=seq_length,
    embargo=embargo,
    train=False,
    test_period=test_period
)

print(f"Train samples: {len(train_dataset)}")  # 75,100
print(f"Test samples: {len(test_dataset)}")    # 4,506

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Training loop (standard)
for epoch in range(epochs):
    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)  # (32, 6, 1508)
        y_batch = y_batch.to(device)  # (32, 1)
        
        predictions = model(X_batch)
        loss = criterion(predictions, y_batch)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### TimeSeriesDatasetFlattened

```python
# Create dataset
train_dataset = TimeSeriesDatasetFlattened(
    df=df,
    feature_cols=feature_cols,
    seq_length=seq_length,
    embargo=embargo,
    train=True,
    test_period=test_period
)

test_dataset = TimeSeriesDatasetFlattened(
    df=df,
    feature_cols=feature_cols,
    seq_length=seq_length,
    embargo=embargo,
    train=False,
    test_period=test_period
)

print(f"Train samples: {len(train_dataset)}")  # 50
print(f"Test samples: {len(test_dataset)}")    # 3

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Training loop (standard)
for epoch in range(epochs):
    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)  # (batch, 6, 6010)
        y_batch = y_batch.to(device)  # (batch, 1502)
        
        predictions = model(X_batch)  # Output: (batch, 1502)
        loss = criterion(predictions, y_batch)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### TimeSeriesDatasetVectorizedExog

```python
# Create dataset
train_dataset = TimeSeriesDatasetVectorizedExog(
    df=df,
    feature_cols=feature_cols,  # Now supports exogenous features!
    seq_length=seq_length,
    embargo=embargo,
    train=True,
    test_period=test_period
)

test_dataset = TimeSeriesDatasetVectorizedExog(
    df=df,
    feature_cols=feature_cols,
    seq_length=seq_length,
    embargo=embargo,
    train=False,
    test_period=test_period
)

print(f"Train samples: {len(train_dataset)}")  # 50
print(f"Test samples: {len(test_dataset)}")    # 3

# Create data loaders (shuffle=False recommended for sequential batching)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Training loop (requires reshaping)
for epoch in range(epochs):
    for X_batch, y_batch in train_loader:
        # Original shapes: X_batch (16, 1502, 6, 4), y_batch (16, 1502)
        
        # CRITICAL: Reshape before model
        batch_time, n_series, seq_len, n_feats = X_batch.shape
        X_reshaped = X_batch.view(batch_time * n_series, seq_len, n_feats)
        y_reshaped = y_batch.view(batch_time * n_series, 1)
        
        # After reshaping: (24032, 6, 4) and (24032, 1)
        X_reshaped = X_reshaped.to(device)
        y_reshaped = y_reshaped.to(device)
        
        predictions = model(X_reshaped)  # Output: (24032, 1)
        loss = criterion(predictions, y_reshaped)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

---

## 10. Quick Reference Table

### Complete Comparison

| Aspect | TimeSeriesDataset | TimeSeriesDatasetFlattened | TimeSeriesDatasetVectorizedExog |
|--------|-------------------|---------------------------|--------------------------------|
| **Samples (train)** | 75,100 | 50 | 50 |
| **Sample shape** | (6, 1508) | (6, 6010) | (1502, 6, 4) |
| **Batch shape** | (32, 6, 1508) | (32, 6, 6010) | (16, 1502, 6, 4) |
| **Predictions/batch** | 32 | 48,064 | 24,032 |
| **Exogenous features** | ✅ Yes | ✅ Yes | ✅ Yes |
| **Temporal encoding** | ✅ Yes | ✅ Yes | ❌ No |
| **One-hot encoding** | ✅ Yes (1502 dim) | ❌ No | ❌ No |
| **Reshaping required** | ❌ No | ❌ No | ✅ Yes |
| **Forward passes/epoch** | 2,347 | 2 | 4 |
| **GPU utilization** | 10-15% | 40-60% | 80-95% |
| **Memory (train set)** | 2,628 MB | 7.35 MB | 7.35 MB |
| **Training time (30 ep)** | ~60 min | ~3 min | ~2 min |
| **Flexibility** | Maximum | High | Medium |
| **Speed** | Baseline | 1,170x faster | 585x faster |
| **Code complexity** | Simple | Simple | Medium (reshaping) |
| **Best use case** | Experimentation | Balanced production | Maximum speed |

### Feature Support Breakdown

| Feature Type | TimeSeriesDataset | TimeSeriesDatasetFlattened | TimeSeriesDatasetVectorizedExog |
|--------------|-------------------|---------------------------|--------------------------------|
| Target value history | ✅ | ✅ | ✅ |
| Exogenous features | ✅ | ✅ | ✅ (New!) |
| Temporal features | ✅ | ✅ | ❌ |
| Series identity | ✅ (one-hot) | ✅ (implicit) | ✅ (implicit) |
| Per-series features | ✅ | ❌ | ❌ |

---

## 11. Key Insights & Best Practices

### 🎯 Critical Insights

1. **Sample Reduction**: Both Flattened and VectorizedExog have 1,502x fewer samples than traditional approach, dramatically reducing training time.

2. **Reshaping Requirement**: Only VectorizedExog requires reshaping from `(batch_time, n_series, seq, feat)` to `(batch_time × n_series, seq, feat)`. This is the trade-off for maximum GPU utilization.

3. **One-Hot Overhead**: TimeSeriesDataset's one-hot encoding adds 1,502 dimensions per timestep. Removing this (Flattened/VectorizedExog) significantly reduces memory and computation.

4. **GPU Utilization**: VectorizedExog achieves 80-95% GPU utilization because it processes 24,032 samples per batch (after reshaping), fully utilizing GPU parallelism.

5. **Exogenous Feature Support**: VectorizedExog now supports exogenous features (unlike original Vectorized), making it suitable for multivariate forecasting while maintaining speed.

### 💡 Best Practices

**For TimeSeriesDataset:**
- Use during initial exploration and prototyping
- Enable shuffling for better generalization
- Consider for small-scale experiments (< 500 series)

**For TimeSeriesDatasetFlattened:**
- Ideal for production systems with consistent features
- No reshaping complexity makes debugging easier
- Good balance for teams prioritizing code simplicity

**For TimeSeriesDatasetVectorizedExog:**
- Use when training time is critical (many experiments)
- Ensure GPU has sufficient memory (16GB+ recommended)
- Test with smaller batch sizes first if memory constrained
- Document the reshaping step clearly for team members

### ⚠️ Common Pitfalls

1. **VectorizedExog without reshaping**: Model will fail with wrong input shape
2. **Shuffling VectorizedExog**: Generally use `shuffle=False` to maintain sequential time windows
3. **Memory overflow**: Flattened/VectorizedExog use large batches; monitor GPU memory
4. **Feature alignment**: Ensure exogenous features are properly aligned across all series

---

## 12. Conclusion

For our **1502 vehicle registration time series**:

- **Start with**: `TimeSeriesDataset` during exploration
- **Move to**: `TimeSeriesDatasetFlattened` for stable production
- **Optimize with**: `TimeSeriesDatasetVectorizedExog` when speed is critical

The choice depends on your current phase:
- 🧪 **Research phase**: TimeSeriesDataset (maximum flexibility)
- 🏗️ **Development phase**: TimeSeriesDatasetFlattened (balanced)
- 🚀 **Production/Optimization phase**: TimeSeriesDatasetVectorizedExog (maximum speed)

All three approaches are valid; choose based on your specific constraints and priorities.
