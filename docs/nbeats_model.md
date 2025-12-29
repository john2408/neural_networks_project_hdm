# NBEATS Model

## Overview
**NBEATS (Neural Basis Expansion Analysis for Time Series)** is a deep learning architecture specifically designed for univariate time series forecasting. Implemented using the NeuralForecast package from Nixtla, it forecasts multiple individual series independently by decomposing the time series into interpretable components (trend and seasonality) using basis functions.

## Architecture

```
Input Sequence (seq_length timesteps)
         ↓
    ┌────────────────────────────────┐
    │   Stack 1: Trend Block(s)      │
    │   - Polynomial basis functions  │
    │   - MLPs for basis coefficients │
    │   - Backcast + Forecast         │
    └────────────────────────────────┘
         ↓ (residual connection)
    ┌────────────────────────────────┐
    │   Stack 2: Seasonality Block(s)│
    │   - Fourier basis functions     │
    │   - MLPs for basis coefficients │
    │   - Backcast + Forecast         │
    └────────────────────────────────┘
         ↓ (residual connection)
    ┌────────────────────────────────┐
    │   Stack 3: Generic Block(s)    │
    │   - Learnable basis functions   │
    │   - MLPs for basis coefficients │
    │   - Backcast + Forecast         │
    └────────────────────────────────┘
         ↓
    Final Forecast (horizon)
```

## Model Structure

### Block Architecture
Each block in NBEATS consists of:
1. **Fully Connected Layers (MLP)**: Process input sequence
2. **Basis Expansion**: Decompose signal using basis functions
   - **Trend Stack**: Polynomial basis for long-term patterns
   - **Seasonality Stack**: Fourier basis (harmonics) for periodic patterns
   - **Generic Stack**: Learnable basis for residual patterns
3. **Doubly Residual Stacking**: 
   - **Backcast**: Reconstructs input to remove explained patterns
   - **Forecast**: Predicts future values
   - Residual passed to next block

### Univariate Approach
- Each time series is modeled **independently**
- No cross-series information sharing
- Scalable to thousands of series (parallel processing)
- Robust scaler applied per series

## Advantages

✅ **Interpretable Decomposition**: Separates trend, seasonality, and residual components  
✅ **Pure Forecasting Architecture**: Designed specifically for time series (no adaptations from other domains)  
✅ **No Manual Feature Engineering**: Automatically learns basis functions  
✅ **Double Residual Learning**: Efficiently captures complex patterns through stacking  
✅ **Univariate Scalability**: Processes multiple series independently without memory overhead

## Limitations

⚠️ **No Exogenous Variables**: NBEATS does not support external features (use NBEATSx for exogenous support)  
⚠️ **Univariate Only**: Cannot leverage cross-series information or correlations  
⚠️ **Fixed Architecture Stacks**: Requires careful tuning of stack types and block counts  
⚠️ **Longer Training**: Multiple stacks and basis expansions increase computational cost

## When to Use

**Ideal for:**
- Univariate time series with **clear trend and seasonal patterns**
- Datasets with **hundreds to thousands of independent series**
- When **interpretability** of forecast components is important
- Medium to long sequences (10-100+ timesteps)
- When cross-series relationships are not relevant

**Not recommended for:**
- Series requiring exogenous variables (use NBEATSx instead)
- Very short sequences (<6 timesteps)
- When multivariate dependencies are critical

## Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `dropout_prob_theta` | 0.5 | Dropout probability applied to the MLP layers that compute basis coefficients (regularization to prevent overfitting) |
| `max_steps` | 500 | Maximum number of training steps (epochs) for model optimization |
| `n_harmonics` | 2 | Number of Fourier harmonic terms used in the seasonality stack to capture periodic patterns |
| `n_polynomials` (n_basis) | 2 | Degree of polynomial basis functions used in the trend stack to model long-term trends |
| `n_blocks` | [1, 1, 1] | Number of blocks per stack (trend, seasonality, generic) – controls model depth and capacity |
| `mlp_units` | [[512, 512], [512, 512], [512, 512]] | Hidden layer sizes for each stack's MLP – defines the width of fully connected layers processing the input |
| `input_size` | 6 | Number of historical timesteps used as input (lookback window) |
| `h` | 3 | Forecast horizon (number of future timesteps to predict) |
| `scaler_type` | 'robust' | Type of scaler for input normalization (robust scaler is less sensitive to outliers) |
| `loss` | MAE | Loss function for training (Mean Absolute Error for robust forecasting) |

### Stack Configuration
- **Stack 1 (Trend)**: Uses `n_polynomials` polynomial basis functions
- **Stack 2 (Seasonality)**: Uses `n_harmonics` Fourier basis functions
- **Stack 3 (Generic)**: Uses learnable basis functions

### MLP Architecture
Each stack has an MLP defined by `mlp_units`:
- Example: `[[512, 512], [512, 512], [512, 512]]` creates 3 stacks
- Each inner list defines hidden layers for that stack's MLP
- `[512, 512]` = 2 hidden layers with 512 units each

## Implementation Details

### NeuralForecast Package (Nixtla)
```python
from neuralforecast import NeuralForecast
from neuralforecast.models import NBEATS

model = NBEATS(
    input_size=6,
    h=3,
    loss=MAE(),
    dropout_prob_theta=0.5,
    max_steps=500,
    n_harmonics=2,
    n_polynomials=2,
    n_blocks=[1, 1, 1],
    mlp_units=[[512, 512], [512, 512], [512, 512]],
    scaler_type='robust',
    random_seed=42
)
```

### Data Format
NeuralForecast expects data in long format:
- `unique_id`: Time series identifier
- `ds`: Date/timestamp column
- `y`: Target variable (values to forecast)

### Forecasting Process
1. Each series is scaled independently using robust scaler
2. Model processes series in parallel (batched by unique_id)
3. Forecasts are generated for all series simultaneously
4. Predictions are inverse-transformed to original scale
