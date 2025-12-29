# NBEATSx Model

## Overview
**NBEATSx (Neural Basis Expansion Analysis for Time Series with Exogenous Variables)** is an extension of NBEATS that incorporates exogenous variables for improved forecasting accuracy. Implemented using the NeuralForecast package from Nixtla, it combines the interpretable basis expansion architecture of NBEATS with the ability to leverage external features like economic indicators, enabling more informed predictions across multiple time series.

## Architecture

```
Input Sequence (seq_length timesteps) + Static Exog + Future Exog
         ↓
    ┌────────────────────────────────────────────┐
    │   Stack 1: Trend Block(s)                  │
    │   - Polynomial basis functions              │
    │   - MLPs for basis coefficients             │
    │   - Static exogenous features injected      │
    │   - Future exogenous features injected      │
    │   - Backcast + Forecast                     │
    └────────────────────────────────────────────┘
         ↓ (residual connection)
    ┌────────────────────────────────────────────┐
    │   Stack 2: Seasonality Block(s)            │
    │   - Fourier basis functions                 │
    │   - MLPs for basis coefficients             │
    │   - Static + Future exog features injected  │
    │   - Backcast + Forecast                     │
    └────────────────────────────────────────────┘
         ↓ (residual connection)
    ┌────────────────────────────────────────────┐
    │   Stack 3: Generic Block(s)                │
    │   - Learnable basis functions               │
    │   - MLPs for basis coefficients             │
    │   - Static + Future exog features injected  │
    │   - Backcast + Forecast                     │
    └────────────────────────────────────────────┘
         ↓
    Final Forecast (horizon)
```

## Model Structure

### Block Architecture with Exogenous Features
Each block in NBEATSx extends the NBEATS architecture by incorporating:
1. **Static Exogenous Features**: Time-invariant characteristics (e.g., series identifiers, categorical attributes)
2. **Future Exogenous Features**: Known future values (e.g., economic indicators, calendar features)
3. **Basis Expansion**: Same as NBEATS (Trend, Seasonality, Generic stacks)
4. **Feature Fusion**: Exogenous features are concatenated with basis coefficients before prediction

### Our Implementation Details

#### Static Exogenous Features (stat_exog_list)
- **One-hot encoded series IDs**: Each time series is uniquely identified via binary encoding
- **Purpose**: Allows the model to learn series-specific patterns and behaviors
- **Example**: For 1502 series, creates 1502 binary features (one per series)

#### Future Exogenous Features (futr_exog_list)
In our vehicle registration forecasting system:
- **GDP**: Gross Domestic Product (economic growth indicator)
- **CPI**: Consumer Price Index (inflation measure)
- **Interest_Rate**: Central bank interest rate (monetary policy indicator)

These macroeconomic variables are shared across all time series and influence purchasing behavior.

## Advantages

✅ **Incorporates External Information**: Leverages exogenous variables for improved accuracy  
✅ **Series-Specific Learning**: One-hot encoding enables individual series customization  
✅ **Interpretable Decomposition**: Maintains NBEATS' trend/seasonality separation  
✅ **Economic Context**: Can model how macroeconomic factors impact forecasts  
✅ **Univariate Scalability**: Processes multiple series independently with shared exogenous features

## Limitations

⚠️ **Requires Future Exogenous Values**: Must have known future values of external features at prediction time  
⚠️ **Increased Complexity**: More hyperparameters and features increase training time and tuning difficulty  
⚠️ **Static Feature Overhead**: One-hot encoding creates high-dimensional input for many series  
⚠️ **No Cross-Series Dependencies**: Still treats each series independently (no multivariate correlations)

## When to Use

**Ideal for:**
- Time series with **available exogenous features** (economic indicators, weather, calendar effects)
- Forecasting problems where **external factors** significantly influence outcomes
- Multiple series with **shared external drivers** but individual characteristics
- Medium to long sequences (10-100+ timesteps) with **known future covariates**

**Not recommended for:**
- When future exogenous values are unavailable
- Very short sequences (<6 timesteps)
- When pure univariate NBEATS performs adequately
- When cross-series dependencies are more important than exogenous features

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
| `stat_exog_list` | [one-hot series IDs] | List of static exogenous feature names (time-invariant characteristics) |
| `futr_exog_list` | ['GDP', 'CPI', 'Interest_Rate'] | List of future exogenous feature names (known future values) |

### Stack Configuration
Same as NBEATS:
- **Stack 1 (Trend)**: Uses `n_polynomials` polynomial basis functions
- **Stack 2 (Seasonality)**: Uses `n_harmonics` Fourier basis functions
- **Stack 3 (Generic)**: Uses learnable basis functions

### Exogenous Feature Integration
- **Static features**: Concatenated to each block's input (same value across all timesteps)
- **Future features**: Concatenated to each timestep's representation (time-varying)
- **Feature scaling**: Applied automatically via `scaler_type='robust'`

## Implementation Details

### NeuralForecast Package (Nixtla)
```python
from neuralforecast import NeuralForecast
from neuralforecast.models import NBEATSx

# Create static dataframe with one-hot encoded series IDs
unique_ids = df['unique_id'].unique()
static_df = pd.DataFrame({'unique_id': unique_ids})
for uid in unique_ids:
    static_df[f'id_{uid}'] = (static_df['unique_id'] == uid).astype(int)

stat_exog_list = [f'id_{uid}' for uid in unique_ids]
futr_exog_list = ['GDP', 'CPI', 'Interest_Rate']

model = NBEATSx(
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
    stat_exog_list=stat_exog_list,
    futr_exog_list=futr_exog_list,
    random_seed=42
)
```

### Data Format
NeuralForecast expects data in long format with exogenous features:

**Training Data:**
- `unique_id`: Time series identifier
- `ds`: Date/timestamp column
- `y`: Target variable (values to forecast)
- `GDP`, `CPI`, `Interest_Rate`: Exogenous features

**Static Data (separate DataFrame):**
- `unique_id`: Time series identifier
- `id_{series_1}`, `id_{series_2}`, ...: One-hot encoded series identifiers

**Future Data (for prediction):**
- `unique_id`: Time series identifier
- `ds`: Future dates to forecast
- `GDP`, `CPI`, `Interest_Rate`: Known future exogenous values

### Forecasting Process
1. **Training**: Fit model with training data (`df`) and static features (`static_df`)
2. **Validation**: Use `val_size` to reserve recent data for validation
3. **Prediction**: Provide future exogenous values via `futr_df` parameter
4. **Scaling**: All features (target + exogenous) scaled independently using robust scaler
5. **Output**: Predictions inverse-transformed to original scale

### Our Implementation Workflow
```python
# Step 1: Prepare training data with exogenous features
df_train = df[['unique_id', 'ds', 'y', 'GDP', 'CPI', 'Interest_Rate']]

# Step 2: Create static dataframe with one-hot encoded IDs
static_df = create_one_hot_static_df(df_train['unique_id'].unique())

# Step 3: Train model
nf = NeuralForecast(models=[nbeatsx_model], freq='ME')
nf.fit(df=df_train, static_df=static_df, val_size=horizon)

# Step 4: Prepare future exogenous features
futr_df = df_test[['unique_id', 'ds', 'GDP', 'CPI', 'Interest_Rate']]

# Step 5: Generate forecasts
forecasts = nf.predict(futr_df=futr_df)
```

## Comparison: NBEATS vs NBEATSx

| Feature | NBEATS | NBEATSx |
|---------|--------|---------|
| Exogenous Variables | ❌ No | ✅ Yes |
| Static Features | ❌ No | ✅ Yes (one-hot series IDs) |
| Future Covariates | ❌ No | ✅ Yes (GDP, CPI, Interest_Rate) |
| Model Complexity | Lower | Higher |
| Training Time | Faster | Slower |
| Interpretability | Trend + Seasonality | Trend + Seasonality + Exog Impact |
| Best Use Case | Pure time series patterns | Time series + external drivers |

## Key Insights

**When NBEATSx Outperforms NBEATS:**
- Strong correlation between exogenous variables and target
- Macroeconomic factors drive purchasing decisions (e.g., vehicle registrations)
- Series share common external influences but have unique characteristics

**When NBEATS May Be Sufficient:**
- Exogenous features unavailable or unreliable
- Primarily seasonality and trend-driven patterns
- Simpler model preferred for faster training/deployment
