import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import warnings
import pickle
from sklearn.preprocessing import StandardScaler
from notebooks.john.LSTM import LSTMForecaster
import os
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')


# Parameters 
SEQ_LENGTH = 8
EMBARGO = 1
device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load data
cwd = os.getcwd()
full_path = os.path.join(cwd, "data", "processed", "monthly_registration_volume_gold.parquet")
df = pd.read_parquet(full_path, engine='fastparquet')
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
# Ensure Date is datetime
df['Date'] = pd.to_datetime(df['Date'])

# Get feature columns (same as training)
date_col = 'Date'
ts_key_col = 'ts_key'
value_col = 'Value'
feature_cols = [col for col in df.columns if col not in [date_col, ts_key_col, value_col]]

print(f"Loaded data shape: {df.shape}")
print(f"Feature columns ({len(feature_cols)}): {feature_cols[:5]}...")  # Show first 5

n_ts_keys = df['ts_key'].nunique()

# Load model checkpoint
model_path = os.path.join(cwd, "models", "lstm", "best_lstm_model_complete.pth")
checkpoint = torch.load(model_path, weights_only=False)

# Extract metadata from checkpoint
INPUT_SIZE = checkpoint['input_size']
scaler_X = checkpoint['scaler_X']
scaler_y = checkpoint['scaler_y']
ts_key_to_idx = checkpoint['ts_key_to_idx']
n_ts_keys = checkpoint['n_ts_keys']

print(f"\nModel configuration:")
print(f"  Input size: {INPUT_SIZE}")
print(f"  Sequence length: {SEQ_LENGTH}")
print(f"  Embargo: {EMBARGO}")
print(f"  Number of time series: {n_ts_keys}")

# Initialize and load model
model = LSTMForecaster(
    input_size=INPUT_SIZE,
    hidden_size=64,
    num_layers=2,
    dropout=0.2
).to(device)

model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
print("✓ Model loaded successfully")

# Filter time series for prediction
ts_keys_samples = df[df['ts_key'].str.contains('A-KLASSE', case=False, na=False)]['ts_key'].unique()[:10]
print(f"\nFound {len(ts_keys_samples)} time series to predict")
print(f"Target months: Aug 2025, Sep 2025, Oct 2025")

# Prepare to store predictions
predictions_data = []

for ts_key in ts_keys_samples:
    print(f"\nProcessing: {ts_key}")
    
    # Get historical data for this time series
    ts_data = df[df['ts_key'] == ts_key].sort_values('Date').copy()

    # Only use data before Aug 2025
    ts_data = ts_data[ts_data['Date'] < '2025-08-01'].copy()
    
    if len(ts_data) < SEQ_LENGTH:
        print(f"Skipping - insufficient data ({len(ts_data)} < {SEQ_LENGTH})")
        continue
    
    # Get the last SEQ_LENGTH observations as the initial window
    window_data = ts_data.tail(SEQ_LENGTH).copy()
    
    # Get ts_key encoding
    if ts_key not in ts_key_to_idx:
        print(f"Skipping - ts_key not in training vocabulary")
        continue
    
    ts_key_idx = ts_key_to_idx[ts_key]
    ts_key_onehot = np.zeros(n_ts_keys, dtype=np.float32)
    ts_key_onehot[ts_key_idx] = 1.0
    
    # Extract initial window data
    initial_values = window_data['Value'].values
    initial_dates = window_data['Date'].values
    initial_features = window_data[feature_cols].values  # Additional features
    
    # Autoregressive prediction for 3 months
    current_window = {
        'values': list(initial_values),
        'dates': list(initial_dates),
        'features': [list(row) for row in initial_features]  # Store as list of lists
    }
    
    predictions_for_series = []
    
    for month_ahead in range(1, 4):  # Aug, Sep, Oct 2025
        # Prepare input features for the sequence
        window_features = []
        
        for i in range(SEQ_LENGTH):
            date = pd.Timestamp(current_window['dates'][i])
            value = current_window['values'][i]
            additional_feats = current_window['features'][i]
            
            # Concatenate features in the same order as training:
            # [Value, Additional_Features..., Year, Month, ts_key_onehot...]
            features = np.concatenate([
                [value],                    # Value (1)
                additional_feats,           # Additional features (34)
                [date.year, date.month],    # Year, Month (2)
                ts_key_onehot              # One-hot encoding (3745)
            ])
            
            window_features.append(features)
        
        X_input = np.array([window_features], dtype=np.float32)  # (1, SEQ_LENGTH, n_features)
        
        # Scale continuous features (Value + additional + year + month)
        n_additional = len(feature_cols)
        n_continuous = 1 + n_additional + 2  # Value + additional + year + month
        
        # Extract continuous and one-hot parts
        X_continuous = X_input[:, :, :n_continuous].reshape(-1, n_continuous)
        X_onehot = X_input[:, :, n_continuous:]
        
        # Scale continuous features
        X_continuous_scaled = scaler_X.transform(X_continuous)
        X_continuous_scaled = X_continuous_scaled.reshape(1, SEQ_LENGTH, n_continuous)
        
        # Concatenate scaled continuous + unscaled one-hot
        X_scaled = np.concatenate([X_continuous_scaled, X_onehot], axis=2)
        X_tensor = torch.FloatTensor(X_scaled).to(device)
        
        # Make prediction
        with torch.no_grad():
            pred_scaled = model(X_tensor).cpu().numpy()
        
        # Inverse transform to original scale
        pred_original = scaler_y.inverse_transform(pred_scaled)[0][0]
        
        # Generate next date
        last_date = pd.Timestamp(current_window['dates'][-1])
        next_date = last_date + pd.DateOffset(months=1)
        
        print(f"  {next_date.strftime('%b %Y')}: {pred_original:.0f}")
        
        predictions_for_series.append({
            'ts_key': ts_key,
            'Date': next_date,
            'Predicted_Value': pred_original,
            'Month': next_date.strftime('%b %Y')
        })
        
        # Update window for next prediction (autoregressive)
        # For the predicted value, we need to estimate the additional features
        # Option 1: Use last known features (simple approach)
        # Option 2: Forecast features separately (more complex)
        
        # Here we use Option 1: carry forward the last known feature values
        last_features = current_window['features'][-1]
        
        # Shift the window: remove first timestep, add prediction
        current_window['values'] = current_window['values'][1:] + [pred_original]
        current_window['dates'] = current_window['dates'][1:] + [next_date]
        current_window['features'] = current_window['features'][1:] + [last_features]
    
    predictions_data.extend(predictions_for_series)

# Save predictions
output_path = os.path.join(cwd, "models", "lstm", "predictions")
os.makedirs(output_path, exist_ok=True)

predictions_file = os.path.join(output_path, 'predictions_2025.csv')
predictions_df = pd.DataFrame(predictions_data)
predictions_df.to_csv(predictions_file, index=False)

print("\n" + "="*60)
print("PREDICTION SUMMARY")
print("="*60)
print(f"✓ Predictions saved to: {predictions_file}")
print(f"  Total predictions: {len(predictions_df)}")
print(f"  Time series with predictions: {predictions_df['ts_key'].nunique()}")

# VISUALIZATION
print("\n" + "="*60)
print("Creating visualization")
print("="*60)

sample_series = list(ts_keys_samples[:6]) 
n_series = len(sample_series)
fig, axes = plt.subplots(n_series, 1, figsize=(16, 5*n_series))

plt.style.use('seaborn-v0_8-darkgrid')

if n_series == 1:
    axes = [axes]

for idx, ts_key in enumerate(sample_series):
    # Get historical data (last 18 months)
    historical = df[df['ts_key'] == ts_key].sort_values('Date').tail(18).copy()
    historical['Date'] = pd.to_datetime(historical['Date'])
    
    # Get predictions
    preds = predictions_df[predictions_df['ts_key'] == ts_key].copy()
    preds['Date'] = pd.to_datetime(preds['Date'])
    
    ax = axes[idx]
    
    # Historical data
    ax.plot(historical['Date'], historical['Value'], 
            marker='o', linewidth=2.5, markersize=6, 
            label='Historical Data', color='#1f77b4', alpha=0.8)
    
    if len(preds) > 0:
        # Connect last historical to first prediction
        last_historical_date = historical['Date'].iloc[-1]
        last_historical_value = historical['Value'].iloc[-1]
        
        ax.plot([last_historical_date, preds['Date'].iloc[0]], 
                [last_historical_value, preds['Predicted_Value'].iloc[0]], 
                linestyle=':', color='gray', alpha=0.6, linewidth=1.5)
        
        # Predictions
        ax.plot(preds['Date'], preds['Predicted_Value'], 
                marker='D', linewidth=2.5, markersize=8,
                label='LSTM Predictions (Aug-Oct 2025)', color='#d62728', 
                linestyle='--', alpha=0.9)
        
        # Add value labels
        for _, row in preds.iterrows():
            ax.annotate(f'{row["Predicted_Value"]:.0f}', 
                       xy=(row['Date'], row['Predicted_Value']),
                       xytext=(0, 10), textcoords='offset points',
                       ha='center', fontsize=9, color='#d62728',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                edgecolor='#d62728', alpha=0.7))
        
        # Shade prediction region
        ax.axvspan(preds['Date'].min(), preds['Date'].max(), 
                  alpha=0.1, color='red', label='_nolegend_')
    
    # Formatting
    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_ylabel('Registration Count', fontsize=12, fontweight='bold')
    
    title = ts_key.replace('_', ' ')
    ax.set_title(f'{title}', fontsize=14, fontweight='bold', pad=15)
    
    ax.legend(fontsize=11, loc='upper left', framealpha=0.9, shadow=True)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.7)
    
    # Format dates
    import matplotlib.dates as mdates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.tick_params(axis='x', rotation=45, labelsize=10)
    ax.tick_params(axis='y', labelsize=10)

plt.tight_layout(pad=2.0)
viz_file = os.path.join(output_path, 'predictions_visualization.png')
plt.savefig(viz_file, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {viz_file}")
plt.close()

print("\n" + "="*60)
print("DONE!")
print("="*60)