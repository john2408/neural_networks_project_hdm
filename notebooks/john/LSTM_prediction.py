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
SEQ_LENGTH = 6
INPUT_SIZE = 2307
EPOCHS = 50
BATCH_SIZE = 256
LEARNING_RATE = 0.001
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Load data

cwd = os.getcwd()
full_path = os.path.join(cwd, "data", "processed", "historical_kba_data.parquet")
df = pd.read_parquet(full_path, engine='fastparquet')
df["ts_key_size"] = df.groupby('ts_key')['ts_key'].transform('size')

# Filter ts_keys with at least 12 entries
df = df[df['ts_key_size'] >= 12].copy()

# Do not include timeseries which have the last 12 months as zero values
recent_12_months = df['Date'].max() - pd.DateOffset(months=12)
recent_data = df[df['Date'] > recent_12_months]
zero_value_ts_keys = recent_data.groupby('ts_key')['Value'].sum()
zero_value_ts_keys = zero_value_ts_keys[zero_value_ts_keys == 0].index
df = df[~df['ts_key'].isin(zero_value_ts_keys)].copy()


columns = ['Date','ts_key', 'Value']
df = df[columns].copy()

n_ts_keys = df['ts_key'].nunique()

# load ts_key_to_idx mapping

with open('models/lstm/ts_key_to_idx.pkl', 'rb') as f:
    ts_key_to_idx = pickle.load(f)


# Filter time series
ts_keys_samples = df[df['ts_key'].str.contains('A-KLASSE', case=False, na=False)]['ts_key'].unique()[:10]
print(f"\nFound {len(ts_keys_samples)}  time series")
print(f"Target months: Aug 2025, Sep 2025, Oct 2025")

model = LSTMForecaster(
    input_size=INPUT_SIZE,
    hidden_size=64,
    num_layers=2,
    dropout=0.2
).to(device)


cwd = os.getcwd()
model_path = os.path.join(cwd, "models", "lstm", "best_lstm_model_complete.pth")

# Load the best model (weights_only=False because checkpoint contains scalers and other objects)
checkpoint = torch.load(model_path, weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Extract scalers from checkpoint
scaler_X = checkpoint['scaler_X']
scaler_y = checkpoint['scaler_y']

# Prepare to store predictions
predictions_data = []
predictions_data = []

for ts_key in ts_keys_samples:
    # Get historical data for this time series
    ts_data = df[df['ts_key'] == ts_key].sort_values('Date').copy()

    # Only historical data before Aug 2025
    ts_data = ts_data[ts_data['Date'] < '2025-08-01'].copy()
    
    if len(ts_data) < SEQ_LENGTH:
        continue
    
    # Get the last 6 months as the initial window (May-Jul 2025 or latest available)
    ts_data['Date'] = pd.to_datetime(ts_data['Date'])
    latest_date = ts_data['Date'].max()
    
    # Use last 6 observations
    window_data = ts_data.tail(SEQ_LENGTH).copy()
    values = window_data['Value'].values
    dates = window_data['Date'].values
    
    # Get ts_key encoding
    if ts_key not in ts_key_to_idx:
        continue
    
    ts_key_idx = ts_key_to_idx[ts_key]
    ts_key_onehot = np.zeros(n_ts_keys, dtype=np.float32)
    ts_key_onehot[ts_key_idx] = 1.0
    
    # Autoregressive prediction for 3 months
    current_window = values.copy()
    current_dates = list(dates)
    predictions_for_series = []
    
    for month_ahead in range(1, 4):  # Aug, Sep, Oct 2025
        # Prepare input features
        window_features = []
        
        for i in range(SEQ_LENGTH):
            date = pd.Timestamp(current_dates[i])
            features = np.concatenate([
                [current_window[i]],
                [date.year],
                [date.month],
                ts_key_onehot
            ])
            window_features.append(features)
        
        X_input = np.array([window_features], dtype=np.float32)  # (1, 6, n_features)
        
        # Scale continuous features
        n_continuous = 3
        X_continuous = X_input[:, :, :n_continuous].reshape(-1, n_continuous)
        X_onehot = X_input[:, :, n_continuous:]
        
        X_continuous_scaled = scaler_X.transform(X_continuous)
        X_continuous_scaled = X_continuous_scaled.reshape(1, SEQ_LENGTH, n_continuous)
        
        X_scaled = np.concatenate([X_continuous_scaled, X_onehot], axis=2)
        X_tensor = torch.FloatTensor(X_scaled).to(device)
        
        # Make prediction
        with torch.no_grad():
            pred_scaled = model(X_tensor).cpu().numpy()
        
        # Inverse transform
        pred_original = scaler_y.inverse_transform(pred_scaled)[0][0]
        
        # Generate next date
        last_date = pd.Timestamp(current_dates[-1])
        next_date = last_date + pd.DateOffset(months=1)
        
        predictions_for_series.append({
            'ts_key': ts_key,
            'Date': next_date,
            'Predicted_Value': pred_original,
            'Month': next_date.strftime('%b %Y')
        })
        
        # Update window for next prediction (autoregressive)
        current_window = np.append(current_window[1:], pred_original)
        current_dates = current_dates[1:] + [next_date]
    
    predictions_data.extend(predictions_for_series)

output_path = os.path.join(cwd, "models", "lstm", "predictions")

predictions_file = os.path.join(output_path, 'predictions_2025.csv')
# Create predictions DataFrame
predictions_df = pd.DataFrame(predictions_data)
predictions_df.to_csv(predictions_file, index=False)
print(f"\n✓ Predictions saved to: {predictions_file}")
print(f"  Total predictions: {len(predictions_df)}")
print(f"  series with predictions: {predictions_df['ts_key'].nunique()}")

# STEP 8: Visualization for sample series
print("\n" + "="*60)
print("STEP 8: Creating visualization")
print("="*60)


sample_series = list(ts_keys_samples[:6]) 

n_series = len(sample_series)
fig, axes = plt.subplots(n_series, 1, figsize=(16, 5*n_series))

# Set style
plt.style.use('seaborn-v0_8-darkgrid')

if n_series == 1:
    axes = [axes]

for idx, ts_key in enumerate(sample_series):
    # Get historical data (last 18 months to focus on recent trend)
    historical = df[df['ts_key'] == ts_key].sort_values('Date').tail(18).copy()
    historical['Date'] = pd.to_datetime(historical['Date'])
    
    # Get predictions
    preds = predictions_df[predictions_df['ts_key'] == ts_key].copy()
    preds['Date'] = pd.to_datetime(preds['Date'])
    
    # Plot
    ax = axes[idx]
    
    # Historical data with better styling
    ax.plot(historical['Date'], historical['Value'], 
            marker='o', linewidth=2.5, markersize=6, 
            label='Historical Data', color='#1f77b4', alpha=0.8)
    
    if len(preds) > 0:
        # Connect last historical point to first prediction with subtle line
        last_historical_date = historical['Date'].iloc[-1]
        last_historical_value = historical['Value'].iloc[-1]
        
        ax.plot([last_historical_date, preds['Date'].iloc[0]], 
                [last_historical_value, preds['Predicted_Value'].iloc[0]], 
                linestyle=':', color='gray', alpha=0.6, linewidth=1.5)
        
        # Predictions with distinct styling
        ax.plot(preds['Date'], preds['Predicted_Value'], 
                marker='D', linewidth=2.5, markersize=8,
                label='LSTM Predictions (Aug-Oct 2025)', color='#d62728', 
                linestyle='--', alpha=0.9)
        
        # Add value labels on prediction points
        for _, row in preds.iterrows():
            ax.annotate(f'{row["Predicted_Value"]:.0f}', 
                       xy=(row['Date'], row['Predicted_Value']),
                       xytext=(0, 10), textcoords='offset points',
                       ha='center', fontsize=9, color='#d62728',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                edgecolor='#d62728', alpha=0.7))
    
    # Formatting
    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_ylabel('Registration Count', fontsize=12, fontweight='bold')
    
    # Clean title
    title = ts_key.replace('_', ' ')
    ax.set_title(f'{title}', fontsize=14, fontweight='bold', pad=15)
    
    ax.legend(fontsize=11, loc='upper left', framealpha=0.9, shadow=True)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.7)
    
    # Format x-axis dates
    import matplotlib.dates as mdates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.tick_params(axis='x', rotation=45, labelsize=10)
    ax.tick_params(axis='y', labelsize=10)
    
    # Add background shading for prediction period
    if len(preds) > 0:
        ax.axvspan(preds['Date'].min(), preds['Date'].max(), 
                  alpha=0.1, color='red', label='_nolegend_')

plt.tight_layout(pad=2.0)
plt.savefig(os.path.join(output_path, 'predictions_visualization.png'), dpi=300, bbox_inches='tight')
print(f"✓ Saved: {os.path.join(output_path, 'predictions_visualization.png')}")
plt.close()
