import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import warnings
import pickle
import time
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
   
warnings.filterwarnings('ignore')


# Check MPS availability
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print(f"✓ MPS device is available")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"✓ CUDA device is available")
else:
    device = torch.device("cpu")
    print(f"Using CPU")
    
print(f"Device: {device}")



class TimeSeriesDataset(Dataset):
    """
    Dataset for MULTIVARIATE time series with sliding window approach and embargo period.
    Each sample: [features(t-seq_length-embargo), ..., features(t-1-embargo)] -> target(t)
    Features include: Value + additional features + year + month + one-hot encoded ts_key
    Memory efficient: doesn't pivot the entire dataframe
    
    Embargo period creates a gap between last observation and prediction to avoid overfitting
    due to autocorrelation. E.g., with seq_length=3 and embargo=1:
        X = [t-4, t-3, t-2] -> Y = t (skipping t-1)
    """
    def __init__(self, df, feature_cols=None, seq_length=6, embargo=1, train=True, train_ratio=0.8, scaler_X=None, scaler_y=None):
        """
        Args:
            df: DataFrame with columns [Date, ts_key, Value, ...additional features]
            feature_cols: List of additional feature column names (e.g., economic indicators)
                         If None, will auto-detect (all columns except Date, ts_key, Value)
            seq_length: Lookback window (number of timesteps)
            embargo: Number of months to skip between last observation and prediction target
            train: If True, create training set, else test set
            train_ratio: Train/test split ratio
            scaler_X: Pre-fitted StandardScaler for features (used for test set)
            scaler_y: Pre-fitted StandardScaler for target (used for test set)
        """
        self.seq_length = seq_length
        self.embargo = embargo
        self.X = []
        self.y = []
        self.ts_keys_list = []
        
        # Auto-detect feature columns if not provided
        if feature_cols is None:
            self.feature_cols = [col for col in df.columns 
                                if col not in ['Date', 'ts_key', 'Value']]
        else:
            self.feature_cols = feature_cols
        
        self.n_features_additional = len(self.feature_cols)
        
        # Create one-hot encoding mapping for ts_keys
        unique_ts_keys = sorted(df['ts_key'].unique())
        self.ts_key_to_idx = {key: idx for idx, key in enumerate(unique_ts_keys)}
        self.n_ts_keys = len(unique_ts_keys)
        
        print(f"Creating dataset with {self.n_ts_keys} time series...")
        print(f"Additional features: {self.n_features_additional}")
        print(f"Sequence length: {seq_length}, Embargo: {embargo}")
        
        # Group by time series key
        grouped = df.sort_values('Date').groupby('ts_key')
        
        for ts_key, group in grouped:
            values = group['Value'].values
            dates = pd.to_datetime(group['Date'].values)
            
            # Extract additional features
            if self.n_features_additional > 0:
                additional_features = group[self.feature_cols].values
            
            # Skip if not enough data (need seq_length + embargo)
            min_length = seq_length + embargo
            if len(values) < min_length + 1:
                continue
            
            # Get one-hot encoding for this ts_key
            ts_key_idx = self.ts_key_to_idx[ts_key]
            ts_key_onehot = np.zeros(self.n_ts_keys, dtype=np.float32)
            ts_key_onehot[ts_key_idx] = 1.0
            
            # Create sliding windows with embargo
            # If embargo=1: X ends at t-2, Y is at t (skipping t-1)
            for i in range(len(values) - seq_length - embargo):
                # Collect features for each timestep in the window
                window_features = []
                
                for j in range(seq_length):
                    # Features: [value, additional_features..., year, month, ts_key_onehot...]
                    date = dates[i + j]
                    
                    features_list = [values[i + j]]  # Value
                    
                    # Add additional features if present
                    if self.n_features_additional > 0:
                        features_list.append(additional_features[i + j])
                    
                    # Add temporal features
                    features_list.extend([date.year, date.month])
                    
                    # Add one-hot encoding
                    features_list.append(ts_key_onehot)
                    
                    # Concatenate all features
                    features = np.concatenate([
                        np.array(f).flatten() if not isinstance(f, (int, float)) else [f]
                        for f in features_list
                    ])
                    
                    window_features.append(features)
                
                self.X.append(np.array(window_features))  # Shape: (seq_length, n_features)
                # Target is embargo periods ahead from last observation
                self.y.append(values[i + seq_length + embargo - 1])
                self.ts_keys_list.append(ts_key)
        
        self.X = np.array(self.X, dtype=np.float32)  # Shape: (n_samples, seq_length, n_features)
        self.y = np.array(self.y, dtype=np.float32)  # Shape: (n_samples,)
        
        print(f"Created {len(self.X)} samples with feature dimension: {self.X.shape[2]}")
        print(f"  - Value: 1")
        print(f"  - Additional features: {self.n_features_additional}")
        print(f"  - Temporal (year, month): 2")
        print(f"  - One-hot ts_key: {self.n_ts_keys}")
        
        # Train-test split (chronological)
        n_samples = len(self.X)
        train_size = int(n_samples * train_ratio)
        
        if train:
            self.X = self.X[:train_size]
            self.y = self.y[:train_size]
            self.ts_keys_list = self.ts_keys_list[:train_size]
        else:
            self.X = self.X[train_size:]
            self.y = self.y[train_size:]
            self.ts_keys_list = self.ts_keys_list[train_size:]
        
        # Standardize features (Value + additional features + year + month - NOT one-hot)
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        
        # Extract continuous features for scaling
        n_samples, seq_len, n_features = self.X.shape
        n_continuous = 1 + self.n_features_additional + 2  # Value + additional + year + month
        
        X_continuous = self.X[:, :, :n_continuous].reshape(-1, n_continuous)
        X_onehot = self.X[:, :, n_continuous:].reshape(-1, self.n_ts_keys)
        
        if train:
            X_continuous_scaled = self.scaler_X.fit_transform(X_continuous)
            X_continuous_scaled = X_continuous_scaled.reshape(n_samples, seq_len, n_continuous)
            X_onehot_reshaped = X_onehot.reshape(n_samples, seq_len, self.n_ts_keys)
            
            # Concatenate scaled continuous + unscaled one-hot
            self.X = np.concatenate([X_continuous_scaled, X_onehot_reshaped], axis=2)
            self.y = self.scaler_y.fit_transform(self.y.reshape(-1, 1)).flatten()
        else:
            # For test set, apply provided scalers or leave unscaled
            if scaler_X is not None and scaler_y is not None:
                self.scaler_X = scaler_X
                self.scaler_y = scaler_y
                
                # Apply the training scalers to test set
                X_continuous_scaled = self.scaler_X.transform(X_continuous)
                X_continuous_scaled = X_continuous_scaled.reshape(n_samples, seq_len, n_continuous)
                X_onehot_reshaped = X_onehot.reshape(n_samples, seq_len, self.n_ts_keys)
                
                # Concatenate scaled continuous + unscaled one-hot
                self.X = np.concatenate([X_continuous_scaled, X_onehot_reshaped], axis=2)
                self.y = self.scaler_y.transform(self.y.reshape(-1, 1)).flatten()
            else:
                # Scalers not provided - data remains unscaled (will need to be scaled externally)
                pass
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.X[idx]), torch.FloatTensor([self.y[idx]])


class LSTMForecaster(nn.Module):
    """
    LSTM model for MULTIVARIATE time series forecasting.
    Architecture: LSTM -> Dropout -> LSTM -> Dropout -> Fully Connected
    Takes multiple input features at each timestep.
    """
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        """
        Args:
            input_size: Number of input features (Value + year + month + one-hot)
            hidden_size: LSTM hidden dimension
            num_layers: Number of LSTM layers
            dropout: Dropout rate
        """
        super(LSTMForecaster, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_size = input_size
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Fully connected output layer
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        # x shape: (batch_size, seq_length, input_size)
        
        # LSTM forward pass
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Take the output from the last time step
        last_output = lstm_out[:, -1, :]  # Shape: (batch_size, hidden_size)
        
        # Apply dropout
        out = self.dropout(last_output)
        
        # Fully connected layer
        out = self.fc(out)  # Shape: (batch_size, 1)
        
        return out

def generate_out_of_sample_predictions_old(
    model, 
    df_test_period,
    df_full, 
    fold_config, 
    features, 
    scaler_X, 
    scaler_y, 
    ts_key_to_idx, 
    n_ts_keys, 
    seq_length, 
    embargo, 
    device
):
    """
    Generate out-of-sample predictions for a test period using autoregressive forecasting.
    
    Args:
        model: Trained PyTorch model
        df_full: Complete dataframe with all data
        fold_config: Dict with 'test_start' and 'test_end' dates
        features: List of feature column names
        scaler_X: Fitted StandardScaler for features
        scaler_y: Fitted StandardScaler for target
        ts_key_to_idx: Dict mapping ts_key to one-hot index
        n_ts_keys: Total number of time series
        seq_length: Lookback window size
        embargo: Gap between last observation and prediction
        device: torch.device (cpu/cuda/mps)
    
    Returns:
        predictions_dict: Dict mapping ts_key -> list of predictions
        actuals_dict: Dict mapping ts_key -> list of actual values
        all_preds: Flattened array of all predictions
        all_acts: Flattened array of all actuals
    """
    print("\n" + "-"*60)
    print("STEP 3: Out-of-sample predictions")
    print("-"*60)

    
    print(f"Test period: {fold_config['test_start']} to {fold_config['test_end']}")
    print(f"Test observations: {len(df_test_period):,}")
    
    # For prediction, we need the last SEQ_LENGTH + EMBARGO observations before test period
    # to build the initial input sequence
    test_start_date = pd.to_datetime(fold_config['test_start'])
    lookback_start = test_start_date - pd.DateOffset(months=seq_length + embargo)
    
    df_for_prediction = df_full[
        (df_full['Date'] >= lookback_start) & 
        (df_full['Date'] <= fold_config['test_end'])
    ].copy()
    
    # Generate predictions using autoregressive approach
    model.eval()
    predictions_dict = {}
    actuals_dict = {}
    
    # Group by time series
    for ts_key, group in df_for_prediction.groupby('ts_key'):
        group = group.sort_values('Date')
        
        # Get ts_key one-hot encoding
        if ts_key not in ts_key_to_idx:
            continue  # Skip if time series not seen during training
        
        ts_key_idx = ts_key_to_idx[ts_key]
        ts_key_onehot = np.zeros(n_ts_keys, dtype=np.float32)
        ts_key_onehot[ts_key_idx] = 1.0
        
        # Get historical data (before test period)
        hist_data = group[group['Date'] < test_start_date]
        
        if len(hist_data) < seq_length + embargo:
            continue  # Not enough history
        
        # Initialize with last SEQ_LENGTH + EMBARGO observations
        recent_values = hist_data['Value'].values[-(seq_length + embargo):]
        recent_features = hist_data[features].values[-(seq_length + embargo):]
        recent_dates = pd.to_datetime(hist_data['Date'].values[-(seq_length + embargo):])
        
        # Predict for each month in test period
        test_dates = group[group['Date'] >= test_start_date]['Date'].values
        test_actuals = group[group['Date'] >= test_start_date]['Value'].values
        
        predictions_list = []
        
        for pred_idx, pred_date in enumerate(test_dates):
            pred_date = pd.to_datetime(pred_date)
            
            # Build input sequence (last SEQ_LENGTH observations, with EMBARGO gap)
            # If EMBARGO=1: use [t-SEQ_LENGTH-1, ..., t-2] to predict t
            sequence = []
            
            for i in range(seq_length):
                idx = len(recent_values) - seq_length - embargo + i
                
                if idx < 0 or idx >= len(recent_values):
                    break
                
                date = pd.to_datetime(recent_dates[idx])
                value = recent_values[idx]
                feat = recent_features[idx]
                
                # Build feature vector
                features_list = [value]
                features_list.append(feat)
                features_list.extend([date.year, date.month])
                features_list.append(ts_key_onehot)
                
                feature_vector = np.concatenate([
                    np.array(f).flatten() if not isinstance(f, (int, float)) else [f]
                    for f in features_list
                ])
                
                sequence.append(feature_vector)
            
            if len(sequence) != seq_length:
                break
            
            # Scale input
            sequence = np.array(sequence, dtype=np.float32)
            n_continuous = 1 + len(features) + 2
            
            seq_continuous = sequence[:, :n_continuous]
            seq_onehot = sequence[:, n_continuous:]
            
            seq_continuous_scaled = scaler_X.transform(seq_continuous)
            sequence_scaled = np.concatenate([seq_continuous_scaled, seq_onehot], axis=1)
            
            # Predict
            with torch.no_grad():
                X_input = torch.FloatTensor(sequence_scaled).unsqueeze(0).to(device)
                pred_scaled = model(X_input).cpu().numpy()[0, 0]
                pred_value = scaler_y.inverse_transform([[pred_scaled]])[0, 0]
            
            predictions_list.append(pred_value)
            
            # Update history for next prediction
            # Get actual features for this date (or carry forward last known)
            actual_row = group[group['Date'] == pred_date]
            if len(actual_row) > 0:
                actual_value = actual_row['Value'].values[0]
                actual_features = actual_row[features].values[0]
            else:
                actual_value = pred_value  # Fallback
                actual_features = recent_features[-1]
            
            recent_values = np.append(recent_values, actual_value)
            recent_features = np.vstack([recent_features, actual_features])
            recent_dates = np.append(recent_dates, pred_date)
        
        if len(predictions_list) > 0:
            predictions_dict[ts_key] = predictions_list
            actuals_dict[ts_key] = test_actuals[:len(predictions_list)]
    
    # Flatten predictions and actuals
    all_preds = np.concatenate([np.array(v) for v in predictions_dict.values()])
    all_acts = np.concatenate([np.array(v) for v in actuals_dict.values()])
    
    print(f"Generated predictions for {len(predictions_dict)} time series")
    print(f"Total predictions: {len(all_preds):,}")
    
    return predictions_dict, actuals_dict, all_preds, all_acts


def generate_out_of_sample_predictions(
    model, 
    df_test_period,
    df_full, 
    fold_config, 
    features, 
    scaler_X, 
    scaler_y, 
    ts_key_to_idx, 
    n_ts_keys, 
    seq_length, 
    embargo, 
    device
):
    """
    Generate out-of-sample predictions using BATCHED autoregressive forecasting.
    
    This function optimizes the prediction process by:
    1. Batches all time series together for each prediction step
    2. Runs ONE forward pass for all time series simultaneously
    3. Updates all histories together
    
    
    Args:
        model: Trained PyTorch model
        df_test_period: Test period dataframe
        df_full: Complete dataframe with all data
        fold_config: Dict with 'test_start' and 'test_end' dates
        features: List of feature column names
        scaler_X: Fitted StandardScaler for features
        scaler_y: Fitted StandardScaler for target
        ts_key_to_idx: Dict mapping ts_key to one-hot index
        n_ts_keys: Total number of time series
        seq_length: Lookback window size
        embargo: Gap between last observation and prediction
        device: torch.device (cpu/cuda/mps)
    
    Returns:
        predictions_dict: Dict mapping ts_key -> list of predictions
        actuals_dict: Dict mapping ts_key -> list of actual values
        all_preds: Flattened array of all predictions
        all_acts: Flattened array of all actuals
    """
    print("\n" + "-"*60)
    print("STEP 3: Out-of-sample predictions (OPTIMIZED)")
    print("-"*60)
    
    print(f"Test period: {fold_config['test_start']} to {fold_config['test_end']}")
    print(f"Test observations: {len(df_test_period):,}")
    
    # Setup
    test_start_date = pd.to_datetime(fold_config['test_start'])
    lookback_start = test_start_date - pd.DateOffset(months=seq_length + embargo)
    
    df_for_prediction = df_full[
        (df_full['Date'] >= lookback_start) & 
        (df_full['Date'] <= fold_config['test_end'])
    ].copy()
    
    model.eval()
    
    # -------------------------------------------------------------------------
    # STEP 1: Initialize data structures for ALL time series at once
    # -------------------------------------------------------------------------
    
    ts_data = {}  # Master dictionary storing everything about each time series
    valid_ts_keys = []  # Time series with enough history
    
    for ts_key, group in df_for_prediction.groupby('ts_key'):
        group = group.sort_values('Date')
        
        # Skip if not in training set
        if ts_key not in ts_key_to_idx:
            continue
        
        # Get historical data
        hist_data = group[group['Date'] < test_start_date]
        
        if len(hist_data) < seq_length + embargo:
            continue
        
        # Get test data
        test_data = group[group['Date'] >= test_start_date]
        
        if len(test_data) == 0:
            continue
        
        # Initialize time series data
        ts_data[ts_key] = {
            'ts_key_idx': ts_key_to_idx[ts_key],
            'recent_values': hist_data['Value'].values[-(seq_length + embargo):].copy(),
            'recent_features': hist_data[features].values[-(seq_length + embargo):].copy(),
            'recent_dates': pd.to_datetime(hist_data['Date'].values[-(seq_length + embargo):]),
            'test_dates': test_data['Date'].values,
            'test_actuals': test_data['Value'].values,
            'predictions': [],
            'n_predictions': len(test_data)
        }
        valid_ts_keys.append(ts_key)
    
    if len(valid_ts_keys) == 0:
        print("No valid time series found!")
        return {}, {}, np.array([]), np.array([])
    
    print(f"Processing {len(valid_ts_keys)} time series in batched mode...")
    
    # -------------------------------------------------------------------------
    # STEP 2: Determine maximum prediction horizon across all time series
    # -------------------------------------------------------------------------
    
    max_horizon = max(ts_data[ts_key]['n_predictions'] for ts_key in valid_ts_keys)
    n_continuous = 1 + len(features) + 2
    
    # -------------------------------------------------------------------------
    # STEP 3: Autoregressive prediction - ONE BATCH PER TIME STEP
    # -------------------------------------------------------------------------
    
    for step in range(max_horizon):
        # Collect all time series that need prediction at this step
        batch_ts_keys = []
        batch_sequences = []
        
        for ts_key in valid_ts_keys:
            data = ts_data[ts_key]
            
            # Skip if this time series has already made all its predictions
            if step >= data['n_predictions']:
                continue
            
            # Build sequence for this time series
            sequence = []
            ts_key_onehot = np.zeros(n_ts_keys, dtype=np.float32)
            ts_key_onehot[data['ts_key_idx']] = 1.0
            
            for i in range(seq_length):
                idx = len(data['recent_values']) - seq_length - embargo + i
                
                if idx < 0 or idx >= len(data['recent_values']):
                    break
                
                date = pd.to_datetime(data['recent_dates'][idx])
                value = data['recent_values'][idx]
                feat = data['recent_features'][idx]
                
                # Build feature vector
                features_list = [value]
                features_list.append(feat)
                features_list.extend([date.year, date.month])
                features_list.append(ts_key_onehot)
                
                feature_vector = np.concatenate([
                    np.array(f).flatten() if not isinstance(f, (int, float)) else [f]
                    for f in features_list
                ])
                
                sequence.append(feature_vector)
            
            # Only add if we have a complete sequence
            if len(sequence) == seq_length:
                batch_ts_keys.append(ts_key)
                batch_sequences.append(sequence)
        
        # If no time series to predict at this step, continue
        if len(batch_sequences) == 0:
            continue
        
        # -------------------------------------------------------------------------
        # STEP 4: Batch prediction for all time series at this step
        # -------------------------------------------------------------------------
        
        # Stack all sequences into a batch
        batch_array = np.array(batch_sequences, dtype=np.float32)  # (batch_size, seq_length, n_features)
        
        # Separate continuous and one-hot features
        batch_continuous = batch_array[:, :, :n_continuous]  # (batch_size, seq_length, n_continuous)
        batch_onehot = batch_array[:, :, n_continuous:]      # (batch_size, seq_length, n_onehot)
        
        # Scale continuous features
        batch_size, seq_len, n_cont = batch_continuous.shape
        batch_continuous_flat = batch_continuous.reshape(-1, n_cont)
        batch_continuous_scaled = scaler_X.transform(batch_continuous_flat)
        batch_continuous_scaled = batch_continuous_scaled.reshape(batch_size, seq_len, n_cont)
        
        # Recombine
        batch_scaled = np.concatenate([batch_continuous_scaled, batch_onehot], axis=2)
        
        # Single forward pass for entire batch
        with torch.no_grad():
            X_batch = torch.FloatTensor(batch_scaled).to(device)
            pred_scaled_batch = model(X_batch).cpu().numpy()  # (batch_size, 1)
            pred_values_batch = scaler_y.inverse_transform(pred_scaled_batch).flatten()
        
        # All negative predictions are set to zero
        pred_values_batch = np.maximum(pred_values_batch, 0.0)

        # -------------------------------------------------------------------------
        # STEP 5: Update histories for all time series in batch
        # -------------------------------------------------------------------------
        
        for i, ts_key in enumerate(batch_ts_keys):
            data = ts_data[ts_key]
            pred_value = pred_values_batch[i]
            
            # Store prediction
            data['predictions'].append(pred_value)
            
            # Get actual value and features for this prediction
            pred_date = pd.to_datetime(data['test_dates'][step])
            
            # Use actual value from test set
            actual_value = data['test_actuals'][step]
            
            # Get actual features or carry forward
            test_group = df_for_prediction[
                (df_for_prediction['ts_key'] == ts_key) & 
                (df_for_prediction['Date'] == pred_date)
            ]
            
            if len(test_group) > 0:
                actual_features = test_group[features].values[0]
            else:
                actual_features = data['recent_features'][-1]
            
            # Update history
            data['recent_values'] = np.append(data['recent_values'], actual_value)
            data['recent_features'] = np.vstack([data['recent_features'], actual_features])
            data['recent_dates'] = np.append(data['recent_dates'], pred_date)
    
    # -------------------------------------------------------------------------
    # STEP 6: Collect results
    # -------------------------------------------------------------------------
    
    predictions_dict = {}
    actuals_dict = {}
    
    for ts_key in valid_ts_keys:
        data = ts_data[ts_key]
        if len(data['predictions']) > 0:
            predictions_dict[ts_key] = data['predictions']
            actuals_dict[ts_key] = data['test_actuals'][:len(data['predictions'])]
    
    all_preds = np.concatenate([np.array(v) for v in predictions_dict.values()])
    all_acts = np.concatenate([np.array(v) for v in actuals_dict.values()])
    
    print(f"Generated predictions for {len(predictions_dict)} time series")
    print(f"Total predictions: {len(all_preds):,}")
    print(f"Optimization: {len(valid_ts_keys) * max_horizon} individual predictions → {max_horizon} batched forward passes")
    
    return predictions_dict, actuals_dict, all_preds, all_acts


def smape(y_true, y_pred):
    """Symmetric Mean Absolute Percentage Error"""
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    diff = np.abs(y_true - y_pred) / denominator
    diff[denominator == 0] = 0.0
    return 100 * np.mean(diff)

def calculate_smape_distribution(predictions_dict, actuals_dict):
    """
    Calculate SMAPE per time series and categorize into quality buckets.
    
    Args:
        predictions_dict: Dict mapping ts_key -> list of predictions
        actuals_dict: Dict mapping ts_key -> list of actual values
        
    Returns:
        DataFrame with ts_key, smape, and category columns
    """
    smape_results = []
    
    for ts_key in predictions_dict.keys():
        preds = np.array(predictions_dict[ts_key])
        acts = np.array(actuals_dict[ts_key])
        
        # Calculate SMAPE for this time series
        ts_smape = smape(acts, preds)
        
        # Categorize SMAPE
        if ts_smape < 10:
            category = '<10%'
        elif ts_smape <= 20:
            category = '10-20%'
        elif ts_smape <= 30:
            category = '20-30%'
        elif ts_smape <= 40:
            category = '30-40%'
        else:
            category = '>40%'
        
        smape_results.append({
            'ts_key': ts_key,
            'smape': ts_smape,
            'category': category
        })
    
    return pd.DataFrame(smape_results)

def train_epoch(model, loader, criterion, optimizer, device):

    # Enable training mode
    model.train()

    total_loss = 0

    for X_batch, y_batch in loader:
        # Move data to MPS/CUDA if available
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        # Clear all gradients from previous iteration
        optimizer.zero_grad()

        # Pass the batch through the model
        predictions = model(X_batch)

        loss = criterion(predictions, y_batch)

        # Backpropagation and optimization
        loss.backward()

        # Prevetns exploding gradients, scales gradients if norm exceeds max_norm
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Update weights
        optimizer.step()

        # Add batch loss to total loss
        total_loss += loss.item()

    # Divides by number of batches to get average loss per batch
    return total_loss / len(loader)

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            total_loss += loss.item()
    return total_loss / len(loader)

if __name__ == "__main__":

    # Load data
    import os
    cwd = os.getcwd()
    full_path = os.path.join(cwd, "data", "gold", "monthly_registration_volume_gold.parquet")
    output_path = os.path.join(cwd, "models", "lstm")
    os.makedirs(output_path, exist_ok=True)

    df_full = pd.read_parquet(full_path, engine='pyarrow')
    df_full['Year'] = df_full['Date'].dt.year
    df_full['Month'] = df_full['Date'].dt.month

    date_col = 'Date'
    ts_key_col = 'ts_key'
    value_col = 'Value'
    features = [col for col in df_full.columns if col not in [date_col, ts_key_col, value_col]]
    

    #Validate not NaN or infinite values in features
    assert not df_full[features].isna().any().any(), "NaN values found in features"
    assert not np.isinf(df_full[features].select_dtypes(include=[np.number])).any().any(), "Infinite values found in features"

    print("="*80)
    print("DATA OVERVIEW")
    print("="*80)
    print(f"Original data shape: {df_full.shape}")
    print(f"Unique time series: {df_full.ts_key.nunique()}")
    print(f"Date range: {df_full.Date.min()} to {df_full.Date.max()}")
    print(f"Total observations: {len(df_full):,}")
    

    # Check original data
    print(f"\nOriginal DataFrame:")
    print(f"  NaN values: {df_full.isna().sum().sum()}")
    print(f"  Inf values: {np.isinf(df_full.select_dtypes(include=[np.number])).sum().sum()}")
    print(f"  Value range: [{df_full['Value'].min():.2f}, {df_full['Value'].max():.2f}]")
    
    # Check for zero/very small values that could cause division issues
    print(f"\nZero values in 'Value' column: {(df_full['Value'] == 0).sum()}")
    print(f"Values < 0.01: {(df_full['Value'] < 0.01).sum()}")
    
    # ========================================================================
    # FOLD CONFIGURATION - As per Problem Description
    # ========================================================================
    
    folds = [
        {
            'name': 'Fold 1',
            'train_end': '2024-09-30',    # Train on data up to Sep 2024
            'test_start': '2024-10-01',   # Test on Oct-Dec 2024
            'test_end': '2024-12-31'
        },
        {
            'name': 'Fold 2',
            'train_end': '2024-12-31',    # Train on data up to Dec 2024
            'test_start': '2025-01-01',   # Test on Jan-Mar 2025
            'test_end': '2025-03-31'
        },
        {
            'name': 'Fold 3',
            'train_end': '2025-06-30',    # Train on data up to Jun 2025
            'test_start': '2025-07-01',   # Test on Jul-Sep 2025
            'test_end': '2025-09-30'
        }
    ]
    
    print("\n" + "="*80)
    print("FOLD CONFIGURATION")
    print("="*80)
    for fold in folds:
        print(f"\n{fold['name']}:")
        print(f"  Training data: up to {fold['train_end']}")
        print(f"  Test period: {fold['test_start']} to {fold['test_end']}")
    
    
    # ========================================================================
    # TRAINING PARAMETERS
    # ========================================================================
    
    SEQ_LENGTH = 8
    TRAIN_RATIO = 0.8
    EMBARGO = 1
    EPOCHS = 25
    BATCH_SIZE = 256
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-5
    
    # ========================================================================
    # FOLD-WISE TRAINING AND EVALUATION
    # ========================================================================
    
    fold_metrics = []
    fold_smape_distributions = []
    import matplotlib.pyplot as plt
    
    for fold_idx, fold_config in enumerate(folds):
        
        # Fold Variables
        fold_output_dir = os.path.join(output_path, f"fold_{fold_idx + 1}")
        os.makedirs(fold_output_dir, exist_ok=True)

        print("\n" + "="*80)
        print(f"{fold_config['name'].upper()}: {fold_config['test_start']} to {fold_config['test_end']}")
        print("="*80)
        
        # Filter training data up to train_end date
        df_train = df_full[df_full['Date'] <= fold_config['train_end']].copy()
        
        print(f"\nFiltered training data up to {fold_config['train_end']}")
        print(f"  Training observations: {len(df_train):,}")
        print(f"  Date range: {df_train['Date'].min()} to {df_train['Date'].max()}")
        
        # -----------------------------------------------------------------
        # STEP 1: Create datasets for model development (train/val split)
        # -----------------------------------------------------------------
        print("\n" + "-"*60)
        print("STEP 1: Creating datasets for model development")
        print("-"*60)
        
        train_dataset = TimeSeriesDataset(
            df_train,
            feature_cols=features, 
            seq_length=SEQ_LENGTH,
            embargo=EMBARGO,
            train=True,
            train_ratio=TRAIN_RATIO
        )
        
        print(f"Training samples: {len(train_dataset):,}")
        
        # Save scalers and metadata
        scaler_X = train_dataset.scaler_X
        scaler_y = train_dataset.scaler_y
        n_ts_keys = train_dataset.n_ts_keys
        ts_key_to_idx = train_dataset.ts_key_to_idx
        
        test_dataset = TimeSeriesDataset(
            df_train,
            feature_cols=features,
            seq_length=SEQ_LENGTH,
            train=False,
            embargo=EMBARGO,
            train_ratio=TRAIN_RATIO,
            scaler_X=scaler_X,
            scaler_y=scaler_y
        )
        
        print(f"Validation samples: {len(test_dataset):,}")
        
        # -----------------------------------------------------------------
        # STEP 2: Initialize and train model
        # -----------------------------------------------------------------
        print("\n" + "-"*60)
        print("STEP 2: Training LSTM model")
        print("-"*60)
        
        INPUT_SIZE = train_dataset.X.shape[2]
        
        model = LSTMForecaster(
            input_size=INPUT_SIZE,
            hidden_size=64,
            num_layers=2,
            dropout=0.2
        ).to(device)
        
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        
        best_val_loss = float('inf')
        train_losses = []
        val_losses = []
        
        start_time = time.time()
        
        for epoch in range(EPOCHS):
            train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
            val_loss = evaluate(model, val_loader, criterion, device)
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            scheduler.step(val_loss)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict().copy()
            
            if (epoch + 1) % 10 == 0 or epoch == 0:
                elapsed = time.time() - start_time
                print(f"Epoch [{epoch+1:3d}/{EPOCHS}] | Train: {train_loss:.6f} | Val: {val_loss:.6f} | Time: {elapsed:.1f}s")
        
        # Load best model
        model.load_state_dict(best_model_state)
        
        print(f"\nTraining completed in {time.time() - start_time:.1f}s")
        print(f"Best validation loss: {best_val_loss:.6f}")
        

        # -----------------------------------------------------------------
        # STEP 3: Out-of-sample predictions on test period
        # -----------------------------------------------------------------

        # Get test period data
        df_test_period = df_full[
            (df_full['Date'] >= fold_config['test_start']) & 
            (df_full['Date'] <= fold_config['test_end'])
        ].copy()


        # Optimized version
        predictions_dict, actuals_dict, all_preds, all_acts = generate_out_of_sample_predictions(model=model,
            df_test_period=df_test_period,
            df_full=df_full,
            fold_config=fold_config,
            features=features,
            scaler_X=scaler_X,
            scaler_y=scaler_y,
            ts_key_to_idx=ts_key_to_idx,
            n_ts_keys=n_ts_keys,
            seq_length=SEQ_LENGTH,
            embargo=EMBARGO,
            device=device
        )


        # -----------------------------------------------------------------
        # Create predictions DataFrame and save as CSV
        # -----------------------------------------------------------------
        predictions_data = []
        for ts_key, preds in predictions_dict.items():
            acts = actuals_dict[ts_key]
            # Get the dates for this time series in the test period
            ts_group = df_test_period[df_test_period['ts_key'] == ts_key].sort_values('Date')
            dates = ts_group['Date'].values[:len(preds)]
            
            for i, (date, pred, actual) in enumerate(zip(dates, preds, acts)):
                predictions_data.append({
                    'ts_key': ts_key,
                    'Date': pd.to_datetime(date),
                    'pred': pred,
                    'true': actual
                })
        
        predictions_df = pd.DataFrame(predictions_data)
        predictions_csv_path = os.path.join(fold_output_dir, 'predictions.csv')
        predictions_df.to_csv(predictions_csv_path, index=False)
        print(f"✓ Saved predictions to: predictions.csv ({len(predictions_df)} rows)")
        
        # -----------------------------------------------------------------
        # Calculate SMAPE distribution
        # -----------------------------------------------------------------
        smape_distribution_df = calculate_smape_distribution(predictions_dict, actuals_dict)
        
        # Calculate category counts and percentages
        category_counts = smape_distribution_df['category'].value_counts()
        total_series = len(smape_distribution_df)
        
        # Define category order
        category_order = ['<10%', '10-20%', '20-30%', '30-40%', '>40%']
        category_distribution = {}
        
        for cat in category_order:
            count = category_counts.get(cat, 0)
            percentage = (count / total_series * 100) if total_series > 0 else 0
            category_distribution[cat] = {
                'count': count,
                'percentage': percentage
            }
        
        fold_smape_distributions.append({
            'fold': fold_config['name'],
            **{f'{cat}_count': category_distribution[cat]['count'] for cat in category_order},
            **{f'{cat}_pct': category_distribution[cat]['percentage'] for cat in category_order}
        })
        
        # Save SMAPE distribution per time series
        smape_distribution_df.to_csv(os.path.join(fold_output_dir, 'smape_per_ts.csv'), index=False)
        
        print(f"\nSMAPE Distribution for {fold_config['name']}:")
        print(f"  Time series evaluated: {total_series}")
        for cat in category_order:
            count = category_distribution[cat]['count']
            pct = category_distribution[cat]['percentage']
            print(f"  {cat:>10}: {count:4d} series ({pct:5.1f}%)")
        
        # -----------------------------------------------------------------
        # STEP 4: Calculate metrics
        # -----------------------------------------------------------------
        print("\n" + "-"*60)
        print("STEP 4: Evaluation Metrics")
        print("-"*60)
        
        mse = mean_squared_error(all_acts, all_preds)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(all_acts, all_preds)
        r2 = r2_score(all_acts, all_preds)
        smape_score = smape(all_acts, all_preds)
        
        fold_metrics.append({
            'fold': fold_config['name'],
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'smape': smape_score
        })
        
        print(f"\n{fold_config['name']} Out-of-Sample Metrics:")
        print(f"  MSE:   {mse:.2f}")
        print(f"  RMSE:  {rmse:.2f}")
        print(f"  MAE:   {mae:.2f}")
        print(f"  R²:    {r2:.4f}")
        print(f"  SMAPE: {smape_score:.2f}%")
                
        torch.save({
            'model_state_dict': best_model_state,
            'input_size': INPUT_SIZE,
            'hidden_size': 64,
            'num_layers': 2,
            'dropout': 0.2,
            'scaler_X': scaler_X,
            'scaler_y': scaler_y,
            'n_ts_keys': n_ts_keys,
            'ts_key_to_idx': ts_key_to_idx,
            'seq_length': SEQ_LENGTH,
            'fold_config': fold_config,
            'metrics': fold_metrics[-1]
        }, os.path.join(fold_output_dir, 'model_checkpoint.pth'))
        
        # Visualization
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Training history
        axes[0].plot(train_losses, label='Training Loss', linewidth=2, color='#2E86AB')
        axes[0].plot(val_losses, label='Validation Loss', linewidth=2, color='#A23B72')
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Loss (MSE)', fontsize=12)
        axes[0].set_title(f'{fold_config["name"]} - Training History', fontsize=13, fontweight='bold')

    # ========================================================================
    # SMAPE DISTRIBUTION ANALYSIS
    # ========================================================================
    
    print("\n" + "="*80)
    print("SMAPE DISTRIBUTION ANALYSIS")
    print("="*80)
    
    smape_dist_df = pd.DataFrame(fold_smape_distributions)
    
    print("\nPer-Fold SMAPE Distribution:")
    category_order = ['<10%', '10-20%', '20-30%', '30-40%', '>40%']
    
    for _, row in smape_dist_df.iterrows():
        print(f"\n{row['fold']}:")
        for cat in category_order:
            count = row[f'{cat}_count']
            pct = row[f'{cat}_pct']
            print(f"  {cat:>10}: {count:4.0f} series ({pct:5.1f}%)")
    
    print("\n" + "-"*60)
    print("AVERAGE SMAPE DISTRIBUTION ACROSS FOLDS:")
    print("-"*60)
    
    for cat in category_order:
        avg_count = smape_dist_df[f'{cat}_count'].mean()
        avg_pct = smape_dist_df[f'{cat}_pct'].mean()
        std_pct = smape_dist_df[f'{cat}_pct'].std()
        print(f"  {cat:>10}: {avg_pct:5.1f}% ± {std_pct:4.1f}% ({avg_count:.0f} series avg)")
    
    # Save SMAPE distribution summary
    smape_dist_df.to_csv(os.path.join(output_path, 'smape_distribution.csv'), index=False)
    
    print(f"\n✓ Saved metrics to: {output_path}/fold_metrics.csv")
    print(f"✓ Saved summary to: {output_path}/summary_metrics.csv")
    print(f"✓ Saved SMAPE distribution to: {output_path}/smape_distribution.csv")
    
    # Predictions vs Actuals
    axes[1].scatter(all_acts, all_preds, alpha=0.3, s=10, color='#2E86AB')
    axes[1].plot([all_acts.min(), all_acts.max()], 
                    [all_acts.min(), all_acts.max()], 
                    'r--', linewidth=2, label='Perfect Prediction')
    axes[1].set_xlabel('Actual Values', fontsize=12)
    axes[1].set_ylabel('Predicted Values', fontsize=12)
    axes[1].set_title(f'{fold_config["name"]} - Predictions (R²={r2:.4f}, SMAPE={smape_score:.2f}%)', 
                        fontsize=13, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(fold_output_dir, 'fold_results.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Saved fold results to: {fold_output_dir}")
    
    # ========================================================================
    # FINAL RESULTS: Average metrics across all folds
    # ========================================================================
    
    print("\n" + "="*80)
    print("FINAL RESULTS: AVERAGE METRICS ACROSS ALL FOLDS")
    print("="*80)
    
    metrics_df = pd.DataFrame(fold_metrics)
    
    print("\nPer-Fold Metrics:")
    print(metrics_df.to_string(index=False))
    
    print("\n" + "-"*60)
    print("AVERAGE PERFORMANCE:")
    print("-"*60)
    avg_metrics = metrics_df[['mse', 'rmse', 'mae', 'r2', 'smape']].mean()
    std_metrics = metrics_df[['mse', 'rmse', 'mae', 'r2', 'smape']].std()
    
    print(f"  MSE:   {avg_metrics['mse']:.2f} ± {std_metrics['mse']:.2f}")
    print(f"  RMSE:  {avg_metrics['rmse']:.2f} ± {std_metrics['rmse']:.2f}")
    print(f"  MAE:   {avg_metrics['mae']:.2f} ± {std_metrics['mae']:.2f}")
    print(f"  R²:    {avg_metrics['r2']:.4f} ± {std_metrics['r2']:.4f}")
    print(f"  SMAPE: {avg_metrics['smape']:.2f}% ± {std_metrics['smape']:.2f}%")
    
    # Save metrics to CSV
    metrics_df.to_csv(os.path.join(output_path, 'fold_metrics.csv'), index=False)
    
    # Save summary
    summary_dict = {
        'metric': ['MSE', 'RMSE', 'MAE', 'R²', 'SMAPE'],
        'mean': [avg_metrics['mse'], avg_metrics['rmse'], avg_metrics['mae'], 
                 avg_metrics['r2'], avg_metrics['smape']],
        'std': [std_metrics['mse'], std_metrics['rmse'], std_metrics['mae'], 
                std_metrics['r2'], std_metrics['smape']]
    }
    summary_df = pd.DataFrame(summary_dict)
    summary_df.to_csv(os.path.join(output_path, 'summary_metrics.csv'), index=False)
    
    # ========================================================================
    # SAVE COMPREHENSIVE RESULTS TO TXT FILE
    # ========================================================================
    
    results_txt_path = os.path.join(output_path, 'final_results_summary.txt')
    
    with open(results_txt_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("LSTM MODEL - FINAL EVALUATION RESULTS\n")
        f.write("="*80 + "\n\n")
        
        # 1. FINAL RESULTS: AVERAGE METRICS ACROSS ALL FOLDS
        f.write("="*80 + "\n")
        f.write("1. FINAL RESULTS: AVERAGE METRICS ACROSS ALL FOLDS\n")
        f.write("="*80 + "\n\n")
        
        f.write("Per-Fold Metrics:\n")
        f.write(metrics_df.to_string(index=False) + "\n\n")
        
        # 4. AVERAGE PERFORMANCE
        f.write("-"*60 + "\n")
        f.write("4. AVERAGE PERFORMANCE:\n")
        f.write("-"*60 + "\n")
        f.write(f"  MSE:   {avg_metrics['mse']:.2f} ± {std_metrics['mse']:.2f}\n")
        f.write(f"  RMSE:  {avg_metrics['rmse']:.2f} ± {std_metrics['rmse']:.2f}\n")
        f.write(f"  MAE:   {avg_metrics['mae']:.2f} ± {std_metrics['mae']:.2f}\n")
        f.write(f"  R²:    {avg_metrics['r2']:.4f} ± {std_metrics['r2']:.4f}\n")
        f.write(f"  SMAPE: {avg_metrics['smape']:.2f}% ± {std_metrics['smape']:.2f}%\n\n")
        
        # 2. SMAPE DISTRIBUTION ANALYSIS per fold
        f.write("="*80 + "\n")
        f.write("2. SMAPE DISTRIBUTION ANALYSIS PER FOLD\n")
        f.write("="*80 + "\n\n")
        
        for _, row in smape_dist_df.iterrows():
            f.write(f"{row['fold']}:\n")
            for cat in category_order:
                count = row[f'{cat}_count']
                pct = row[f'{cat}_pct']
                f.write(f"  {cat:>10}: {count:4.0f} series ({pct:5.1f}%)\n")
            f.write("\n")
        
        # 3. AVERAGE SMAPE DISTRIBUTION ACROSS FOLDS
        f.write("-"*60 + "\n")
        f.write("3. AVERAGE SMAPE DISTRIBUTION ACROSS FOLDS:\n")
        f.write("-"*60 + "\n")
        
        for cat in category_order:
            avg_count = smape_dist_df[f'{cat}_count'].mean()
            avg_pct = smape_dist_df[f'{cat}_pct'].mean()
            std_pct = smape_dist_df[f'{cat}_pct'].std()
            f.write(f"  {cat:>10}: {avg_pct:5.1f}% ± {std_pct:4.1f}% ({avg_count:.0f} series avg)\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("END OF REPORT\n")
        f.write("="*80 + "\n")
    
    print(f"\n✓ Saved metrics to: {output_path}/fold_metrics.csv")
    print(f"✓ Saved summary to: {output_path}/summary_metrics.csv")
    print(f"✓ Saved final results summary to: {output_path}/final_results_summary.txt")
    print("\n" + "="*80)
    
    
    