import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler


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
    # Enable evaluation mode (disables dropout and batch norm updates)
    model.eval()
    
    total_loss = 0
    
    # Disable gradient calculation for evaluation (saves memory and computation)
    with torch.no_grad():
        for X_batch, y_batch in loader:
            # Move data to MPS/CUDA if available
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            # Pass the batch through the model
            predictions = model(X_batch)
            
            # Calculate loss
            loss = criterion(predictions, y_batch)
            
            # Add batch loss to total loss
            total_loss += loss.item()
    
    # Divides by number of batches to get average loss per batch
    return total_loss / len(loader)


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
