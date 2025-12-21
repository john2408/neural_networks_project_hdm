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


class TimeSeriesDatasetVectorized(Dataset):
    """
    VECTORIZED Dataset for UNIVARIATE time series forecasting.
    
    This is the FAST approach used by Nixtla's NeuralForecast models.
    Instead of creating N_series × N_windows individual samples, this creates
    N_windows time blocks, where each block contains ALL series simultaneously.
    
    Key Differences from TimeSeriesDataset:
    ----------------------------------------
    1. Dataset size = N_windows (not N_series × N_windows)
    2. Each sample contains ALL time series at one time window
    3. Batching happens across TIME dimension (all series processed together)
    4. Results in 500x fewer forward passes with 500x larger effective batch size
    
    Performance Impact:
    ------------------
    Traditional approach: 1000 series × 40 windows = 40,000 samples
                         → 1,250 forward passes (batch_size=32)
                         → Small matrix multiplications
                         → Poor GPU utilization (~5-15%)
    
    Vectorized approach:  40 time windows (each with 1000 series)
                         → 2.5 forward passes (batch_size=16, but 16×1000=16,000 predictions)
                         → Massive matrix multiplications
                         → Excellent GPU utilization (~80-95%)
                         → 20-50x faster in practice!
    
    Data Structure:
    --------------
    Sample at time index t:
        Input X:  All series, time window [t:t+seq_length]
                  Shape: (n_series, seq_length, n_features)
        Target y: All series, value at t+seq_length+embargo
                  Shape: (n_series,)
    
    Example with 1000 series, seq_length=6:
        len(dataset) = 40 time windows
        dataset[0] returns:
            X: (1000, 6, 1)  # 1000 series × 6 timesteps × 1 feature
            y: (1000,)        # 1000 target values
    
    Features per series per timestep:
        - Value (1)
        - Total: 1 feature
    
    Note: No one-hot encoding needed since all series are in the same batch!
    Note: Only accepts dataframes with columns: ['ts_key', 'Date', 'Value']
    """
    
    def __init__(self, df, seq_length=6, embargo=1,
                 train=True, train_ratio=0.8, scaler_X=None, scaler_y=None):
        """
        Args:
            df: DataFrame with ONLY [Date, ts_key, Value] columns
            seq_length: Lookback window size
            embargo: Gap between last observation and prediction target
            train: If True, create training set
            train_ratio: Train/test split ratio
            scaler_X: Pre-fitted StandardScaler for features (for test set)
            scaler_y: Pre-fitted StandardScaler for targets (for test set)
        """
        # Validate dataframe columns
        required_cols = {'Date', 'ts_key', 'Value'}
        if set(df.columns) != required_cols:
            raise ValueError(
                f"DataFrame must have exactly these columns: {required_cols}\n"
                f"Found: {set(df.columns)}"
            )
        
        self.seq_length = seq_length
        self.embargo = embargo
        
        # Get unique time series
        unique_ts_keys = sorted(df['ts_key'].unique())
        self.n_series = len(unique_ts_keys)
        self.ts_key_to_idx = {key: idx for idx, key in enumerate(unique_ts_keys)}
        
        print(f"\n{'='*70}")
        print("Creating VECTORIZED dataset (Nixtla-style)")
        print(f"{'='*70}")
        print(f"  Time series: {self.n_series:,}")
        print(f"  Univariate forecasting: Value only (no exogenous features)")
        print(f"  Sequence length: {seq_length}, Embargo: {embargo}")
        
        # =====================================================================
        # STEP 1: Pivot data - align all series by date
        # =====================================================================
        print("\n[1/5] Pivoting data to align all series by date...")
        
        df_pivot_values = df.pivot_table(
            index='Date',
            columns='ts_key',
            values='Value',
            aggfunc='first'
        ).sort_index()
        
        # Extract dates and values matrix
        dates = pd.to_datetime(df_pivot_values.index)
        values_matrix = df_pivot_values.values  # Shape: (n_timesteps, n_series)
        
        print(f"  Total timesteps: {len(dates)}")
        print(f"  Value matrix shape: {values_matrix.shape}")
        
        # Check for missing data
        nan_count = np.isnan(values_matrix).sum()
        if nan_count > 0:
            print(f"  WARNING: Found {nan_count} NaN values in pivoted data!")
            print(f"  Filling NaN with forward fill + backward fill...")
            df_pivot_values = df_pivot_values.fillna(method='ffill').fillna(method='bfill')
            values_matrix = df_pivot_values.values
            remaining_nans = np.isnan(values_matrix).sum()
            if remaining_nans > 0:
                raise ValueError(f"Still {remaining_nans} NaN values after filling!")
        
        # =====================================================================
        # STEP 2: Create time-indexed windows
        # =====================================================================
        print("\n[2/5] Creating time-indexed windows...")
        
        min_length = seq_length + embargo + 1
        if len(dates) < min_length:
            raise ValueError(f"Not enough timesteps. Need {min_length}, have {len(dates)}")
        
        self.X = []  # Will store (n_windows, n_series, seq_length, n_features)
        self.y = []  # Will store (n_windows, n_series)
        
        # Create one sample per time window (NOT per series!)
        n_windows = len(dates) - seq_length - embargo
        
        for t in range(n_windows):
            # Extract window using NumPy slicing: (seq_length, n_series) -> (n_series, seq_length, 1)
            window_slice = values_matrix[t:t+seq_length, :].T[:, :, np.newaxis]
            self.X.append(window_slice)
            
            # Target: all series at prediction timestep
            target_idx = t + seq_length + embargo - 1
            self.y.append(values_matrix[target_idx, :])
        
        self.X = np.array(self.X, dtype=np.float32)  # (n_windows, n_series, seq_length, n_features)
        self.y = np.array(self.y, dtype=np.float32)  # (n_windows, n_series)
        
        print(f"  Created {len(self.X)} time windows")
        print(f"  Input shape: {self.X.shape}")
        print(f"  Output shape: {self.y.shape}")
        
        n_features_total = 1  # Value only
        print(f"  Features per series: {n_features_total}")
        print(f"    - Value: 1")
        
        # =====================================================================
        # STEP 3: Performance comparison
        # =====================================================================
        traditional_samples = self.n_series * n_windows
        reduction_factor = traditional_samples / len(self.X)
        
        print(f"\n[3/5] Performance comparison:")
        print(f"  Traditional approach would create: {traditional_samples:,} samples")
        print(f"  Vectorized approach creates: {len(self.X)} time windows")
        print(f"  Sample reduction: {reduction_factor:.0f}x fewer!")
        print(f"  → With batch_size=16: {reduction_factor/16:.0f}x fewer forward passes")
        print(f"  → Effective batch size: 16 × {self.n_series:,} = {16 * self.n_series:,} predictions per batch")
        
        # =====================================================================
        # STEP 4: Train-test split (chronological)
        # =====================================================================
        print(f"\n[4/5] Train-test split...")
        
        n_samples = len(self.X)
        train_size = int(n_samples * train_ratio)
        
        if train:
            self.X = self.X[:train_size]
            self.y = self.y[:train_size]
            print(f"  Training set: {len(self.X)} time windows")
        else:
            self.X = self.X[train_size:]
            self.y = self.y[train_size:]
            print(f"  Test set: {len(self.X)} time windows")
        
        # =====================================================================
        # STEP 5: Standardization
        # =====================================================================
        print(f"\n[5/5] Standardizing features...")
        
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        
        # Reshape for scaling: (n_windows, n_series, seq_length, 1)
        # → (n_windows × n_series × seq_length, 1)
        n_windows, n_series, seq_len, n_feats = self.X.shape
        
        # Flatten all dimensions except last for scaling
        X_flat = self.X.reshape(-1, 1)
        
        if train:
            # Fit and transform
            X_scaled = self.scaler_X.fit_transform(X_flat)
            self.X = X_scaled.reshape(n_windows, n_series, seq_len, 1)
            
            # Scale targets: (n_windows, n_series) → (n_windows × n_series, 1)
            y_reshaped = self.y.reshape(-1, 1)
            y_scaled = self.scaler_y.fit_transform(y_reshaped)
            self.y = y_scaled.reshape(n_windows, n_series)
            
            print(f"  Fitted scalers on training data")
        else:
            # Transform using provided scalers
            if scaler_X is not None and scaler_y is not None:
                self.scaler_X = scaler_X
                self.scaler_y = scaler_y
                
                X_scaled = self.scaler_X.transform(X_flat)
                self.X = X_scaled.reshape(n_windows, n_series, seq_len, 1)
                
                y_reshaped = self.y.reshape(-1, 1)
                y_scaled = self.scaler_y.transform(y_reshaped)
                self.y = y_scaled.reshape(n_windows, n_series)
                
                print(f"  Applied training scalers to test data")
            else:
                print(f"  WARNING: No scalers provided for test set!")
        
        # Final validation
        X_nans = np.isnan(self.X).sum()
        y_nans = np.isnan(self.y).sum()
        if X_nans > 0 or y_nans > 0:
            raise ValueError(f"NaN values after scaling! X: {X_nans}, y: {y_nans}")
        
        print(f"  ✓ Standardization complete, no NaN values")
        print(f"\n{'='*70}")
        print("Dataset ready for training!")
        print(f"{'='*70}\n")
    
    def __len__(self):
        """Returns number of time windows (NOT number of series × windows)."""
        return len(self.X)
    
    def __getitem__(self, idx):
        """
        Get all series at time window idx.
        
        Returns:
            X: (n_series, seq_length, n_features) - all series, one time window
            y: (n_series,) - all series targets
        """
        return torch.FloatTensor(self.X[idx]), torch.FloatTensor(self.y[idx])


class TimeSeriesDatasetFlattened(Dataset):
    """
    Dataset for FLATTENED MULTIVARIATE time series forecasting.
    
    Instead of one-hot encoding, this dataset organizes data by timestep,
    with ALL time series values flattened into each sample.
    
    Key differences from TimeSeriesDataset:
    - Each sample contains ALL time series at a sequence of timesteps
    - No one-hot encoding
    - Data organized by global timestep, not by individual series
    - More efficient for many time series (100s-1000s)
    
    Structure:
    --------
    Sample format (per timestep window):
        Input X: [val₁(t-L), ..., val₁(t-1), val₂(t-L), ..., valₙ(t-1),
                  feat₁₁(t-L), ..., featₙₖ(t-1), year, month]
        Target y: [val₁(t), val₂(t), ..., valₙ(t)]
    
    Example with 1000 series, 5 features, seq_length=6:
        Input shape:  (batch, 6, 1000 + 1000*5 + 2) = (batch, 6, 6002)
        Output shape: (batch, 1000)
    """
    def __init__(self, df, feature_cols=None, seq_length=6, embargo=1, 
                 train=True, train_ratio=0.8, scaler_X=None, scaler_y=None):
        """
        Args:
            df: DataFrame with [Date, ts_key, Value, ...features]
            feature_cols: List of additional feature column names
            seq_length: Lookback window size
            embargo: Gap between last observation and prediction
            train: If True, create training set
            train_ratio: Train/test split ratio
            scaler_X: Pre-fitted StandardScaler for features
            scaler_y: Pre-fitted StandardScaler for targets
        """
        self.seq_length = seq_length
        self.embargo = embargo
        
        # Auto-detect feature columns
        if feature_cols is None:
            self.feature_cols = [col for col in df.columns 
                                if col not in ['Date', 'ts_key', 'Value']]
        else:
            self.feature_cols = feature_cols
        
        self.n_features_additional = len(self.feature_cols)
        
        # Get unique time series and dates
        unique_ts_keys = sorted(df['ts_key'].unique())
        self.n_series = len(unique_ts_keys)
        self.ts_key_to_idx = {key: idx for idx, key in enumerate(unique_ts_keys)}
        
        print(f"\nCreating FLATTENED dataset:")
        print(f"  Time series: {self.n_series}")
        print(f"  Additional features per series: {self.n_features_additional}")
        print(f"  Sequence length: {seq_length}, Embargo: {embargo}")
        
        # Pivot data to have all series aligned by date
        # This assumes all series have observations at the same dates
        df_pivot = df.pivot_table(
            index='Date',
            columns='ts_key',
            values='Value',
            aggfunc='first'
        ).sort_index()
        
        # Get feature DataFrames for each feature column
        df_features = {}
        if self.n_features_additional > 0:
            for feat in self.feature_cols:
                df_features[feat] = df.pivot_table(
                    index='Date',
                    columns='ts_key',
                    values=feat,
                    aggfunc='first'
                ).sort_index()
        
        # Extract dates and check for missing data
        dates = pd.to_datetime(df_pivot.index)
        values_matrix = df_pivot.values  # Shape: (n_timesteps, n_series)
        
        print(f"  Total timesteps: {len(dates)}")
        nan_count = np.isnan(values_matrix).sum()
        print(f"  Missing values in pivoted data: {nan_count}")
        
        if nan_count > 0:
            raise ValueError(
                f"Found {nan_count} NaN values in pivoted data! "
                f"All time series must have the same length. "
                f"Please preprocess data to pad time series to equal length before creating dataset."
            )
        
        # Check if we have enough data
        min_length = seq_length + embargo + 1
        if len(dates) < min_length:
            raise ValueError(f"Not enough timesteps. Need {min_length}, have {len(dates)}")
        
        # Create sliding windows
        self.X = []
        self.y = []
        
        for i in range(len(dates) - seq_length - embargo):
            # Build input sequence
            sequence_features = []
            
            for t in range(seq_length):
                idx = i + t
                
                # Collect values for all series at this timestep
                timestep_features = []
                
                # Add all series values
                timestep_features.extend(values_matrix[idx, :])
                
                # Add all series features
                if self.n_features_additional > 0:
                    for feat in self.feature_cols:
                        feat_values = df_features[feat].iloc[idx].values
                        timestep_features.extend(feat_values)
                
                # Add temporal features (same for all series at this timestep)
                date = dates[idx]
                timestep_features.extend([date.year, date.month])
                
                sequence_features.append(timestep_features)
            
            self.X.append(np.array(sequence_features, dtype=np.float32))
            
            # Target: all series values at prediction timestep
            target_idx = i + seq_length + embargo - 1
            self.y.append(values_matrix[target_idx, :])
        
        self.X = np.array(self.X, dtype=np.float32)  # (n_samples, seq_length, n_features_total)
        self.y = np.array(self.y, dtype=np.float32)  # (n_samples, n_series)
        
        print(f"  Created {len(self.X)} samples")
        print(f"  Input shape: {self.X.shape}")
        print(f"  Output shape: {self.y.shape}")
        
        # Validate no NaN in created samples
        X_nans = np.isnan(self.X).sum()
        y_nans = np.isnan(self.y).sum()
        if X_nans > 0 or y_nans > 0:
            raise ValueError(f"NaN values found in samples! X: {X_nans}, y: {y_nans}")
        
        # Train-test split (chronological)
        n_samples = len(self.X)
        train_size = int(n_samples * train_ratio)
        
        if train:
            self.X = self.X[:train_size]
            self.y = self.y[:train_size]
        else:
            self.X = self.X[train_size:]
            self.y = self.y[train_size:]
        
        # Standardization
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        
        # Separate features for scaling
        # Don't scale temporal features (year, month)
        n_samples, seq_len, n_feats = self.X.shape
        n_continuous = self.n_series * (1 + self.n_features_additional)
        
        # Extract continuous features (values + features, exclude year/month)
        X_continuous = self.X[:, :, :n_continuous].reshape(-1, n_continuous)
        X_temporal = self.X[:, :, n_continuous:].reshape(-1, 2)
        
        if train:
            # Fit and transform on training data
            X_continuous_scaled = self.scaler_X.fit_transform(X_continuous)
            X_continuous_scaled = X_continuous_scaled.reshape(n_samples, seq_len, n_continuous)
            X_temporal_reshaped = X_temporal.reshape(n_samples, seq_len, 2)
            
            self.X = np.concatenate([X_continuous_scaled, X_temporal_reshaped], axis=2)
            self.y = self.scaler_y.fit_transform(self.y)
        else:
            # Transform using training scalers
            if scaler_X is not None and scaler_y is not None:
                self.scaler_X = scaler_X
                self.scaler_y = scaler_y
                
                X_continuous_scaled = self.scaler_X.transform(X_continuous)
                X_continuous_scaled = X_continuous_scaled.reshape(n_samples, seq_len, n_continuous)
                X_temporal_reshaped = X_temporal.reshape(n_samples, seq_len, 2)
                
                self.X = np.concatenate([X_continuous_scaled, X_temporal_reshaped], axis=2)
                self.y = self.scaler_y.transform(self.y)
        
        # Final validation after scaling
        X_nans_final = np.isnan(self.X).sum()
        y_nans_final = np.isnan(self.y).sum()
        if X_nans_final > 0 or y_nans_final > 0:
            raise ValueError(f"NaN values after scaling! X: {X_nans_final}, y: {y_nans_final}")
        
        print(f"  Scaled dataset size: {len(self.X)} samples")
        print(f"  ✓ No NaN values in final dataset")
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.X[idx]), torch.FloatTensor(self.y[idx])


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


def generate_out_of_sample_predictions_vectorized(
    model, 
    df_test_period,
    df_full, 
    fold_config, 
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
    
    VECTORIZED VERSION - For univariate forecasting with TimeSeriesDatasetVectorized.
    Uses only Value column (no exogenous features, no temporal encoding, no one-hot).
    
    This function optimizes the prediction process by:
    1. Batches all time series together for each prediction step
    2. Runs ONE forward pass for all time series simultaneously
    3. Updates all histories together
    
    Args:
        model: Trained PyTorch model
        df_test_period: Test period dataframe with ['Date', 'ts_key', 'Value']
        df_full: Complete dataframe with all data
        fold_config: Dict with 'test_start' and 'test_end' dates
        scaler_X: Fitted StandardScaler for Value
        scaler_y: Fitted StandardScaler for target
        ts_key_to_idx: Dict mapping ts_key to index
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
    print("STEP 3: Out-of-sample predictions (VECTORIZED)")
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
        
        # Initialize time series data (only Value, no features)
        ts_data[ts_key] = {
            'ts_key_idx': ts_key_to_idx[ts_key],
            'recent_values': hist_data['Value'].values[-(seq_length + embargo):].copy(),
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
            
            # Build sequence for this time series (only values)
            sequence = []
            
            for i in range(seq_length):
                idx = len(data['recent_values']) - seq_length - embargo + i
                
                if idx < 0 or idx >= len(data['recent_values']):
                    break
                
                value = data['recent_values'][idx]
                
                # Feature vector is just the value (shape will be (seq_length, 1))
                sequence.append([value])
            
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
        batch_array = np.array(batch_sequences, dtype=np.float32)  # (batch_size, seq_length, 1)
        
        # Scale features
        batch_size, seq_len, n_feats = batch_array.shape
        batch_flat = batch_array.reshape(-1, 1)
        batch_scaled = scaler_X.transform(batch_flat)
        batch_scaled = batch_scaled.reshape(batch_size, seq_len, 1)
        
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
            
            # Get actual value for this prediction
            pred_date = pd.to_datetime(data['test_dates'][step])
            
            # Use actual value from test set
            actual_value = data['test_actuals'][step]
            
            # Update history (only values, no features)
            data['recent_values'] = np.append(data['recent_values'], actual_value)
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


def create_model_from_trial(model_type, trial, n_series, n_features_additional, seq_length):
    """
    Create a model instance based on trial hyperparameters.
    
    Args:
        model_type: Model architecture name
        trial: Optuna trial object
        n_series: Number of time series
        n_features_additional: Number of additional features per series
        seq_length: Sequence length
    
    Returns:
        model: Instantiated model
        hyperparams: Dict of hyperparameters used
    """
    from neuralts.core.models import (
        MLPMultivariate, LSTMForecasterMultivariate,
        RNNForecasterMultivariate, GRUForecasterMultivariate,
        CNN1DForecasterMultivariate, TransformerForecasterMultivariate
    )
    
    if model_type == 'MLPMultivariate':
        num_layers = trial.suggest_int('num_layers', 1, 4)
        hidden_size = trial.suggest_categorical('hidden_size', [128, 256, 512, 1024])
        dropout = trial.suggest_float('dropout', 0.1, 0.5)
        
        model = MLPMultivariate(
            input_size=seq_length,
            n_series=n_series,
            n_features_additional=n_features_additional,
            num_layers=num_layers,
            hidden_size=hidden_size,
            dropout=dropout
        )
        hyperparams = {'num_layers': num_layers, 'hidden_size': hidden_size, 'dropout': dropout}
    
    elif model_type == 'LSTMMultivariate':
        num_layers = trial.suggest_int('num_layers', 1, 3)
        hidden_size = trial.suggest_categorical('hidden_size', [64, 128, 256])
        dropout = trial.suggest_float('dropout', 0.1, 0.5)
        
        model = LSTMForecasterMultivariate(
            input_size=seq_length,
            n_series=n_series,
            n_features_additional=n_features_additional,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout
        )
        hyperparams = {'num_layers': num_layers, 'hidden_size': hidden_size, 'dropout': dropout}
    
    elif model_type == 'RNNMultivariate':
        num_layers = trial.suggest_int('num_layers', 1, 3)
        hidden_size = trial.suggest_categorical('hidden_size', [64, 128, 256])
        dropout = trial.suggest_float('dropout', 0.1, 0.5)
        
        model = RNNForecasterMultivariate(
            input_size=seq_length,
            n_series=n_series,
            n_features_additional=n_features_additional,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout
        )
        hyperparams = {'num_layers': num_layers, 'hidden_size': hidden_size, 'dropout': dropout}
    
    elif model_type == 'GRUMultivariate':
        num_layers = trial.suggest_int('num_layers', 1, 3)
        hidden_size = trial.suggest_categorical('hidden_size', [64, 128, 256])
        dropout = trial.suggest_float('dropout', 0.1, 0.5)
        
        model = GRUForecasterMultivariate(
            input_size=seq_length,
            n_series=n_series,
            n_features_additional=n_features_additional,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout
        )
        hyperparams = {'num_layers': num_layers, 'hidden_size': hidden_size, 'dropout': dropout}
    
    elif model_type == 'CNN1DMultivariate':
        num_layers = trial.suggest_int('num_layers', 2, 4)
        hidden_size = trial.suggest_categorical('hidden_size', [32, 64, 128])
        dropout = trial.suggest_float('dropout', 0.1, 0.5)
        
        model = CNN1DForecasterMultivariate(
            input_size=seq_length,
            n_series=n_series,
            n_features_additional=n_features_additional,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout
        )
        hyperparams = {'num_layers': num_layers, 'hidden_size': hidden_size, 'dropout': dropout}
    
    elif model_type == 'TransformerMultivariate':
        d_model = trial.suggest_categorical('d_model', [32, 64, 128])
        nhead = trial.suggest_categorical('nhead', [2, 4, 8])
        num_layers = trial.suggest_int('num_layers', 1, 3)
        dim_feedforward = trial.suggest_categorical('dim_feedforward', [128, 256, 512])
        dropout = trial.suggest_float('dropout', 0.1, 0.5)
        
        # Ensure d_model is divisible by nhead
        if d_model % nhead != 0:
            d_model = nhead * (d_model // nhead)
        
        model = TransformerForecasterMultivariate(
            input_size=seq_length,
            n_series=n_series,
            n_features_additional=n_features_additional,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        hyperparams = {
            'd_model': d_model,
            'nhead': nhead,
            'num_layers': num_layers,
            'dim_feedforward': dim_feedforward,
            'dropout': dropout
        }
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model, hyperparams


def create_univariate_model_from_trial(model_type, trial, input_size, seq_length):
    """
    Create a univariate model instance based on trial hyperparameters.
    
    Args:
        model_type: Model architecture name (LSTM, RNN, GRU, CNN1D, MLP, Transformer, TransformerCLS)
        trial: Optuna trial object
        input_size: Number of input features (including one-hot encoding)
        seq_length: Sequence length
    
    Returns:
        model: Instantiated model
        hyperparams: Dict of hyperparameters used
    """
    from neuralts.core.models import (
        LSTMForecaster, RNNForecaster, GRUForecaster,
        CNN1DForecaster, MLPForecaster,
        TransformerForecaster, TransformerForecasterCLS
    )
    
    if model_type == 'LSTM':
        num_layers = trial.suggest_int('num_layers', 1, 3)
        hidden_size = trial.suggest_categorical('hidden_size', [32, 64, 128, 256])
        dropout = trial.suggest_float('dropout', 0.1, 0.5)
        
        model = LSTMForecaster(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout
        )
        hyperparams = {'num_layers': num_layers, 'hidden_size': hidden_size, 'dropout': dropout}
    
    elif model_type == 'RNN':
        num_layers = trial.suggest_int('num_layers', 1, 3)
        hidden_size = trial.suggest_categorical('hidden_size', [32, 64, 128, 256])
        dropout = trial.suggest_float('dropout', 0.1, 0.5)
        
        model = RNNForecaster(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout
        )
        hyperparams = {'num_layers': num_layers, 'hidden_size': hidden_size, 'dropout': dropout}
    
    elif model_type == 'GRU':
        num_layers = trial.suggest_int('num_layers', 1, 3)
        hidden_size = trial.suggest_categorical('hidden_size', [32, 64, 128, 256])
        dropout = trial.suggest_float('dropout', 0.1, 0.5)
        
        model = GRUForecaster(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout
        )
        hyperparams = {'num_layers': num_layers, 'hidden_size': hidden_size, 'dropout': dropout}
    
    elif model_type == 'CNN1D':
        num_layers = trial.suggest_int('num_layers', 2, 4)
        hidden_size = trial.suggest_categorical('hidden_size', [32, 64, 128, 256])
        dropout = trial.suggest_float('dropout', 0.1, 0.5)
        
        model = CNN1DForecaster(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout
        )
        hyperparams = {'num_layers': num_layers, 'hidden_size': hidden_size, 'dropout': dropout}
    
    elif model_type == 'MLP':
        num_layers = trial.suggest_int('num_layers', 2, 4)
        hidden_size = trial.suggest_categorical('hidden_size', [128, 256, 512, 1024])
        dropout = trial.suggest_float('dropout', 0.1, 0.5)
        
        model = MLPForecaster(
            input_size=input_size,
            seq_length=seq_length,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout
        )
        hyperparams = {'num_layers': num_layers, 'hidden_size': hidden_size, 'dropout': dropout}
    
    elif model_type == 'Transformer':
        d_model = trial.suggest_categorical('d_model', [32, 64, 128])
        nhead = trial.suggest_categorical('nhead', [2, 4, 8])
        num_layers = trial.suggest_int('num_layers', 1, 3)
        dim_feedforward = trial.suggest_categorical('dim_feedforward', [128, 256, 512])
        dropout = trial.suggest_float('dropout', 0.1, 0.5)
        
        # Ensure d_model is divisible by nhead
        if d_model % nhead != 0:
            d_model = nhead * (d_model // nhead)
        
        model = TransformerForecaster(
            input_size=input_size,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        hyperparams = {
            'd_model': d_model,
            'nhead': nhead,
            'num_layers': num_layers,
            'dim_feedforward': dim_feedforward,
            'dropout': dropout
        }
    
    elif model_type == 'TransformerCLS':
        d_model = trial.suggest_categorical('d_model', [32, 64, 128])
        nhead = trial.suggest_categorical('nhead', [2, 4, 8])
        num_layers = trial.suggest_int('num_layers', 1, 3)
        dim_feedforward = trial.suggest_categorical('dim_feedforward', [128, 256, 512])
        dropout = trial.suggest_float('dropout', 0.1, 0.5)
        
        # Ensure d_model is divisible by nhead
        if d_model % nhead != 0:
            d_model = nhead * (d_model // nhead)
        
        model = TransformerForecasterCLS(
            input_size=input_size,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        hyperparams = {
            'd_model': d_model,
            'nhead': nhead,
            'num_layers': num_layers,
            'dim_feedforward': dim_feedforward,
            'dropout': dropout
        }
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model, hyperparams


def train_with_early_stopping(model, train_loader, val_loader, device, 
                               learning_rate=0.001, weight_decay=1e-5,
                               max_epochs=30, patience=5):
    """
    Train model with early stopping.
    
    Args:
        model: PyTorch model
        train_loader: Training data loader
        val_loader: Validation data loader
        device: torch.device
        learning_rate: Learning rate
        weight_decay: L2 regularization
        max_epochs: Maximum training epochs
        patience: Early stopping patience
    
    Returns:
        best_val_loss: Best validation loss achieved
        best_model_state: State dict of best model
    """
    import torch.nn as nn
    import torch.optim as optim
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(max_epochs):
        # Training
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validation
        val_loss = evaluate(model, val_loader, criterion, device)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            break
    
    return best_val_loss, best_model_state


def train_with_early_stopping_vectorized(model, train_loader, val_loader, device, seq_length,
                                          learning_rate=0.001, weight_decay=1e-5,
                                          max_epochs=30, patience=5):
    """
    Train model with early stopping using VECTORIZED batching.
    
    This function handles the special batching format from TimeSeriesDatasetVectorized:
    - Input batches: (batch_time, n_series, seq_length, n_features)
    - Reshapes to: (batch_time * n_series, seq_length, n_features) for model
    
    Args:
        model: PyTorch model
        train_loader: Training data loader (from TimeSeriesDatasetVectorized)
        val_loader: Validation data loader (from TimeSeriesDatasetVectorized)
        device: torch.device
        seq_length: Sequence length for reshaping
        learning_rate: Learning rate
        weight_decay: L2 regularization
        max_epochs: Maximum training epochs
        patience: Early stopping patience
    
    Returns:
        best_val_loss: Best validation loss achieved
        best_model_state: State dict of best model
    """
    import torch.nn as nn
    import torch.optim as optim
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(max_epochs):
        # Training with vectorized batching
        model.train()
        total_train_loss = 0
        
        for X_batch, y_batch in train_loader:
            # X_batch: (batch_time, n_series, seq_length, n_features)
            # y_batch: (batch_time, n_series)
            
            # Reshape to (batch_time * n_series, seq_length, n_features)
            n_features = X_batch.shape[3]
            X_batch = X_batch.reshape(-1, seq_length, n_features).to(device)
            y_batch = y_batch.reshape(-1, 1).to(device)
            
            optimizer.zero_grad()
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_train_loss += loss.item()
        
        train_loss = total_train_loss / len(train_loader)
        
        # Validation with vectorized batching
        model.eval()
        total_val_loss = 0
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                n_features = X_batch.shape[3]
                X_batch = X_batch.reshape(-1, seq_length, n_features).to(device)
                y_batch = y_batch.reshape(-1, 1).to(device)
                
                predictions = model(X_batch)
                loss = criterion(predictions, y_batch)
                total_val_loss += loss.item()
        
        val_loss = total_val_loss / len(val_loader)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            break
    
    return best_val_loss, best_model_state
