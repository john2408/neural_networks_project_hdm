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
    Dataset for MULTIVARIATE time series with sliding window approach.
    Each sample: [features(t-6), ..., features(t-1)] -> target(t)
    Features include: Value + one-hot encoded ts_key + year + month
    Memory efficient: doesn't pivot the entire dataframe
    """
    def __init__(self, df, seq_length=6, train=True, train_ratio=0.8):
        """
        Args:
            df: DataFrame with columns [Date, ts_key, Value]
            seq_length: Lookback window (6 months)
            train: If True, create training set, else test set
            train_ratio: Train/test split ratio
        """
        self.seq_length = seq_length
        self.X = []
        self.y = []
        self.ts_keys_list = []
        
        # Create one-hot encoding mapping for ts_keys
        unique_ts_keys = sorted(df['ts_key'].unique())
        self.ts_key_to_idx = {key: idx for idx, key in enumerate(unique_ts_keys)}
        self.n_ts_keys = len(unique_ts_keys)
        
        print(f"Creating one-hot encoding for {self.n_ts_keys} time series...")
        
        # Group by time series key
        grouped = df.sort_values('Date').groupby('ts_key')
        
        for ts_key, group in grouped:
            values = group['Value'].values
            dates = pd.to_datetime(group['Date'].values)
            
            # Skip if not enough data
            if len(values) < seq_length + 1:
                continue
            
            # Get one-hot encoding for this ts_key
            ts_key_idx = self.ts_key_to_idx[ts_key]
            ts_key_onehot = np.zeros(self.n_ts_keys, dtype=np.float32)
            ts_key_onehot[ts_key_idx] = 1.0
            
            # Create sliding windows
            for i in range(len(values) - seq_length):
                # Collect features for each timestep in the window
                window_features = []
                
                for j in range(seq_length):
                    # Features: [value, year, month, ts_key_onehot...]
                    date = dates[i + j]
                    features = np.concatenate([
                        [values[i + j]],           # Value
                        [date.year],               # Year
                        [date.month],              # Month
                        ts_key_onehot              # One-hot encoded ts_key
                    ])
                    window_features.append(features)
                
                self.X.append(np.array(window_features))  # Shape: (6, n_features)
                self.y.append(values[i + seq_length])      # Target value
                self.ts_keys_list.append(ts_key)
        
        self.X = np.array(self.X, dtype=np.float32)  # Shape: (n_samples, 6, n_features)
        self.y = np.array(self.y, dtype=np.float32)  # Shape: (n_samples,)
        
        print(f"Created {len(self.X)} samples with feature dimension: {self.X.shape[2]}")
        
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
        
        # Standardize features (only Value, year, month - not one-hot)
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        
        # Extract continuous features (Value, year, month) for scaling
        n_samples, seq_len, n_features = self.X.shape
        n_continuous = 3  # Value, year, month
        
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
            # For test set, scalers will be set from training set
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


if __name__ == "__main__":

    # Load data
    import os
    cwd = os.getcwd()
    full_path = os.path.join(cwd, "data", "processed", "historical_kba_data.parquet")
    output_path = os.path.join(cwd, "models", "lstm")
    os.makedirs(output_path, exist_ok=True)

    df = pd.read_parquet(full_path, engine='fastparquet')
    df["ts_key_size"] = df.groupby('ts_key')['ts_key'].transform('size')

    # Filter ts_keys with at least 12 entries
    df = df[df['ts_key_size'] >= 12].copy()

    columns = ['Date','ts_key', 'Value']
    df = df[columns].copy()
    
    print("="*60)
    print("DATA OVERVIEW")
    print("="*60)
    print(f"Original data shape: {df.shape}")
    print(f"Unique time series: {df.ts_key.nunique()}")
    print(f"Date range: {df.Date.min()} to {df.Date.max()}")
    print(f"Total observations: {len(df):,}")
    
    # STEP 1: Create datasets with one-hot encoding
    print("\n" + "="*60)
    print("STEP 1: Creating multivariate dataset")
    print("="*60)
    print("Features per timestep:")
    print("  - Value (1)")
    print("  - Year (1)")
    print("  - Month (1)")
    print(f"  - ts_key one-hot ({df.ts_key.nunique()})")
    print(f"  = Total: {3 + df.ts_key.nunique()} features")
    
    SEQ_LENGTH = 8
    TRAIN_RATIO = 0.8
    
    train_dataset = TimeSeriesDataset(
        df,
        seq_length=SEQ_LENGTH,
        train=True,
        train_ratio=TRAIN_RATIO
    )
    
    print(f"\nTraining samples: {len(train_dataset):,}")
    print(f"Sample input shape: {train_dataset.X[0].shape}")  # (6, n_features)
    print(f"  → (seq_length={SEQ_LENGTH}, features={train_dataset.X.shape[2]})")
    
    # Save scalers and metadata for test set
    scaler_X = train_dataset.scaler_X
    scaler_y = train_dataset.scaler_y
    n_ts_keys = train_dataset.n_ts_keys
    ts_key_to_idx = train_dataset.ts_key_to_idx
    
    # Store ts_key_to_idx as pickle object
    with open('models/lstm/ts_key_to_idx.pkl', 'wb') as f:
        pickle.dump(ts_key_to_idx, f)

    test_dataset = TimeSeriesDataset(
        df,
        seq_length=SEQ_LENGTH,
        train=False,
        train_ratio=TRAIN_RATIO
    )
    
    # Apply training scalers to test set
    n_samples, seq_len, n_features = test_dataset.X.shape
    n_continuous = 3  # Value, year, month
    
    X_continuous = test_dataset.X[:, :, :n_continuous].reshape(-1, n_continuous)
    X_onehot = test_dataset.X[:, :, n_continuous:].reshape(-1, n_ts_keys)
    
    X_continuous_scaled = scaler_X.transform(X_continuous)
    X_continuous_scaled = X_continuous_scaled.reshape(n_samples, seq_len, n_continuous)
    X_onehot_reshaped = X_onehot.reshape(n_samples, seq_len, n_ts_keys)
    
    test_dataset.X = np.concatenate([X_continuous_scaled, X_onehot_reshaped], axis=2)
    test_dataset.y = scaler_y.transform(test_dataset.y.reshape(-1, 1)).flatten()
    test_dataset.scaler_X = scaler_X
    test_dataset.scaler_y = scaler_y
    
    print(f"Test samples: {len(test_dataset):,}")
    
    # STEP 2: Initialize model with correct input size
    print("\n" + "="*60)
    print("STEP 2: Initializing multivariate LSTM model")
    print("="*60)
    
    INPUT_SIZE = train_dataset.X.shape[2]  # Value + year + month + one-hot
    
    model = LSTMForecaster(
        input_size=INPUT_SIZE,
        hidden_size=64,
        num_layers=2,
        dropout=0.2
    ).to(device)
    
    print(f"\nModel architecture:")
    print(f"  Input size: {INPUT_SIZE} features")
    print(f"    - Continuous: 3 (Value, year, month)")
    print(f"    - One-hot: {n_ts_keys} (ts_key encoding)")
    print(f"  Hidden size: 64")
    print(f"  Num layers: 2")
    print(f"  Output size: 1")
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # STEP 3: Training setup
    print("\n" + "="*60)
    print("STEP 3: Training setup")
    print("="*60)
    
    EPOCHS = 50
    BATCH_SIZE = 256
    LEARNING_RATE = 0.001
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    print(f"Epochs: {EPOCHS}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Batches per epoch: {len(train_loader)}")
    
    # STEP 4: Training loop
    print("\n" + "="*60)
    print("STEP 4: Training")
    print("="*60)
    
    
    def train_epoch(model, loader, criterion, optimizer, device):
        model.train()
        total_loss = 0
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
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
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    start_time = time.time()
    
    for epoch in range(EPOCHS):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = evaluate(model, test_loader, criterion, device)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Save both model state and full model
            torch.save(model.state_dict(), os.path.join(output_path, 'best_lstm_model.pth'))
            torch.save({
                'model_state_dict': model.state_dict(),
                'input_size': INPUT_SIZE,
                'hidden_size': 64,
                'num_layers': 2,
                'dropout': 0.2,
                'scaler_X': scaler_X,
                'scaler_y': scaler_y,
                'n_ts_keys': n_ts_keys,
                'ts_key_to_idx': ts_key_to_idx,
                'seq_length': SEQ_LENGTH
                
            }, os.path.join(output_path, 'best_lstm_model_complete.pth'))
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            elapsed = time.time() - start_time
            print(f"Epoch [{epoch+1:3d}/{EPOCHS}] | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Time: {elapsed:.1f}s")
    
    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"\n✓ Model saved to: best_lstm_model_complete.pth")
    
    # STEP 5: Evaluation metrics
    print("\n" + "="*60)
    print("STEP 5: Evaluation Metrics")
    print("="*60)
    
 
    def smape(y_true, y_pred):
        """Symmetric Mean Absolute Percentage Error"""
        denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
        diff = np.abs(y_true - y_pred) / denominator
        diff[denominator == 0] = 0.0
        return 100 * np.mean(diff)
    
    # Load best model
    model.load_state_dict(torch.load('best_lstm_model.pth'))
    model.eval()
    
    # Get predictions
    all_predictions = []
    all_actuals = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            predictions = model(X_batch)
            all_predictions.extend(predictions.cpu().numpy())
            all_actuals.extend(y_batch.numpy())
    
    all_predictions = np.array(all_predictions).flatten()
    all_actuals = np.array(all_actuals).flatten()
    
    # Inverse transform to original scale
    all_predictions_original = scaler_y.inverse_transform(all_predictions.reshape(-1, 1)).flatten()
    all_actuals_original = scaler_y.inverse_transform(all_actuals.reshape(-1, 1)).flatten()
    
    # Calculate metrics
    mse = mean_squared_error(all_actuals_original, all_predictions_original)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(all_actuals_original, all_predictions_original)
    r2 = r2_score(all_actuals_original, all_predictions_original)
    smape_score = smape(all_actuals_original, all_predictions_original)
    
    print("\nMetrics on Test Set (Original Scale):")
    print(f"  MSE:   {mse:.2f}")
    print(f"  RMSE:  {rmse:.2f}")
    print(f"  MAE:   {mae:.2f}")
    print(f"  R²:    {r2:.4f}")
    print(f"  SMAPE: {smape_score:.2f}%")
    
    # STEP 6: Visualization
    print("\n" + "="*60)
    print("STEP 6: Saving visualizations")
    print("="*60)
    
    import matplotlib.pyplot as plt
    
    # Plot 1: Training history
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    ax.plot(train_losses, label='Training Loss', linewidth=2, color='#2E86AB')
    ax.plot(val_losses, label='Validation Loss', linewidth=2, color='#A23B72')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss (MSE)', fontsize=12)
    ax.set_title('LSTM Training History - Multivariate Forecasting', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'lstm_training_history.png'), dpi=300, bbox_inches='tight')
    print("✓ Saved: lstm_training_history.png")
    plt.close()
    
    # Plot 2: Predictions vs Actuals
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Scatter plot
    axes[0].scatter(all_actuals_original, all_predictions_original, alpha=0.3, s=10, color='#2E86AB')
    axes[0].plot([all_actuals_original.min(), all_actuals_original.max()], 
                 [all_actuals_original.min(), all_actuals_original.max()], 
                 'r--', linewidth=2, label='Perfect Prediction')
    axes[0].set_xlabel('Actual Values', fontsize=12)
    axes[0].set_ylabel('Predicted Values', fontsize=12)
    axes[0].set_title(f'Predictions vs Actuals (R²={r2:.4f})', fontsize=13, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Residuals
    residuals = all_actuals_original - all_predictions_original
    axes[1].hist(residuals, bins=50, edgecolor='black', alpha=0.7, color='#A23B72')
    axes[1].axvline(x=0, color='r', linestyle='--', linewidth=2)
    axes[1].set_xlabel('Residuals (Actual - Predicted)', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_title(f'Residual Distribution (SMAPE={smape_score:.2f}%)', fontsize=13, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('lstm_predictions_analysis.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: lstm_predictions_analysis.png")
    plt.close()
    
    
    