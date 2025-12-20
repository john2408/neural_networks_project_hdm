import torch.nn as nn
import torch
import math

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


class RNNForecaster(nn.Module):
    """
    Vanilla RNN model for MULTIVARIATE time series forecasting.
    Architecture: RNN -> Dropout -> RNN -> Dropout -> Fully Connected
    Takes multiple input features at each timestep.
    
    Simpler than LSTM - no cell state, only hidden state.
    Faster training but may struggle with long-term dependencies.
    """
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        """
        Args:
            input_size: Number of input features (Value + year + month + one-hot)
            hidden_size: RNN hidden dimension
            num_layers: Number of RNN layers
            dropout: Dropout rate
        """
        super(RNNForecaster, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_size = input_size
        
        # RNN layers (using Tanh activation by default)
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            nonlinearity='tanh'  # Can also use 'relu'
        )
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Fully connected output layer
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        # x shape: (batch_size, seq_length, input_size)
        
        # RNN forward pass
        # rnn_out: (batch_size, seq_length, hidden_size)
        # h_n: (num_layers, batch_size, hidden_size)
        rnn_out, h_n = self.rnn(x)
        
        # Take the output from the last time step
        last_output = rnn_out[:, -1, :]  # Shape: (batch_size, hidden_size)
        
        # Apply dropout
        out = self.dropout(last_output)
        
        # Fully connected layer
        out = self.fc(out)  # Shape: (batch_size, 1)
        
        return out


class GRUForecaster(nn.Module):
    """
    GRU model for MULTIVARIATE time series forecasting.
    Architecture: GRU -> Dropout -> GRU -> Dropout -> Fully Connected
    Takes multiple input features at each timestep.
    
    GRU is similar to LSTM but with fewer parameters (no cell state).
    Generally faster than LSTM while maintaining good performance.
    Uses reset and update gates instead of LSTM's input/forget/output gates.
    """
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        """
        Args:
            input_size: Number of input features (Value + year + month + one-hot)
            hidden_size: GRU hidden dimension
            num_layers: Number of GRU layers
            dropout: Dropout rate
        """
        super(GRUForecaster, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_size = input_size
        
        # GRU layers
        self.gru = nn.GRU(
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
        
        # GRU forward pass
        # gru_out: (batch_size, seq_length, hidden_size)
        # h_n: (num_layers, batch_size, hidden_size)
        gru_out, h_n = self.gru(x)
        
        # Take the output from the last time step
        last_output = gru_out[:, -1, :]  # Shape: (batch_size, hidden_size)
        
        # Apply dropout
        out = self.dropout(last_output)
        
        # Fully connected layer
        out = self.fc(out)  # Shape: (batch_size, 1)
        
        return out


class CNN1DForecaster(nn.Module):
    """
    1D CNN model for MULTIVARIATE time series forecasting.
    Architecture: 
        Conv1D blocks (Conv -> ReLU -> MaxPool) -> 
        Adaptive Average Pooling -> Flatten -> 
        Fully Connected -> Dropout -> Output
    
    CNNs can capture local patterns and temporal dependencies efficiently.
    Uses multiple kernel sizes to capture patterns at different scales.
    Processes sequences in parallel (unlike RNN/LSTM/GRU).
    """
    def __init__(self, input_size, hidden_size=64, num_layers=3, dropout=0.2):
        """
        Args:
            input_size: Number of input features (Value + year + month + one-hot)
            hidden_size: Number of filters in conv layers
            num_layers: Number of convolutional blocks (minimum 1)
            dropout: Dropout rate
        """
        super(CNN1DForecaster, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = max(1, num_layers)
        
        # Convolutional layers
        conv_layers = []
        
        # First conv block: input_size -> hidden_size
        conv_layers.extend([
            nn.Conv1d(in_channels=input_size, out_channels=hidden_size, 
                     kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=1, padding=1)
        ])
        
        # Additional conv blocks: hidden_size -> hidden_size
        for i in range(1, self.num_layers):
            conv_layers.extend([
                nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, 
                         kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2, stride=1, padding=1)
            ])
        
        self.conv_blocks = nn.Sequential(*conv_layers)
        
        # Adaptive pooling to fixed size output
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        # x shape: (batch_size, seq_length, input_size)
        
        # Conv1d expects (batch_size, channels, seq_length)
        # Transpose from (batch, seq, features) to (batch, features, seq)
        x = x.transpose(1, 2)  # (batch_size, input_size, seq_length)
        
        # Apply convolutional blocks
        x = self.conv_blocks(x)  # (batch_size, hidden_size, seq_length')
        
        # Adaptive pooling to reduce to (batch_size, hidden_size, 1)
        x = self.adaptive_pool(x)  # (batch_size, hidden_size, 1)
        
        # Flatten
        x = x.squeeze(-1)  # (batch_size, hidden_size)
        
        # Fully connected layers
        x = self.fc1(x)  # (batch_size, hidden_size)
        x = self.relu(x)
        x = self.dropout(x)
        out = self.fc2(x)  # (batch_size, 1)
        
        return out


class MLPForecaster(nn.Module):
    """
    MLP model for UNIVARIATE time series forecasting with one-hot encoding.
    Architecture: 
        Input Flattening -> MLP Layers (Linear -> ReLU -> Dropout) -> Output
    
    Takes input sequences with multiple features (Value + features + year + month + one-hot)
    and flattens them before processing through fully connected layers.
    
    Unlike MLPMultivariate:
    - Uses one-hot encoding to identify individual time series
    - Predicts one value at a time for a specific series
    - Processes entire sequence as flattened vector
    
    Good for:
    - Capturing non-linear patterns across the entire sequence
    - Faster training than RNN/LSTM for shorter sequences
    - Learning complex feature interactions
    """
    def __init__(self, input_size, seq_length, hidden_size=512, num_layers=3, dropout=0.2):
        """
        Args:
            input_size: Number of input features (Value + year + month + one-hot)
            seq_length: Sequence length (lookback window)
            hidden_size: Number of units in each MLP layer
            num_layers: Number of MLP layers
            dropout: Dropout rate
        """
        super(MLPForecaster, self).__init__()
        
        self.input_size = input_size
        self.seq_length = seq_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Calculate input dimension after flattening
        self.input_dim = seq_length * input_size
        
        # Build MLP layers
        layers = []
        
        # First layer
        layers.append(nn.Linear(self.input_dim, hidden_size))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        
        # Hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        
        self.mlp = nn.Sequential(*layers)
        
        # Output layer
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_length, input_size)
        
        Returns:
            predictions: (batch_size, 1) - single value prediction
        """
        batch_size = x.size(0)
        
        # Flatten entire sequence
        x_flat = x.reshape(batch_size, -1)  # (batch_size, seq_length * input_size)
        
        # Pass through MLP
        x = self.mlp(x_flat)  # (batch_size, hidden_size)
        
        # Output prediction
        out = self.fc(x)  # (batch_size, 1)
        
        return out


class MLPMultivariate(nn.Module):
    """
    MLP for Multivariate time series forecasting using FLATTENING approach.
    
    Instead of one-hot encoding, this model flattens all time series values
    into a single input vector, learning cross-series relationships directly.
    
    Architecture:
        Input Flattening -> MLP Layers (ReLU) -> Output Layer
    
    Input structure (flattened):
        [Value^1_t-L, ..., Value^1_t-1, Value^2_t-L, ..., Value^N_t-1,
         Feature^1_t-L, ..., Feature^N_t-1, Year, Month]
    
    This is more efficient than one-hot encoding for datasets with many time series.
    
    Key difference from other models:
    - Does NOT use one-hot encoding for time series identity
    - Flattens all series values across the sequence dimension
    - Learns relationships between different time series
    """
    def __init__(self, input_size, n_series, n_features_additional=0, 
                 num_layers=2, hidden_size=512, dropout=0.2):
        """
        Args:
            input_size: Sequence length (lookback window)
            n_series: Number of time series
            n_features_additional: Number of additional features per timestep
            num_layers: Number of MLP layers
            hidden_size: Number of units in each MLP layer
            dropout: Dropout rate
        """
        super(MLPMultivariate, self).__init__()
        
        self.input_size = input_size
        self.n_series = n_series
        self.n_features_additional = n_features_additional
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        
        # Calculate input dimension after flattening
        # For each timestep: n_series values + n_series * n_features_additional
        # Plus 2 temporal features (year, month) shared across all series
        self.input_dim = (
            n_series * input_size +  # All series values across sequence
            n_series * n_features_additional * input_size +  # All features across sequence
            2  # Temporal features (year, month) from last timestep
        )
        
        # Build MLP layers
        layers = []
        
        # First layer
        layers.append(nn.Linear(self.input_dim, hidden_size))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        
        # Hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        
        self.mlp = nn.Sequential(*layers)
        
        # Output layer: predict all n_series at once
        self.out = nn.Linear(hidden_size, n_series)
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_length, n_features)
               where n_features = n_series + n_series * n_features_additional + 2
               
               Structure per timestep:
               [value_series1, value_series2, ..., value_seriesN,
                feat1_series1, feat2_series1, ..., featK_series1,
                feat1_series2, ..., featK_seriesN,
                year, month]
        
        Returns:
            predictions: (batch_size, n_series) - one prediction per series
        """
        batch_size = x.size(0)
        
        # Extract temporal features from last timestep
        temporal_features = x[:, -1, -2:]  # (batch_size, 2) - [year, month]
        
        # Extract values and features (exclude temporal from each timestep)
        x_without_temporal = x[:, :, :-2]  # (batch_size, seq_length, n_series + n_series*n_features)
        
        # Flatten across sequence dimension
        x_flat = x_without_temporal.reshape(batch_size, -1)  # (batch_size, seq_length * (n_series + n_series*n_features))
        
        # Concatenate with temporal features
        x_input = torch.cat([x_flat, temporal_features], dim=1)  # (batch_size, input_dim)
        
        # Pass through MLP
        x = self.mlp(x_input)  # (batch_size, hidden_size)
        
        # Output predictions for all series
        predictions = self.out(x)  # (batch_size, n_series)
        
        # Return predictions (shape matches other models: (batch_size, 1) for single series prediction)
        # For multivariate: we need to extract the specific series prediction
        # This will be handled in the dataset/training loop
        return predictions


class LSTMForecasterMultivariate(nn.Module):
    """
    LSTM for Multivariate time series forecasting using FLATTENING approach.
    
    Similar to MLPMultivariate but uses LSTM to capture temporal dependencies.
    Processes flattened multivariate sequences and predicts all series simultaneously.
    
    Architecture:
        LSTM Layers -> Dropout -> Fully Connected -> Output (all series)
    
    Input structure (flattened per timestep):
        [value_series1, value_series2, ..., value_seriesN,
         feat1_series1, feat2_series1, ..., featK_seriesN,
         year, month]
    
    Key advantages:
    - Captures temporal dependencies across all time series
    - More parameter-efficient than one-hot encoding for many series
    - Learns cross-series relationships through shared LSTM layers
    """
    def __init__(self, input_size, n_series, n_features_additional=0,
                 hidden_size=128, num_layers=2, dropout=0.2):
        """
        Args:
            input_size: Sequence length (lookback window)
            n_series: Number of time series
            n_features_additional: Number of additional features per series per timestep
            hidden_size: LSTM hidden dimension
            num_layers: Number of LSTM layers
            dropout: Dropout rate
        """
        super(LSTMForecasterMultivariate, self).__init__()
        
        self.input_size = input_size
        self.n_series = n_series
        self.n_features_additional = n_features_additional
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Calculate feature dimension per timestep
        # Each timestep has: n_series values + n_series * n_features + 2 temporal features
        self.features_per_timestep = n_series * (1 + n_features_additional) + 2
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=self.features_per_timestep,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Output layer: predict all n_series at once
        self.fc = nn.Linear(hidden_size, n_series)
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_length, n_features)
               where n_features = n_series + n_series * n_features_additional + 2
               
               Structure per timestep:
               [value_series1, value_series2, ..., value_seriesN,
                feat1_series1, feat2_series1, ..., featK_series1,
                feat1_series2, ..., featK_seriesN,
                year, month]
        
        Returns:
            predictions: (batch_size, n_series) - one prediction per series
        """
        # x shape: (batch_size, seq_length, features_per_timestep)
        
        # LSTM forward pass
        lstm_out, (h_n, c_n) = self.lstm(x)
        # lstm_out: (batch_size, seq_length, hidden_size)
        
        # Take the output from the last time step
        last_output = lstm_out[:, -1, :]  # (batch_size, hidden_size)
        
        # Apply dropout
        out = self.dropout(last_output)
        
        # Predict all series
        predictions = self.fc(out)  # (batch_size, n_series)
        
        return predictions


class RNNForecasterMultivariate(nn.Module):
    """
    RNN for Multivariate time series forecasting using FLATTENING approach.
    
    Similar to LSTMForecasterMultivariate but uses vanilla RNN cells.
    Simpler architecture, faster training, but may struggle with long-term dependencies.
    
    Architecture:
        RNN Layers -> Dropout -> Fully Connected -> Output (all series)
    
    Input structure (flattened per timestep):
        [value_series1, value_series2, ..., value_seriesN,
         feat1_series1, feat2_series1, ..., featK_seriesN,
         year, month]
    """
    def __init__(self, input_size, n_series, n_features_additional=0,
                 hidden_size=128, num_layers=2, dropout=0.2):
        """
        Args:
            input_size: Sequence length (lookback window)
            n_series: Number of time series
            n_features_additional: Number of additional features per series per timestep
            hidden_size: RNN hidden dimension
            num_layers: Number of RNN layers
            dropout: Dropout rate
        """
        super(RNNForecasterMultivariate, self).__init__()
        
        self.input_size = input_size
        self.n_series = n_series
        self.n_features_additional = n_features_additional
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Calculate feature dimension per timestep
        # Each timestep has: n_series values + n_series * n_features + 2 temporal features
        self.features_per_timestep = n_series * (1 + n_features_additional) + 2
        
        # RNN layers
        self.rnn = nn.RNN(
            input_size=self.features_per_timestep,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            nonlinearity='tanh'
        )
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Output layer: predict all n_series at once
        self.fc = nn.Linear(hidden_size, n_series)
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_length, n_features)
               where n_features = n_series + n_series * n_features_additional + 2
               
               Structure per timestep:
               [value_series1, value_series2, ..., value_seriesN,
                feat1_series1, feat2_series1, ..., featK_series1,
                feat1_series2, ..., featK_seriesN,
                year, month]
        
        Returns:
            predictions: (batch_size, n_series) - one prediction per series
        """
        # x shape: (batch_size, seq_length, features_per_timestep)
        
        # RNN forward pass
        rnn_out, h_n = self.rnn(x)
        # rnn_out: (batch_size, seq_length, hidden_size)
        
        # Take the output from the last time step
        last_output = rnn_out[:, -1, :]  # (batch_size, hidden_size)
        
        # Apply dropout
        out = self.dropout(last_output)
        
        # Predict all series
        predictions = self.fc(out)  # (batch_size, n_series)
        
        return predictions


class GRUForecasterMultivariate(nn.Module):
    """
    GRU for Multivariate time series forecasting using FLATTENING approach.
    
    Similar to LSTMForecasterMultivariate but uses GRU cells.
    Fewer parameters than LSTM while maintaining good performance.
    
    Architecture:
        GRU Layers -> Dropout -> Fully Connected -> Output (all series)
    
    Input structure (flattened per timestep):
        [value_series1, value_series2, ..., value_seriesN,
         feat1_series1, feat2_series1, ..., featK_seriesN,
         year, month]
    """
    def __init__(self, input_size, n_series, n_features_additional=0,
                 hidden_size=128, num_layers=2, dropout=0.2):
        """
        Args:
            input_size: Sequence length (lookback window)
            n_series: Number of time series
            n_features_additional: Number of additional features per series per timestep
            hidden_size: GRU hidden dimension
            num_layers: Number of GRU layers
            dropout: Dropout rate
        """
        super(GRUForecasterMultivariate, self).__init__()
        
        self.input_size = input_size
        self.n_series = n_series
        self.n_features_additional = n_features_additional
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Calculate feature dimension per timestep
        # Each timestep has: n_series values + n_series * n_features + 2 temporal features
        self.features_per_timestep = n_series * (1 + n_features_additional) + 2
        
        # GRU layers
        self.gru = nn.GRU(
            input_size=self.features_per_timestep,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Output layer: predict all n_series at once
        self.fc = nn.Linear(hidden_size, n_series)
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_length, n_features)
               where n_features = n_series + n_series * n_features_additional + 2
               
               Structure per timestep:
               [value_series1, value_series2, ..., value_seriesN,
                feat1_series1, feat2_series1, ..., featK_series1,
                feat1_series2, ..., featK_seriesN,
                year, month]
        
        Returns:
            predictions: (batch_size, n_series) - one prediction per series
        """
        # x shape: (batch_size, seq_length, features_per_timestep)
        
        # GRU forward pass
        gru_out, h_n = self.gru(x)
        # gru_out: (batch_size, seq_length, hidden_size)
        
        # Take the output from the last time step
        last_output = gru_out[:, -1, :]  # (batch_size, hidden_size)
        
        # Apply dropout
        out = self.dropout(last_output)
        
        # Predict all series
        predictions = self.fc(out)  # (batch_size, n_series)
        
        return predictions


class CNN1DForecasterMultivariate(nn.Module):
    """
    1D CNN for Multivariate time series forecasting using FLATTENING approach.
    
    Applies 1D convolutions across the temporal dimension to capture patterns.
    Processes all time series simultaneously through shared conv layers.
    
    Architecture:
        Conv1D blocks (Conv -> ReLU -> MaxPool) -> 
        Adaptive Average Pooling -> Flatten -> 
        Fully Connected -> Dropout -> Output (all series)
    
    Input structure (flattened per timestep):
        [value_series1, value_series2, ..., value_seriesN,
         feat1_series1, feat2_series1, ..., featK_seriesN,
         year, month]
    """
    def __init__(self, input_size, n_series, n_features_additional=0,
                 hidden_size=64, num_layers=3, dropout=0.2):
        """
        Args:
            input_size: Sequence length (lookback window)
            n_series: Number of time series
            n_features_additional: Number of additional features per series per timestep
            hidden_size: Number of filters in conv layers
            num_layers: Number of convolutional blocks
            dropout: Dropout rate
        """
        super(CNN1DForecasterMultivariate, self).__init__()
        
        self.input_size = input_size
        self.n_series = n_series
        self.n_features_additional = n_features_additional
        self.hidden_size = hidden_size
        self.num_layers = max(1, num_layers)
        
        # Calculate feature dimension per timestep
        # Each timestep has: n_series values + n_series * n_features + 2 temporal features
        self.features_per_timestep = n_series * (1 + n_features_additional) + 2
        
        # Convolutional layers
        conv_layers = []
        
        # First conv block: features_per_timestep -> hidden_size
        conv_layers.extend([
            nn.Conv1d(in_channels=self.features_per_timestep, out_channels=hidden_size, 
                     kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=1, padding=1)
        ])
        
        # Additional conv blocks: hidden_size -> hidden_size
        for i in range(1, self.num_layers):
            conv_layers.extend([
                nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, 
                         kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2, stride=1, padding=1)
            ])
        
        self.conv_blocks = nn.Sequential(*conv_layers)
        
        # Adaptive pooling to fixed size output
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
        # Output layer: predict all n_series at once
        self.fc2 = nn.Linear(hidden_size, n_series)
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_length, n_features)
               where n_features = n_series + n_series * n_features_additional + 2
               
               Structure per timestep:
               [value_series1, value_series2, ..., value_seriesN,
                feat1_series1, feat2_series1, ..., featK_series1,
                feat1_series2, ..., featK_seriesN,
                year, month]
        
        Returns:
            predictions: (batch_size, n_series) - one prediction per series
        """
        # x shape: (batch_size, seq_length, features_per_timestep)
        
        # Conv1d expects (batch_size, channels, seq_length)
        # Transpose from (batch, seq, features) to (batch, features, seq)
        x = x.transpose(1, 2)  # (batch_size, features_per_timestep, seq_length)
        
        # Apply convolutional blocks
        x = self.conv_blocks(x)  # (batch_size, hidden_size, seq_length')
        
        # Adaptive pooling to reduce to (batch_size, hidden_size, 1)
        x = self.adaptive_pool(x)  # (batch_size, hidden_size, 1)
        
        # Flatten
        x = x.squeeze(-1)  # (batch_size, hidden_size)
        
        # Fully connected layers
        x = self.fc1(x)  # (batch_size, hidden_size)
        x = self.relu(x)
        x = self.dropout(x)
        
        # Output predictions for all series
        predictions = self.fc2(x)  # (batch_size, n_series)
        
        return predictions


class TransformerForecasterMultivariate(nn.Module):
    """
    Transformer for Multivariate time series forecasting using FLATTENING approach.
    
    Uses self-attention mechanism to capture dependencies across time and features.
    Processes all time series simultaneously through shared transformer layers.
    
    Architecture:
        Input Projection -> Positional Encoding -> 
        Transformer Encoder -> Global Average Pooling -> 
        Dropout -> Fully Connected -> Output (all series)
    
    Input structure (flattened per timestep):
        [value_series1, value_series2, ..., value_seriesN,
         feat1_series1, feat2_series1, ..., featK_seriesN,
         year, month]
    """
    def __init__(self, input_size, n_series, n_features_additional=0,
                 d_model=64, nhead=4, num_layers=2, dim_feedforward=256, dropout=0.2):
        """
        Args:
            input_size: Sequence length (lookback window)
            n_series: Number of time series
            n_features_additional: Number of additional features per series per timestep
            d_model: Dimension of the model (must be divisible by nhead)
            nhead: Number of attention heads
            num_layers: Number of transformer encoder layers
            dim_feedforward: Dimension of feedforward network
            dropout: Dropout rate
        """
        super(TransformerForecasterMultivariate, self).__init__()
        
        self.input_size = input_size
        self.n_series = n_series
        self.n_features_additional = n_features_additional
        self.d_model = d_model
        
        # Calculate feature dimension per timestep
        # Each timestep has: n_series values + n_series * n_features + 2 temporal features
        self.features_per_timestep = n_series * (1 + n_features_additional) + 2
        
        # Input projection: map features_per_timestep to d_model
        self.input_projection = nn.Linear(self.features_per_timestep, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Output layer: predict all n_series at once
        self.fc = nn.Linear(d_model, n_series)
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_length, n_features)
               where n_features = n_series + n_series * n_features_additional + 2
               
               Structure per timestep:
               [value_series1, value_series2, ..., value_seriesN,
                feat1_series1, feat2_series1, ..., featK_series1,
                feat1_series2, ..., featK_seriesN,
                year, month]
        
        Returns:
            predictions: (batch_size, n_series) - one prediction per series
        """
        # x shape: (batch_size, seq_length, features_per_timestep)
        
        # Project input to d_model dimensions
        x = self.input_projection(x)  # (batch_size, seq_length, d_model)
        
        # Add positional encoding
        x = self.pos_encoder(x)  # (batch_size, seq_length, d_model)
        
        # Pass through transformer encoder
        transformer_out = self.transformer_encoder(x)  # (batch_size, seq_length, d_model)
        
        # Global average pooling over sequence dimension
        pooled = transformer_out.mean(dim=1)  # (batch_size, d_model)
        
        # Apply dropout
        out = self.dropout(pooled)
        
        # Predict all series
        predictions = self.fc(out)  # (batch_size, n_series)
        
        return predictions


class PositionalEncoding(nn.Module):
    """
    Positional encoding for Transformer model.
    Adds information about the position of tokens in the sequence.
    """
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # x shape: (batch_size, seq_length, d_model)
        return x + self.pe[:, :x.size(1), :]


class TransformerForecaster(nn.Module):
    """
    Transformer model for MULTIVARIATE time series forecasting.
    Architecture: 
        Input Projection -> Positional Encoding -> 
        Transformer Encoder -> Global Average Pooling -> 
        Dropout -> Fully Connected
    
    Uses self-attention mechanism to capture dependencies.
    Can process entire sequence in parallel (unlike RNN/LSTM).
    """
    def __init__(self, input_size, d_model=64, nhead=4, num_layers=2, 
                 dim_feedforward=256, dropout=0.2):
        """
        Args:
            input_size: Number of input features (Value + year + month + one-hot)
            d_model: Dimension of the model (must be divisible by nhead)
            nhead: Number of attention heads
            num_layers: Number of transformer encoder layers
            dim_feedforward: Dimension of feedforward network
            dropout: Dropout rate
        """
        super(TransformerForecaster, self).__init__()
        
        self.input_size = input_size
        self.d_model = d_model
        
        # Input projection: map input_size to d_model
        self.input_projection = nn.Linear(input_size, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True  # Important: batch_first=True for (batch, seq, feature) format
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Output layer
        self.fc = nn.Linear(d_model, 1)
    
    def forward(self, x):
        # x shape: (batch_size, seq_length, input_size)
        
        # Project input to d_model dimensions
        x = self.input_projection(x)  # (batch_size, seq_length, d_model)
        
        # Add positional encoding
        x = self.pos_encoder(x)  # (batch_size, seq_length, d_model)
        
        # Pass through transformer encoder
        transformer_out = self.transformer_encoder(x)  # (batch_size, seq_length, d_model)
        
        # Global average pooling over sequence dimension
        # Alternative: use last token or first token (like BERT's [CLS])
        pooled = transformer_out.mean(dim=1)  # (batch_size, d_model)
        
        # Apply dropout
        out = self.dropout(pooled)
        
        # Fully connected layer
        out = self.fc(out)  # (batch_size, 1)
        
        return out


class TransformerForecasterCLS(nn.Module):
    """
    Alternative Transformer implementation using CLS token approach.
    Similar to BERT - adds a learnable [CLS] token and uses its output.
    """
    def __init__(self, input_size, d_model=64, nhead=4, num_layers=2, 
                 dim_feedforward=256, dropout=0.2):
        """
        Args:
            input_size: Number of input features
            d_model: Dimension of the model (must be divisible by nhead)
            nhead: Number of attention heads
            num_layers: Number of transformer encoder layers
            dim_feedforward: Dimension of feedforward network
            dropout: Dropout rate
        """
        super(TransformerForecasterCLS, self).__init__()
        
        self.input_size = input_size
        self.d_model = d_model
        
        # Input projection
        self.input_projection = nn.Linear(input_size, d_model)
        
        # Learnable CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Output layer
        self.fc = nn.Linear(d_model, 1)
    
    def forward(self, x):
        # x shape: (batch_size, seq_length, input_size)
        batch_size = x.size(0)
        
        # Project input to d_model dimensions
        x = self.input_projection(x)  # (batch_size, seq_length, d_model)
        
        # Expand CLS token for batch
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (batch_size, 1, d_model)
        
        # Concatenate CLS token to sequence
        x = torch.cat([cls_tokens, x], dim=1)  # (batch_size, seq_length+1, d_model)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Pass through transformer encoder
        transformer_out = self.transformer_encoder(x)  # (batch_size, seq_length+1, d_model)
        
        # Take CLS token output
        cls_output = transformer_out[:, 0, :]  # (batch_size, d_model)
        
        # Apply dropout
        out = self.dropout(cls_output)
        
        # Fully connected layer
        out = self.fc(out)  # (batch_size, 1)
        
        return out