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