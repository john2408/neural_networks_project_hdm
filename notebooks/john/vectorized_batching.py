"""
Comprehensive Benchmark: Naive vs Vectorized Time Series Batching

This script demonstrates why Nixtla's NeuralForecast models (NBEATS, NHITS) train 
10-50x faster than traditional approaches. The key insight: batch across BOTH 
time and series dimensions simultaneously.

Key Difference:
- Naive: Creates 40,000 samples (1000 series × 40 windows) → 1,250 forward passes
- Vectorized: Creates 40 time blocks (each with 1000 series) → 2.5 forward passes

Result: Same training, 500x fewer forward passes, 20-40x speedup in practice.
"""

import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ============================================================
# Configuration
# ============================================================
N_SERIES = 1000      # Number of time series
TS_LEN = 60          # Length of each time series
INPUT_SIZE = 20      # Lookback window size
HORIZON = 1          # Forecast horizon
EPOCHS = 3           # Training epochs
NAIVE_BATCH = 32     # Naive approach batch size
VECTORIZED_BATCH = 16  # Vectorized approach batch size (but processes N_SERIES × this)

# Device selection
DEVICE = "mps" if torch.backends.mps.is_available() else \
         "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 70)
print("VECTORIZED BATCHING BENCHMARK")
print("=" * 70)
print(f"\nConfiguration:")
print(f"  Time series: {N_SERIES:,}")
print(f"  Series length: {TS_LEN}")
print(f"  Lookback window: {INPUT_SIZE}")
print(f"  Forecast horizon: {HORIZON}")
print(f"  Epochs: {EPOCHS}")
print(f"  Device: {DEVICE.upper()}")
print(f"  Naive batch size: {NAIVE_BATCH}")
print(f"  Vectorized batch size: {VECTORIZED_BATCH} (effective: {VECTORIZED_BATCH * N_SERIES:,})")
print()

# Set seed for reproducibility
torch.manual_seed(42)

# Generate synthetic data: (N_SERIES, TS_LEN)
data = torch.randn(N_SERIES, TS_LEN)

# ============================================================
# Shared Model Architecture
# ============================================================
class MLP(nn.Module):
    """Simple MLP for time series forecasting."""
    def __init__(self, input_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)

# ============================================================
# Approach A: NAIVE (Per-Window-Per-Series)
# ============================================================
class NaiveTSDataset(Dataset):
    """
    Traditional approach: Create one sample per window per series.
    
    Dataset size = N_SERIES × N_WINDOWS
    Each sample: one window from one series
    Batch: random mixture of windows from different series
    
    Result: Small batches, many forward passes, poor GPU utilization
    """
    def __init__(self, data, input_size, horizon):
        print(f"Creating NAIVE dataset...")
        self.samples = []
        
        # Double loop: iterate over series, then windows
        for s in range(data.shape[0]):
            ts = data[s]
            for t in range(len(ts) - input_size - horizon + 1):
                x = ts[t:t+input_size]
                y = ts[t+input_size:t+input_size+horizon]
                self.samples.append((x, y))
        
        print(f"  Created {len(self.samples):,} individual samples")
        print(f"  Sample shape: x={self.samples[0][0].shape}, y={self.samples[0][1].shape}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def train_naive():
    """Train using naive per-window-per-series batching."""
    print("\n" + "=" * 70)
    print("APPROACH A: NAIVE (Traditional)")
    print("=" * 70)
    
    dataset = NaiveTSDataset(data, INPUT_SIZE, HORIZON)
    loader = DataLoader(dataset, batch_size=NAIVE_BATCH, shuffle=True)
    
    n_batches = len(loader)
    print(f"  Batches per epoch: {n_batches:,}")
    print(f"  Predictions per batch: {NAIVE_BATCH}")
    print(f"  Total forward passes: {n_batches * EPOCHS:,}")
    
    model = MLP(INPUT_SIZE).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    print(f"\nTraining for {EPOCHS} epochs...")
    start = time.perf_counter()
    
    batch_count = 0
    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        for x, y in loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            
            y_hat = model(x)
            loss = criterion(y_hat, y)
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            epoch_loss += loss.item()
            batch_count += 1
        
        avg_loss = epoch_loss / len(loader)
        print(f"  Epoch {epoch+1}/{EPOCHS} - Loss: {avg_loss:.6f}")
    
    end = time.perf_counter()
    training_time = end - start
    
    print(f"\n  Total batches processed: {batch_count:,}")
    print(f"  Training time: {training_time:.2f} seconds")
    print(f"  Time per epoch: {training_time/EPOCHS:.2f} seconds")
    print(f"  Time per batch: {training_time/batch_count*1000:.2f} ms")
    
    return training_time, batch_count

# ============================================================
# Approach B: VECTORIZED (Nixtla-style)
# ============================================================
class VectorizedTSDataset(Dataset):
    """
    Vectorized approach: Batch across series AND time simultaneously.
    
    Dataset size = N_WINDOWS (not N_SERIES × N_WINDOWS!)
    Each sample: ALL series at one time window
    Batch: multiple time windows, each containing ALL series
    
    Result: Massive batches, few forward passes, excellent GPU utilization
    """
    def __init__(self, data, input_size, horizon):
        print(f"Creating VECTORIZED dataset...")
        self.data = data
        self.input_size = input_size
        self.horizon = horizon
        self.n_series, self.ts_len = data.shape
        self.valid_t = self.ts_len - input_size - horizon + 1
        
        print(f"  Dataset length: {self.valid_t} (time windows)")
        print(f"  Each window contains: {self.n_series:,} series")
        print(f"  Total predictions per sample: {self.n_series:,}")

    def __len__(self):
        return self.valid_t

    def __getitem__(self, t):
        """
        Return all series at time window t.
        
        Returns:
            x: (n_series, input_size) - all series, one time window
            y: (n_series, horizon) - all series, forecast targets
        """
        x = self.data[:, t:t+self.input_size]
        y = self.data[:, t+self.input_size:t+self.input_size+self.horizon]
        return x, y


def train_vectorized():
    """Train using vectorized cross-series-cross-time batching."""
    print("\n" + "=" * 70)
    print("APPROACH B: VECTORIZED (Nixtla-style)")
    print("=" * 70)
    
    dataset = VectorizedTSDataset(data, INPUT_SIZE, HORIZON)
    loader = DataLoader(dataset, batch_size=VECTORIZED_BATCH, shuffle=True)
    
    n_batches = len(loader)
    effective_batch_size = VECTORIZED_BATCH * N_SERIES
    print(f"  Batches per epoch: {n_batches}")
    print(f"  Time windows per batch: {VECTORIZED_BATCH}")
    print(f"  Series per time window: {N_SERIES:,}")
    print(f"  Effective predictions per batch: {effective_batch_size:,}")
    print(f"  Total forward passes: {n_batches * EPOCHS}")
    
    model = MLP(INPUT_SIZE).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    print(f"\nTraining for {EPOCHS} epochs...")
    start = time.perf_counter()
    
    batch_count = 0
    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        for x, y in loader:
            # x: (batch_time, n_series, input_size)
            # y: (batch_time, n_series, horizon)
            
            # Reshape to treat all (time × series) as batch dimension
            batch_size_time = x.shape[0]
            x = x.reshape(-1, INPUT_SIZE).to(DEVICE)  # (batch_time × n_series, input_size)
            y = y.reshape(-1, 1).to(DEVICE)           # (batch_time × n_series, 1)
            
            y_hat = model(x)
            loss = criterion(y_hat, y)
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            epoch_loss += loss.item()
            batch_count += 1
        
        avg_loss = epoch_loss / len(loader)
        print(f"  Epoch {epoch+1}/{EPOCHS} - Loss: {avg_loss:.6f}")
    
    end = time.perf_counter()
    training_time = end - start
    
    print(f"\n  Total batches processed: {batch_count}")
    print(f"  Training time: {training_time:.2f} seconds")
    print(f"  Time per epoch: {training_time/EPOCHS:.2f} seconds")
    print(f"  Time per batch: {training_time/batch_count*1000:.2f} ms")
    
    return training_time, batch_count

# ============================================================
# Run Benchmark
# ============================================================
if __name__ == "__main__":
    # Train with naive approach
    naive_time, naive_batches = train_naive()
    
    # Train with vectorized approach
    vectorized_time, vectorized_batches = train_vectorized()
    
    # Compare results
    print("\n" + "=" * 70)
    print("BENCHMARK RESULTS")
    print("=" * 70)
    print(f"\n{'Metric':<30} {'Naive':<15} {'Vectorized':<15} {'Improvement':<15}")
    print("-" * 70)
    
    speedup = naive_time / vectorized_time
    batch_reduction = naive_batches / vectorized_batches
    
    print(f"{'Training time (seconds)':<30} {naive_time:<15.2f} {vectorized_time:<15.2f} {speedup:<15.1f}x")
    print(f"{'Total forward passes':<30} {naive_batches:<15,} {vectorized_batches:<15} {batch_reduction:<15.1f}x")
    print(f"{'Avg predictions per pass':<30} {NAIVE_BATCH:<15} {VECTORIZED_BATCH * N_SERIES:<15,} {(VECTORIZED_BATCH * N_SERIES) / NAIVE_BATCH:<15.1f}x")
    
    print(f"\n{'='*70}")
    print(f"SPEEDUP: {speedup:.1f}x faster with vectorized batching!")
    print(f"{'='*70}")
    
    print("\nKey Insight:")
    print("  The model, optimizer, and loss function are IDENTICAL.")
    print("  The only difference is HOW we batch the data.")
    print("  Vectorized batching → larger matrix operations → better GPU utilization")
    print("  → massive speedup with zero change to model architecture!")
    print()