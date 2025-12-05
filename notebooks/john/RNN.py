import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

# --------------------------------------------------
# 1. Load and pivot data
# --------------------------------------------------

# df has columns: timestamp, timeseries_key, value
# Example:
# df = pd.read_csv("data.csv")

df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp")

# Pivot to shape: [time, 3000 series]
pivot_df = df.pivot(index="timestamp", columns="timeseries_key", values="value")

# Fill missing values
pivot_df = pivot_df.fillna(method="ffill").fillna(method="bfill")

data = pivot_df.values.astype(np.float32)

# --------------------------------------------------
# 2. Train-test split (80/20 by time)
# --------------------------------------------------

split = int(len(data) * 0.8)
train_data = data[:split]
test_data = data[split:]

# --------------------------------------------------
# 3. Sliding window dataset
# --------------------------------------------------

class TimeSeriesDataset(Dataset):
    def _init_(self, data, seq_len=24):
        self.data = data
        self.seq_len = seq_len

    def _len_(self):
        return len(self.data) - self.seq_len

    def _getitem_(self, idx):
        x = self.data[idx : idx + self.seq_len]
        y = self.data[idx + self.seq_len]   # next-step prediction
        return torch.from_numpy(x), torch.from_numpy(y)

seq_len = 24  # choose any window

train_ds = TimeSeriesDataset(train_data, seq_len)
test_ds = TimeSeriesDataset(test_data, seq_len)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
test_loader  = DataLoader(test_ds, batch_size=32)

# --------------------------------------------------
# 4. Simple RNN model
# --------------------------------------------------

input_size = data.shape[1]   # 3000 time series
hidden_size = 64
num_layers = 1
output_size = input_size     # predict next step for all features

class RNNModel(nn.Module):
    def _init_(self):
        super()._init_()
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        last_hidden = out[:, -1, :]   # last time step
        return self.fc(last_hidden)

model = RNNModel()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# --------------------------------------------------
# 5. Training loop
# --------------------------------------------------

epochs = 5

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for X, y in train_loader:
        optimizer.zero_grad()
        preds = model(X)
        loss = criterion(preds, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs} - Train Loss: {total_loss/len(train_loader):.4f}")

# --------------------------------------------------
# 6. Testing
# --------------------------------------------------

model.eval()
test_loss = 0
with torch.no_grad():
    for X, y in test_loader:
        preds = model(X)
        loss = criterion(preds, y)
        test_loss += loss.item()

print(f"Test Loss: {test_loss/len(test_loader):.4f}")