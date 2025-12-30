# TCN (Temporal Convolutional Network) Implementation

## Overview

The **Temporal Convolutional Network (TCN)** is a neural network architecture designed for sequence modeling tasks, particularly time series forecasting. Unlike recurrent architectures (RNN/LSTM/GRU), TCNs use convolutional layers to process sequences in parallel, making them faster to train while maintaining the ability to capture long-range dependencies.

## Architecture

The `TCNForecaster` consists of two main components:

1. **TemporalBlock**: Building block containing dilated causal convolutions with residual connections
2. **TCNForecaster**: The main forecasting model that stacks temporal blocks and adds an output layer

### Model Structure

```
Input (batch_size, seq_length, input_size)
    ↓
Transpose to (batch_size, input_size, seq_length)
    ↓
Temporal Block 1 (dilation=1)
    ↓
Temporal Block 2 (dilation=2)
    ↓
Temporal Block 3 (dilation=4)
    ↓
...
    ↓
Temporal Block N (dilation=2^(N-1))
    ↓
Extract last timestep (batch_size, hidden_size)
    ↓
Fully Connected Layer
    ↓
Output (batch_size, 1)
```

## Key Components

### 1. Temporal Block

Each `TemporalBlock` implements the core TCN building block with:

- **Two dilated causal convolution layers**
- **ReLU activations**
- **Dropout for regularization**
- **Residual connection** (skip connection from input to output)

#### Structure of a Temporal Block:

```
Input (batch_size, channels, seq_length)
    ↓
[Conv1D → Chomp → ReLU → Dropout]  ← First conv block
    ↓
[Conv1D → Chomp → ReLU → Dropout]  ← Second conv block
    ↓
Add residual connection (input or downsample(input))
    ↓
ReLU
    ↓
Output (batch_size, channels, seq_length)
```

#### Dilated Causal Convolution

- **Causal**: Only uses past and current information, no future data leakage
- **Dilated**: Gaps between kernel elements increase receptive field exponentially
- **Chomping**: Padding is added during convolution, then removed to maintain causality

**Dilation visualization:**
```
Dilation = 1:  [x][x][x]        (kernel size 3, covers 3 timesteps)
Dilation = 2:  [x]_[x]_[x]      (kernel size 3, covers 5 timesteps)
Dilation = 4:  [x]___[x]___[x]  (kernel size 3, covers 9 timesteps)
```

### 2. Exponential Dilation Growth

TCN uses exponentially growing dilation rates: `2^0, 2^1, 2^2, 2^3, ...` (i.e., 1, 2, 4, 8, ...)

**Receptive Field Calculation:**
- With kernel size `k=3` and `n` layers with dilations `[1, 2, 4, ..., 2^(n-1)]`
- Receptive field = `1 + 2 × (k-1) × (2^n - 1)`

**Example with 4 layers (k=3):**
- Layer 1 (dilation=1): receptive field = 3
- Layer 2 (dilation=2): receptive field = 7
- Layer 3 (dilation=4): receptive field = 15
- Layer 4 (dilation=8): receptive field = 31

This exponential growth allows TCN to capture very long-range dependencies efficiently.

### 3. Residual Connections

Each temporal block includes a **skip connection** that adds the input directly to the output:

```python
output = ReLU(conv_output + input)
```

If input and output dimensions differ, a 1×1 convolution (`downsample`) adjusts the input dimensions:

```python
output = ReLU(conv_output + downsample(input))
```

**Benefits:**
- Prevents vanishing gradients in deep networks
- Allows learning identity mappings
- Improves gradient flow during backpropagation

## Data Flow Through TCNForecaster

### Input Shape Transformation

Let's trace a sample through the network with these parameters:
- `batch_size = 32`
- `seq_length = 24` (24 timesteps)
- `input_size = 10` (10 features per timestep)
- `hidden_size = 64`
- `num_layers = 3`
- `kernel_size = 3`

#### Step 1: Input
```
Shape: (32, 24, 10)
Description: 32 samples, each with 24 timesteps and 10 features
```

#### Step 2: Transpose for Conv1D
```python
x = x.transpose(1, 2)
Shape: (32, 10, 24)
Description: Conv1D expects (batch, channels, length)
```

#### Step 3: Temporal Block 1 (dilation=1)
```
Input:  (32, 10, 24)
    ↓ Conv1D (10 → 64 channels, kernel=3, dilation=1, padding=2)
    ↓ Chomp (remove 2 timesteps from end)
    ↓ ReLU, Dropout
    ↓ Conv1D (64 → 64 channels, kernel=3, dilation=1, padding=2)
    ↓ Chomp (remove 2 timesteps from end)
    ↓ ReLU, Dropout
    ↓ Add residual: downsample(input) via 1×1 conv (10 → 64)
    ↓ ReLU
Output: (32, 64, 24)
```

#### Step 4: Temporal Block 2 (dilation=2)
```
Input:  (32, 64, 24)
    ↓ Conv1D (64 → 64 channels, kernel=3, dilation=2, padding=4)
    ↓ Chomp (remove 4 timesteps from end)
    ↓ ReLU, Dropout
    ↓ Conv1D (64 → 64 channels, kernel=3, dilation=2, padding=4)
    ↓ Chomp (remove 4 timesteps from end)
    ↓ ReLU, Dropout
    ↓ Add residual: identity (no downsample needed)
    ↓ ReLU
Output: (32, 64, 24)
```

#### Step 5: Temporal Block 3 (dilation=4)
```
Input:  (32, 64, 24)
    ↓ Conv1D (64 → 64 channels, kernel=3, dilation=4, padding=8)
    ↓ Chomp (remove 8 timesteps from end)
    ↓ ReLU, Dropout
    ↓ Conv1D (64 → 64 channels, kernel=3, dilation=4, padding=8)
    ↓ Chomp (remove 8 timesteps from end)
    ↓ ReLU, Dropout
    ↓ Add residual: identity
    ↓ ReLU
Output: (32, 64, 24)
```

#### Step 6: Extract Last Timestep
```python
out = out[:, :, -1]
Shape: (32, 64)
Description: Take the last timestep's representation
```

#### Step 7: Fully Connected Output
```python
out = self.fc(out)
Shape: (32, 1)
Description: Final prediction for each sample
```

## Advantages of TCN

### 1. **Parallelization**
- Unlike RNNs, TCN can process all timesteps simultaneously
- No sequential dependency in forward pass
- Faster training on modern GPUs

### 2. **Stable Gradients**
- Residual connections enable direct gradient flow
- No vanishing gradient problem like vanilla RNNs
- Deeper networks possible without gradient issues

### 3. **Flexible Receptive Field**
- Easy to adjust receptive field by changing:
  - Number of layers
  - Dilation factors
  - Kernel size
- Can capture both short and long-term dependencies

### 4. **Causality**
- Built-in causal structure prevents information leakage
- No risk of using future data for predictions
- Essential for real-world time series forecasting

### 5. **Lower Memory Footprint**
- No need to store hidden states across timesteps during training
- More memory-efficient than LSTM/GRU for long sequences

## Hyperparameters

### `input_size`
- Number of features per timestep
- Example: Value + year + month + one-hot encoding = total features

### `hidden_size`
- Number of filters/channels in convolutional layers
- Controls model capacity
- Default: 64
- Typical range: 32-256

### `num_layers`
- Number of temporal blocks to stack
- More layers = larger receptive field
- Default: 3
- Typical range: 2-8

### `kernel_size`
- Size of the convolutional kernel
- Larger kernels = faster receptive field growth
- Default: 3
- Typical range: 2-7

### `dropout`
- Dropout probability for regularization
- Applied after each convolution
- Default: 0.2
- Typical range: 0.1-0.5

## Usage Example

```python
import torch
from models import TCNForecaster

# Initialize model
model = TCNForecaster(
    input_size=10,      # 10 features per timestep
    hidden_size=64,     # 64 filters per conv layer
    num_layers=3,       # 3 temporal blocks
    kernel_size=3,      # Kernel size of 3
    dropout=0.2         # 20% dropout
)

# Sample input: 32 samples, 24 timesteps, 10 features
x = torch.randn(32, 24, 10)

# Forward pass
output = model(x)
print(output.shape)  # torch.Size([32, 1])
```

## Comparison with Other Models

| Feature | TCN | LSTM | GRU | Transformer |
|---------|-----|------|-----|-------------|
| **Parallelization** | ✅ Yes | ❌ No | ❌ No | ✅ Yes |
| **Long-term Memory** | ✅ Yes (via dilations) | ✅ Yes | ✅ Yes | ✅ Yes (via attention) |
| **Training Speed** | ⚡ Fast | 🐌 Slow | 🐌 Moderate | ⚡ Fast |
| **Memory Efficiency** | ✅ High | ❌ Low | ❌ Moderate | ❌ Low (for long sequences) |
| **Causality** | ✅ Built-in | ⚠️ Requires masking | ⚠️ Requires masking | ⚠️ Requires masking |
| **Receptive Field** | 📈 Exponential growth | ♾️ Unlimited | ♾️ Unlimited | ♾️ Unlimited |
| **Parameters** | Medium | High | Medium | Very High |

## When to Use TCN

### ✅ Good For:
- Time series forecasting with moderate to long sequences
- Real-time applications requiring fast inference
- Problems requiring explicit causality
- When training speed is important
- Datasets with clear temporal patterns at multiple scales

### ⚠️ Consider Alternatives When:
- Very short sequences (< 10 timesteps) → Use MLP
- Need bidirectional context → Use Transformer or BiLSTM
- Complex attention mechanisms needed → Use Transformer
- Standard benchmark tasks where LSTM is well-tuned

## Implementation Details

### Causal Padding (Chomping)

To maintain causality, we:
1. Add padding of size `(kernel_size - 1) × dilation` to the **left** (past)
2. Apply convolution
3. **Chomp** (remove) the same amount from the **right** (future)

```python
padding = (kernel_size - 1) * dilation
self.conv = nn.Conv1d(..., padding=padding, dilation=dilation)
self.chomp = nn.ConstantPad1d((0, -padding), 0)
```

This ensures the output at timestep `t` only depends on inputs up to time `t`, never on future timesteps.

### Weight Initialization

PyTorch's default initialization for Conv1d layers works well, but you can customize:

```python
def init_weights(m):
    if isinstance(m, nn.Conv1d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

model.apply(init_weights)
```

## References

- **Original Paper**: [An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling](https://arxiv.org/abs/1803.01271) (Bai et al., 2018)
- **Key Innovation**: Dilated causal convolutions with residual connections for sequence modeling
- **Code Inspiration**: PyTorch implementation following the original paper's architecture

## Summary

The TCN architecture provides an efficient alternative to recurrent networks for time series forecasting. By leveraging:
- **Dilated causal convolutions** for exponentially growing receptive fields
- **Residual connections** for stable gradient flow
- **Parallel processing** for fast training

TCN achieves competitive performance with RNN/LSTM/GRU while being significantly faster to train and easier to optimize.
