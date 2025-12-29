# MLP Model (Multi-Layer Perceptron)

The MLP (Multi-Layer Perceptron) Forecaster is a feedforward neural network designed for multivariate time series forecasting. It flattens the entire input sequence into a single vector and processes it through fully connected layers, treating the sequence as a fixed-length feature vector without explicitly modeling temporal dependencies.

**Layer Breakdown:**

- **Input Flattening**: Reshapes (batch_size, seq_length, input_size) → (batch_size, seq_length × input_size)
- **Hidden Layers**: 3 stacked fully connected layers with ReLU activation
- **Hidden Size**: 512 units per layer
- **Dropout**: Applied after each hidden layer (0.2)
- **Output Layer**: Single fully connected layer producing 1-step forecast

**Advantages:**

- **Simplicity**: Easy to implement, debug, and understand
- **Fast Training**: No recurrent computations, fully parallelizable
- **Global Pattern Recognition**: Sees entire sequence at once, captures long-range interactions
- **No Gradient Vanishing**: No backpropagation through time (unlike RNN/LSTM)

**Limitations:**

- **No Explicit Temporal Modeling**: Doesn't naturally capture sequential dependencies
- **High Parameter Count**: Flattening creates large first layer (scales with seq_length)
- **Poor for Long Sequences**: Parameter explosion and loss of temporal structure (best for <15 timesteps)
- **Fixed Input Length**: Requires same sequence length for all inputs
