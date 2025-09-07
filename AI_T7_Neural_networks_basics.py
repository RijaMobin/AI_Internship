import numpy as np

# ------------------------------
# 1. Activation Functions
# ------------------------------

# Sigmoid: squashes values between 0 and 1 (good for probabilities)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of sigmoid (needed in backpropagation)
def sigmoid_derivative(x):
    # here x is already the output of sigmoid, so derivative = x * (1-x)
    return x * (1 - x)

# ReLU: outputs 0 if input is negative, else keeps the value
def relu(x):
    return np.maximum(0, x)

# Derivative of ReLU (used in backprop)
def relu_derivative(x):
    return np.where(x > 0, 1, 0)

# Softmax: converts values into probabilities (for multi-class problems)
def softmax(x):
    exp_x = np.exp(x - np.max(x))  # subtract max for numerical stability
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# ------------------------------
# 2. Training Data (XOR problem)
# ------------------------------
# XOR truth table:
# Input -> Output
# (0,0) -> 0
# (0,1) -> 1
# (1,0) -> 1
# (1,1) -> 0

X = np.array([[0,0],
              [0,1],
              [1,0],
              [1,1]])

y = np.array([[0],
              [1],
              [1],
              [0]])

# ------------------------------
# 3. Initialize Network Parameters
# ------------------------------
np.random.seed(42)  # for reproducibility

input_size = 2    # 2 inputs (XOR has 2 bits)
hidden_size = 2   # hidden layer with 2 neurons
output_size = 1   # output is single value (0 or 1)

# Random weights and biases
W1 = np.random.randn(input_size, hidden_size)   # weights input -> hidden
b1 = np.zeros((1, hidden_size))                 # bias for hidden layer
W2 = np.random.randn(hidden_size, output_size)  # weights hidden -> output
b2 = np.zeros((1, output_size))                 # bias for output layer

# ------------------------------
# 4. Training Settings
# ------------------------------
lr = 0.1        # learning rate (step size for gradient descent)
epochs = 10000  # number of training loops

# ------------------------------
# 5. Training Loop
# ------------------------------
for epoch in range(epochs):
    # -------- Forward Pass --------
    # Step 1: Input → Hidden
    z1 = np.dot(X, W1) + b1      # weighted sum
    a1 = sigmoid(z1)             # activation function (hidden layer)

    # Step 2: Hidden → Output
    z2 = np.dot(a1, W2) + b2     # weighted sum
    a2 = sigmoid(z2)             # activation function (output layer)

    # -------- Loss Calculation --------
    # Mean Squared Error (difference between true and predicted)
    loss = np.mean((y - a2)**2)

    # -------- Backward Pass (Backpropagation) --------
    # Error at output
    error_output = (y - a2) * sigmoid_derivative(a2)

    # Error at hidden layer (propagated backward)
    error_hidden = error_output.dot(W2.T) * sigmoid_derivative(a1)

    # -------- Gradient Descent (Update Weights) --------
    # Update weights for hidden → output
    W2 += lr * a1.T.dot(error_output)
    b2 += lr * np.sum(error_output, axis=0, keepdims=True)

    # Update weights for input → hidden
    W1 += lr * X.T.dot(error_hidden)
    b1 += lr * np.sum(error_hidden, axis=0, keepdims=True)

    # Print progress every 2000 epochs
    if epoch % 2000 == 0:
        print(f"Epoch {epoch}, Loss = {loss:.4f}")

# ------------------------------
# 6. Testing After Training
# ------------------------------
print("\nFinal Predictions after training:")
for xi in X:
    # Forward pass with learned weights
    hidden = sigmoid(np.dot(xi, W1) + b1)
    output = sigmoid(np.dot(hidden, W2) + b2)
    print(f"{xi} -> {output.round(3)}")
