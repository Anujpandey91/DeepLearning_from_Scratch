import numpy as np

from ..implementations.convolution import convolve_multifilter

from ..implementations.activations import (
    relu,
    relu_backward,
    softmax,
)

from ..implementations.pooling import (
    max_pool_multichannel,
    max_pool_multichannel_backward,
)

from ..implementations.layers import (
    flatten,
    flatten_backward,
    dense_forward,
    dense_backward,
)

from ..implementations.losses import (
    categorical_cross_entropy,
    categorical_cross_entropy_backward,
)

# ============================================================
# Forward Pass
# ============================================================

image = np.arange(1, 49).reshape(4, 4, 3).astype(float)

filter_0 = np.ones((3, 3, 3))

filter_1 = -np.ones((3, 3, 3))

filters = np.stack([filter_0, filter_1], axis=-1)


# Convolution
conv_output = convolve_multifilter(image, filters, stride=1, padding=0)


# ReLU
relu_output = relu(conv_output)


# Max Pooling
pool_output = max_pool_multichannel(relu_output, pool_size=2, stride=2, padding=0)


# Flatten
flat_output = flatten(pool_output)

X_dense = flat_output.reshape(-1, 1)


# Dense layer
W = np.array([[0.5, 0.2], [0.1, -0.3], [-0.2, 0.4]])

b = np.zeros((3, 1))

logits = dense_forward(X_dense, W, b)


# Softmax
probabilities = softmax(logits)


# True label
Y = np.array([[0], [1], [0]])


# Loss
loss = categorical_cross_entropy(Y, probabilities)


# ============================================================
# Backward Pass
# ============================================================

# CCE + Softmax backward
dZ = categorical_cross_entropy_backward(Y, probabilities)


# Dense backward
dW, db, dX = dense_backward(dZ, X_dense, W)


# Flatten backward
dflat = dX.reshape(-1)

dPool = flatten_backward(dflat, pool_output.shape)


# Max Pooling backward
dReLU = max_pool_multichannel_backward(
    dPool, relu_output, pool_size=2, stride=2, padding=0
)


# ReLU backward
dConv = relu_backward(dReLU, conv_output)


# ============================================================
# Print Forward Values
# ============================================================

print("\n========== FORWARD PASS ==========")

print("\nInput shape:")
print(image.shape)

print("\nConv output:")
print(conv_output)

print("\nReLU output:")
print(relu_output)

print("\nPool output:")
print(pool_output)

print("\nFlatten output:")
print(flat_output)

print("\nLogits:")
print(logits)

print("\nProbabilities:")
print(probabilities)

print("\nProbability sum:")
print(np.sum(probabilities))

print("\nLoss:")
print(loss)


# ============================================================
# Print Backward Values
# ============================================================

print("\n========== BACKWARD PASS ==========")

print("\ndZ:")
print(dZ)

print("\ndW:")
print(dW)

print("\ndb:")
print(db)

print("\ndX:")
print(dX)

print("\ndflat:")
print(dflat)

print("\ndPool:")
print(dPool)

print("\ndReLU:")
print(dReLU)

print("\ndConv:")
print(dConv)
