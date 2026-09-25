import numpy as np


def flatten(x):
    """Flatten a feature tensor ``(H, W, C)`` into a vector ``(HWC,)``."""
    if x.ndim != 3:
        raise ValueError("Input must be a 3D array")

    return x.reshape(-1)


def dense_forward(X, W, b):
    """Compute a dense layer using the project column-wise convention.

    ``X`` has shape ``(n_features, m)``, ``W`` has shape
    ``(n_units, n_features)``, and ``b`` has shape ``(n_units, 1)``.
    The forward equation is ``Z = W X + b``; the bias broadcasts across
    the ``m`` examples.
    """

    if X.ndim != 2:
        raise ValueError("X must be a 2D array")

    if W.ndim != 2:
        raise ValueError("W must be a 2D array")

    if b.ndim != 2:
        raise ValueError("b must be a 2D array")

    if W.shape[1] != X.shape[0]:
        raise ValueError("Number of W columns must match number of X features")

    if W.shape[0] != b.shape[0]:
        raise ValueError("Number of neurons must match number of biases")

    if b.shape[1] != 1:
        raise ValueError("b must have shape (n_units, 1)")

    Z = np.dot(W, X) + b

    return Z

def dense_backward(dZ, X, W):
    """Compute gradients for a dense layer averaged over ``m`` examples.

    With ``X`` shaped ``(n_features, m)`` and ``dZ`` shaped
    ``(n_units, m)``, the equations are ``dW = (1/m) dZ X^T``,
    ``db = (1/m) sum(dZ)``, and ``dX = W^T dZ``.
    """
    if dZ.ndim != 2:
        raise ValueError("dZ must be a 2D array")

    if X.ndim != 2:
        raise ValueError("X must be a 2D array")

    if W.ndim != 2:
        raise ValueError("W must be a 2D array")

    if W.shape[1] != X.shape[0]:
        raise ValueError("Number of W columns must match number of X features")

    if dZ.shape[0] != W.shape[0]:
        raise ValueError("dZ and W must have the same number of units")

    if dZ.shape[1] != X.shape[1]:
        raise ValueError("dZ and X must have the same number of examples")

    _, m = X.shape

    dW = (1 / m) * np.dot(dZ, X.T)
    db = np.sum(dZ, axis=1, keepdims=True) / m
    dX = np.dot(W.T, dZ)

    return dW, db, dX


def flatten_backward(dflat, input_shape):
    """Restore a flattened gradient to its original ``(H, W, C)`` shape."""
    if dflat.ndim != 1:
        raise ValueError("dflat must be a 1D array")

    if len(input_shape) != 3:
        raise ValueError("input_shape must contain 3 dimensions")

    if dflat.size != np.prod(input_shape):
        raise ValueError("dflat size must match the number of elements in input_shape")

    dX = dflat.reshape(input_shape)

    return dX





