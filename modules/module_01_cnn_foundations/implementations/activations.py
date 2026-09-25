import numpy as np


def relu(X):
    """Apply the element-wise ReLU activation, ``max(0, X)``."""
    return np.maximum(0, X)


def relu_backward(dA, Z):
    """Propagate a gradient through ReLU using the pre-activation ``Z``."""
    return dA * (Z > 0)


def softmax(Z):
    """
    Compute numerically stable softmax probabilities.

    Parameters
    ----------
    Z : np.ndarray
        Logits with shape ``(n_classes, m)``. The project convention is
        classes/features on rows and examples on columns.

    Returns
    -------
    np.ndarray
        Softmax probabilities with the same shape as Z.
    """
    
    if Z.ndim != 2:
        raise ValueError("Z must be a 2D array")

    # Subtracting the largest logit preserves probabilities while avoiding
    # overflow in exp for large input values.
    Z_shifted = Z - np.max(Z, axis=0, keepdims=True)

    # Exponentiate
    exp_Z = np.exp(Z_shifted)

    # Normalize across classes
    probabilities = exp_Z / np.sum(exp_Z, axis=0, keepdims=True)

    return probabilities


