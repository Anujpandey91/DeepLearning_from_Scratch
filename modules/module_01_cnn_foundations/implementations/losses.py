import numpy as np


def categorical_cross_entropy(Y, A):
    """Compute mean categorical cross-entropy for column-wise examples.

    ``Y`` and ``A`` both have shape ``(n_classes, m)``. The objective is
    ``J = -(1/m) sum(Y * log(A))``. Predictions are clipped only to keep
    ``log`` finite at exact zero and one.
    """
    if Y.ndim != 2:
        raise ValueError("The Y must be 2D")
    if A.ndim != 2:
        raise ValueError("The A must be 2D")
    if A.shape != Y.shape:
        raise ValueError("The shape of both Y and A should be equal")

    _, m = Y.shape

    epsilon = 1e-15
    A_safe = np.clip(A, epsilon, 1 - epsilon)

    loss = np.sum(Y * np.log(A_safe))

    return -(1 / m) * loss


def categorical_cross_entropy_backward(Y, A):
    """Return the softmax-plus-cross-entropy gradient ``(A - Y) / m``.

    Both inputs and the returned gradient use shape ``(n_classes, m)``.
    """
    if Y.ndim != 2:
        raise ValueError("The Y must be 2D")
    if A.ndim != 2:
        raise ValueError("The A must be 2D")
    if A.shape != Y.shape:
        raise ValueError("The shape of both Y and A should be equal")   

    _, m = Y.shape
    
    return (1 / m) * (A - Y)