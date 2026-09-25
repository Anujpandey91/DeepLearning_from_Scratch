"""Tests for categorical cross-entropy forward and backward operations."""

import numpy as np
import pytest

from ..implementations.losses import categorical_cross_entropy, categorical_cross_entropy_backward


def test_categorical_cross_entropy_basic_single_example():
    Y = np.array([[0], [1], [0]])

    A = np.array([[0.1], [0.7], [0.2]])

    result = categorical_cross_entropy(Y, A)

    expected = -np.log(0.7)

    assert np.isclose(result, expected)


def test_categorical_cross_entropy_basic_multiple_examples():
    Y = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    A = np.array([[0.7, 0.2, 0.1], [0.2, 0.6, 0.2], [0.1, 0.2, 0.7]])

    result = categorical_cross_entropy(Y, A)

    expected = (-np.log(0.7) - np.log(0.6) - np.log(0.7)) / 3

    assert np.isclose(result, expected)


def test_categorical_cross_entropy_basic_perfect_predictions():
    Y = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    A = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    result = categorical_cross_entropy(Y, A)

    assert np.isfinite(result)
    assert result < 1e-10


def test_categorical_cross_entropy_basic_bad_predictions():
    Y = np.array([[1], [0], [0]])

    A = np.array([[0.01], [0.49], [0.50]])

    result = categorical_cross_entropy(Y, A)

    expected = -np.log(0.01)

    assert np.isclose(result, expected)


def test_categorical_cross_entropy_basic_zero_probability():
    Y = np.array([[1], [0], [0]])

    A = np.array([[0.0], [0.5], [0.5]])

    result = categorical_cross_entropy(Y, A)

    assert np.isfinite(result)


def test_categorical_cross_entropy_basic_output_is_scalar():
    Y = np.array([[1, 0], [0, 1]])

    A = np.array([[0.8, 0.2], [0.2, 0.8]])

    result = categorical_cross_entropy(Y, A)

    assert np.isscalar(result)


def test_categorical_cross_entropy_basic_invalid_Y_dimension():
    Y = np.array([1, 0, 0])

    A = np.array([[0.7], [0.2], [0.1]])

    with pytest.raises(ValueError):
        categorical_cross_entropy(Y, A)


def test_categorical_cross_entropy_basic_invalid_A_dimension():
    Y = np.array([[1], [0], [0]])

    A = np.array([0.7, 0.2, 0.1])

    with pytest.raises(ValueError):
        categorical_cross_entropy(Y, A)


def test_categorical_cross_entropy_basic_shape_mismatch():
    Y = np.array([[1, 0], [0, 1], [0, 0]])

    A = np.array([[0.7, 0.2], [0.2, 0.8]])

    with pytest.raises(ValueError):
        categorical_cross_entropy(Y, A)

# ============================================================
# Categorical Cross-Entropy Forward
# ============================================================


def test_categorical_cross_entropy_single_example():
    Y = np.array([[0], [1], [0]])

    A = np.array([[0.1], [0.7], [0.2]])

    result = categorical_cross_entropy(Y, A)

    expected = -np.log(0.7)

    assert np.isclose(result, expected)


def test_categorical_cross_entropy_multiple_examples():
    Y = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    A = np.array([[0.7, 0.2, 0.1], [0.2, 0.6, 0.2], [0.1, 0.2, 0.7]])

    result = categorical_cross_entropy(Y, A)

    expected = (-np.log(0.7) - np.log(0.6) - np.log(0.7)) / 3

    assert np.isclose(result, expected)


def test_categorical_cross_entropy_perfect_predictions():
    Y = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    A = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    result = categorical_cross_entropy(Y, A)

    assert np.isfinite(result)
    assert result < 1e-10


def test_categorical_cross_entropy_bad_prediction():
    Y = np.array([[1], [0], [0]])

    A = np.array([[0.01], [0.49], [0.50]])

    result = categorical_cross_entropy(Y, A)

    expected = -np.log(0.01)

    assert np.isclose(result, expected)


def test_categorical_cross_entropy_zero_probability():
    Y = np.array([[1], [0], [0]])

    A = np.array([[0.0], [0.5], [0.5]])

    result = categorical_cross_entropy(Y, A)

    assert np.isfinite(result)


def test_categorical_cross_entropy_output_is_scalar():
    Y = np.array([[1, 0], [0, 1]])

    A = np.array([[0.8, 0.2], [0.2, 0.8]])

    result = categorical_cross_entropy(Y, A)

    assert np.isscalar(result)


def test_categorical_cross_entropy_invalid_Y_dimension():
    Y = np.array([1, 0, 0])

    A = np.array([[0.7], [0.2], [0.1]])

    with pytest.raises(ValueError):
        categorical_cross_entropy(Y, A)


def test_categorical_cross_entropy_invalid_A_dimension():
    Y = np.array([[1], [0], [0]])

    A = np.array([0.7, 0.2, 0.1])

    with pytest.raises(ValueError):
        categorical_cross_entropy(Y, A)


def test_categorical_cross_entropy_shape_mismatch():
    Y = np.array([[1, 0], [0, 1], [0, 0]])

    A = np.array([[0.7, 0.2], [0.2, 0.8]])

    with pytest.raises(ValueError):
        categorical_cross_entropy(Y, A)


# ============================================================
# Categorical Cross-Entropy Backward
# ============================================================


def test_categorical_cross_entropy_backward_single_example():
    Y = np.array([[0], [1], [0]])

    A = np.array([[0.2], [0.7], [0.1]])

    result = categorical_cross_entropy_backward(Y, A)

    expected = np.array([[0.2], [-0.3], [0.1]])

    assert np.allclose(result, expected)


def test_categorical_cross_entropy_backward_multiple_examples():
    Y = np.array([[1, 0], [0, 1], [0, 0]])

    A = np.array([[0.7, 0.2], [0.2, 0.6], [0.1, 0.2]])

    result = categorical_cross_entropy_backward(Y, A)

    expected = np.array([[-0.15, 0.10], [0.10, -0.20], [0.05, 0.10]])

    assert np.allclose(result, expected)


def test_categorical_cross_entropy_backward_batch_scaling():
    Y = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1]])

    A = np.array([[0.8, 0.1, 0.2, 0.1], [0.1, 0.7, 0.3, 0.2], [0.1, 0.2, 0.5, 0.7]])

    result = categorical_cross_entropy_backward(Y, A)

    expected = (A - Y) / 4

    assert np.allclose(result, expected)


def test_categorical_cross_entropy_backward_output_shape():
    Y = np.zeros((5, 10))
    A = np.ones((5, 10)) / 5

    result = categorical_cross_entropy_backward(Y, A)

    assert result.shape == Y.shape


def test_categorical_cross_entropy_backward_invalid_Y_dimension():
    Y = np.array([1, 0, 0])

    A = np.array([[0.7], [0.2], [0.1]])

    with pytest.raises(ValueError):
        categorical_cross_entropy_backward(Y, A)


def test_categorical_cross_entropy_backward_invalid_A_dimension():
    Y = np.array([[1], [0], [0]])

    A = np.array([0.7, 0.2, 0.1])

    with pytest.raises(ValueError):
        categorical_cross_entropy_backward(Y, A)


def test_categorical_cross_entropy_backward_shape_mismatch():
    Y = np.array([[1, 0], [0, 1], [0, 0]])

    A = np.array([[0.7, 0.2], [0.2, 0.8]])

    with pytest.raises(ValueError):
        categorical_cross_entropy_backward(Y, A)
