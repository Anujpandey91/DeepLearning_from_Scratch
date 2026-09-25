"""Tests for ReLU and softmax activations."""

import pytest
import numpy as np
from ..implementations.activations import relu, relu_backward, softmax


def test_relu():
    x = np.array([[-2, 4], [4, -9]])
    expected = np.array([[0, 4], [4, 0]])

    result = relu(x)

    assert np.array_equal(result, expected)


def test_relu_zero():
    x = np.array([0])
    expected = np.array([0])

    result = relu(x)

    assert np.array_equal(result, expected)


def test_relu_3d():
    x = np.array([[[-1, 2], [3, -4]], [[5, -6], [-7, 8]]])

    expected = np.array([[[0, 2], [3, 0]], [[5, 0], [0, 8]]])

    result = relu(x)

    assert np.array_equal(result, expected)


def test_relu_backward():
    Z = np.array([[-2, 3], [4, -5]])

    dA = np.array([[10, 20], [30, 40]])

    expected = np.array([[0, 20], [30, 0]])

    result = relu_backward(dA, Z)

    assert np.array_equal(result, expected)


def test_softmax():
    Z = np.array([[2.0], [1.0], [0.0]])

    result = softmax(Z)

    expected = np.array([[0.66524096], [0.24472847], [0.09003057]])

    assert np.allclose(result, expected)


def test_softmax_output_shape():
    Z = np.random.randn(5, 10)

    result = softmax(Z)

    assert result.shape == Z.shape


def test_softmax_column_sums():
    Z = np.array([[2.0, 1.0, 0.5], [1.0, 3.0, 0.2], [0.0, 2.0, 4.0]])

    result = softmax(Z)

    column_sums = np.sum(result, axis=0)

    assert np.allclose(column_sums, np.ones(3))


def test_softmax_probability_range():
    Z = np.random.randn(4, 8)

    result = softmax(Z)

    assert np.all(result >= 0)
    assert np.all(result <= 1)


def test_softmax_larger_logit_higher_probability():
    Z = np.array([[3.0], [1.0], [0.0]])

    result = softmax(Z)

    assert result[0, 0] > result[1, 0]
    assert result[1, 0] > result[2, 0]


def test_softmax_numerical_stability():
    Z = np.array([[1000.0], [1001.0], [1002.0]])

    result = softmax(Z)

    assert np.all(np.isfinite(result))
    assert np.allclose(np.sum(result), 1.0)


def test_softmax_multiple_examples():
    Z = np.array([[2.0, 1.0], [1.0, 3.0], [0.0, 2.0]])

    result = softmax(Z)

    assert np.allclose(np.sum(result, axis=0), np.ones(2))

    assert result[0, 0] > result[1, 0]
    assert result[1, 1] > result[0, 1]


def test_softmax_invalid_dimension():
    Z = np.random.randn(3, 4, 2)

    with pytest.raises(ValueError):
        softmax(Z)

