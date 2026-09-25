"""Tests for flattening and dense-layer forward behavior."""

import numpy as np
import pytest

from ..implementations.layers import flatten, dense_forward


def test_flatten_values():
    x = np.array([[[1, 10], [2, 20]], [[3, 30], [4, 40]]])

    result = flatten(x)

    expected = np.array([1, 10, 2, 20, 3, 30, 4, 40])

    assert np.array_equal(result, expected)


def test_flatten_shape():
    x = np.zeros((4, 4, 3))

    result = flatten(x)

    assert result.shape == (48,)


def test_flatten_different_shape():
    x = np.zeros((2, 3, 4))

    result = flatten(x)

    assert result.shape == (24,)


def test_flatten_can_be_restored():
    x = np.array([[[1, 10], [2, 20]], [[3, 30], [4, 40]]])

    flattened = flatten(x)
    restored = flattened.reshape(x.shape)

    assert np.array_equal(restored, x)


@pytest.mark.parametrize(
    "shape",
    [
        (4, 4),
        (4, 4, 3, 2),
        (10,),
    ],
)
def test_flatten_invalid_dimensions(shape):
    x = np.zeros(shape)

    with pytest.raises(ValueError):
        flatten(x)


def test_dense_forward_values():
    X = np.array([[1, 2, 3], [4, 5, 6]])

    W = np.array([[1, 2], [3, 4]])

    b = np.array([[1], [2]])

    result = dense_forward(X, W, b)

    expected = np.array([[10, 13, 16], [21, 28, 35]])

    assert np.array_equal(result, expected)


def test_dense_forward_output_shape():
    X = np.zeros((6, 10))
    W = np.zeros((4, 6))
    b = np.zeros((4, 1))

    result = dense_forward(X, W, b)

    assert result.shape == (4, 10)


def test_dense_forward_bias_broadcasting():
    X = np.zeros((3, 5))
    W = np.zeros((2, 3))
    b = np.array([[5], [10]])

    result = dense_forward(X, W, b)

    expected = np.array([[5, 5, 5, 5, 5], [10, 10, 10, 10, 10]])

    assert np.array_equal(result, expected)


def test_dense_forward_invalid_X_dimension():
    X = np.zeros(5)
    W = np.zeros((3, 5))
    b = np.zeros((3, 1))

    with pytest.raises(ValueError):
        dense_forward(X, W, b)


def test_dense_forward_invalid_W_dimension():
    X = np.zeros((5, 4))
    W = np.zeros(5)
    b = np.zeros((3, 1))

    with pytest.raises(ValueError):
        dense_forward(X, W, b)


def test_dense_forward_invalid_b_dimension():
    X = np.zeros((5, 4))
    W = np.zeros((3, 5))
    b = np.zeros(3)

    with pytest.raises(ValueError):
        dense_forward(X, W, b)


def test_dense_forward_feature_mismatch():
    X = np.zeros((6, 4))
    W = np.zeros((3, 5))
    b = np.zeros((3, 1))

    with pytest.raises(ValueError):
        dense_forward(X, W, b)


def test_dense_forward_bias_neuron_mismatch():
    X = np.zeros((5, 4))
    W = np.zeros((3, 5))
    b = np.zeros((2, 1))

    with pytest.raises(ValueError):
        dense_forward(X, W, b)


def test_dense_forward_invalid_bias_shape():
    X = np.zeros((5, 4))
    W = np.zeros((3, 5))
    b = np.zeros((3, 2))

    with pytest.raises(ValueError):
        dense_forward(X, W, b)

