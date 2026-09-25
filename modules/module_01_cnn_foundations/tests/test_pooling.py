"""Tests for max-pooling and average-pooling behavior."""

import pytest
import numpy as np

from ..implementations.pooling import max_pool2D, max_pool_multichannel, average_pool2D, average_pool_multichannel


def test_max_pool2d_values():
    image = np.array([[1, 5, 2, 4], [3, 7, 1, 2], [8, 2, 6, 3], [4, 1, 5, 9]])

    result = max_pool2D(image, pool_size=2, stride=2)

    expected = np.array([[7, 4], [8, 9]])

    assert np.array_equal(result, expected)


def test_max_pool2d_stride_one():
    image = np.array([[1, 5, 2, 4], [3, 7, 1, 2], [8, 2, 6, 3], [4, 1, 5, 9]])

    result = max_pool2D(image, pool_size=2, stride=1)

    expected = np.array([[7, 7, 4], [8, 7, 6], [8, 6, 9]])

    assert np.array_equal(result, expected)


def test_max_pool2d_output_shape():
    image = np.zeros((6, 6))

    result = max_pool2D(image, pool_size=2, stride=2)

    assert result.shape == (3, 3)


def test_max_pool2d_output_shape_stride_one():
    image = np.zeros((6, 6))

    result = max_pool2D(image, pool_size=3, stride=1)

    assert result.shape == (4, 4)


def test_max_pool2d_padding():
    image = np.array([[1, 2], [3, 4]])

    result = max_pool2D(image, pool_size=2, stride=1, padding=1)

    assert result.shape == (3, 3)


def test_max_pool2d_invalid_pool_size():
    image = np.zeros((4, 4))

    with pytest.raises(ValueError):
        max_pool2D(image, pool_size=0)


def test_max_pool2d_invalid_stride():
    image = np.zeros((4, 4))

    with pytest.raises(ValueError):
        max_pool2D(image, stride=0)


def test_max_pool2d_negative_padding():
    image = np.zeros((4, 4))

    with pytest.raises(ValueError):
        max_pool2D(image, padding=-1)


def test_max_pool2d_invalid_image():
    image = np.zeros((4, 4, 3))

    with pytest.raises(ValueError):
        max_pool2D(image)


def test_max_pool2d_pool_size_too_large():
    image = np.zeros((3, 3))

    with pytest.raises(ValueError):
        max_pool2D(image, pool_size=4)


def test_max_pool_multichannel_values():
    image = np.array([[[1, 5], [2, 6]], [[3, 7], [4, 8]]])

    result = max_pool_multichannel(image, pool_size=2, stride=1)

    expected = np.array([[[4, 8]]])

    assert np.array_equal(result, expected)


def test_max_pool_multichannel_output_shape():
    image = np.zeros((6, 6, 3))

    result = max_pool_multichannel(image, pool_size=2, stride=2)

    assert result.shape == (3, 3, 3)


@pytest.mark.parametrize("channels", [1, 2, 3, 5])
def test_max_pool_multichannel_number_of_channels(channels):
    image = np.zeros((4, 4, channels))

    result = max_pool_multichannel(image, pool_size=2, stride=2)

    assert result.shape == (2, 2, channels)


def test_max_pool_multichannel_stride_one():
    image = np.array(
        [
            [[1, 10], [5, 20], [2, 30]],
            [[3, 40], [7, 50], [1, 60]],
            [[8, 70], [2, 80], [6, 90]],
        ]
    )

    result = max_pool_multichannel(image, pool_size=2, stride=1)

    expected = np.array([[[7, 50], [7, 60]], [[8, 80], [7, 90]]])

    assert np.array_equal(result, expected)


def test_max_pool_multichannel_padding():
    image = np.array([[[1, 5], [2, 6]], [[3, 7], [4, 8]]])

    result = max_pool_multichannel(image, pool_size=2, stride=1, padding=1)

    assert result.shape == (3, 3, 2)


def test_max_pool_multichannel_invalid_image():
    image = np.zeros((4, 4))

    with pytest.raises(ValueError):
        max_pool_multichannel(image)


def test_max_pool_multichannel_4d_input():
    image = np.zeros((4, 4, 3, 2))

    with pytest.raises(ValueError):
        max_pool_multichannel(image)


def test_average_pool2d_values():
    image = np.array([[1, 5], [3, 7]])

    result = average_pool2D(image, pool_size=2, stride=1)

    expected = np.array([[4.0]])

    assert np.array_equal(result, expected)


def test_average_pool2d_multiple_windows():
    image = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])

    result = average_pool2D(image, pool_size=2, stride=2)

    expected = np.array([[3.5, 5.5], [11.5, 13.5]])

    assert np.array_equal(result, expected)


def test_average_pool2d_stride_one():
    image = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    result = average_pool2D(image, pool_size=2, stride=1)

    expected = np.array([[3.0, 4.0], [6.0, 7.0]])

    assert np.array_equal(result, expected)


def test_average_pool2d_output_shape():
    image = np.zeros((6, 6))

    result = average_pool2D(image, pool_size=2, stride=2)

    assert result.shape == (3, 3)


def test_average_pool2d_output_shape_stride_one():
    image = np.zeros((6, 6))

    result = average_pool2D(image, pool_size=3, stride=1)

    assert result.shape == (4, 4)


def test_average_pool2d_padding():
    image = np.array([[1, 2], [3, 4]])

    result = average_pool2D(image, pool_size=2, stride=1, padding=1)

    assert result.shape == (3, 3)


def test_average_pool_multichannel_values():
    image = np.array([[[1, 5], [2, 6]], [[3, 7], [4, 8]]])

    result = average_pool_multichannel(image, pool_size=2, stride=1)

    expected = np.array([[[2.5, 6.5]]])

    assert np.array_equal(result, expected)


def test_average_pool_multichannel_output_shape():
    image = np.zeros((6, 6, 3))

    result = average_pool_multichannel(image, pool_size=2, stride=2)

    assert result.shape == (3, 3, 3)


@pytest.mark.parametrize("channels", [1, 2, 3, 5])
def test_average_pool_multichannel_number_of_channels(channels):
    image = np.zeros((4, 4, channels))

    result = average_pool_multichannel(image, pool_size=2, stride=2)

    assert result.shape == (2, 2, channels)


def test_average_pool_multichannel_stride_one():
    image = np.array(
        [
            [[1, 10], [5, 20], [2, 30]],
            [[3, 40], [7, 50], [1, 60]],
            [[8, 70], [2, 80], [6, 90]],
        ]
    )

    result = average_pool_multichannel(image, pool_size=2, stride=1)

    expected = np.array([[[4.0, 30.0], [3.75, 40.0]], [[5.0, 60.0], [4.0, 70.0]]])

    assert np.array_equal(result, expected)


def test_average_pool_multichannel_padding():
    image = np.array([[[1, 5], [2, 6]], [[3, 7], [4, 8]]])

    result = average_pool_multichannel(image, pool_size=2, stride=1, padding=1)

    assert result.shape == (3, 3, 2)


def test_average_pool_multichannel_invalid_image():
    image = np.zeros((4, 4))

    with pytest.raises(ValueError):
        average_pool_multichannel(image)


def test_average_pool_multichannel_4d_input():
    image = np.zeros((4, 4, 3, 2))

    with pytest.raises(ValueError):
        average_pool_multichannel(image)


def test_average_pool2d_invalid_pool_size():
    image = np.zeros((4, 4))

    with pytest.raises(ValueError):
        average_pool2D(image, pool_size=0)


def test_average_pool2d_invalid_stride():
    image = np.zeros((4, 4))

    with pytest.raises(ValueError):
        average_pool2D(image, stride=0)


def test_average_pool2d_negative_padding():
    image = np.zeros((4, 4))

    with pytest.raises(ValueError):
        average_pool2D(image, padding=-1)


def test_average_pool2d_pool_size_too_large():
    image = np.zeros((3, 3))

    with pytest.raises(ValueError):
        average_pool2D(image, pool_size=4)
