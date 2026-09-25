"""Tests for forward convolution behavior and input validation."""

import numpy as np
import pytest

from ..implementations.convolution import (
    convolve2d,
    convolve_multichannel,
    convolve_multifilter,
)

# ============================================================
# 2D CONVOLUTION
# ============================================================


@pytest.mark.parametrize(
    "stride, padding, expected_shape",
    [
        (1, 0, (8, 8)),
        (1, 1, (10, 10)),
        (2, 0, (4, 4)),
        (2, 1, (5, 5)),
    ],
)
def test_convolution_output_shape(stride, padding, expected_shape):

    image = np.zeros((10, 10))

    kernel = np.array(
        [
            [-1, 0, 1],
            [-1, 0, 1],
            [-1, 0, 1],
        ]
    )

    result = convolve2d(
        image,
        kernel,
        stride,
        padding,
    )

    assert result.shape == expected_shape


def test_convolution_numerical_result():

    image = np.array(
        [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ]
    )

    kernel = np.array(
        [
            [1, 0],
            [0, 1],
        ]
    )

    expected = np.array(
        [
            [6, 8],
            [12, 14],
        ]
    )

    result = convolve2d(
        image,
        kernel,
    )

    assert np.array_equal(
        result,
        expected,
    )


# ============================================================
# MULTI-CHANNEL CONVOLUTION
# ============================================================


def test_multichannel_convolution_numerical_result():

    image = np.array(
        [
            [[1, 5], [2, 6]],
            [[3, 7], [4, 8]],
        ]
    )

    kernel = np.array(
        [
            [[1, 0], [0, 1]],
            [[0, 1], [1, 0]],
        ]
    )

    expected = np.array([[18]])

    result = convolve_multichannel(
        image,
        kernel,
    )

    assert np.array_equal(
        result,
        expected,
    )


@pytest.mark.parametrize(
    "stride, padding, expected_shape",
    [
        (1, 0, (3, 3)),
        (1, 1, (5, 5)),
        (2, 0, (2, 2)),
        (2, 1, (3, 3)),
    ],
)
def test_multichannel_convolution_output_shape(
    stride,
    padding,
    expected_shape,
):

    image = np.zeros((4, 4, 2))
    kernel = np.ones((2, 2, 2))

    result = convolve_multichannel(
        image,
        kernel,
        stride=stride,
        padding=padding,
    )

    assert result.shape == expected_shape


def test_multichannel_convolution_rejects_invalid_stride():

    image = np.zeros((4, 4, 2))
    kernel = np.ones((2, 2, 2))

    with pytest.raises(ValueError):
        convolve_multichannel(
            image,
            kernel,
            stride=0,
        )


def test_multichannel_convolution_rejects_negative_padding():

    image = np.zeros((4, 4, 2))
    kernel = np.ones((2, 2, 2))

    with pytest.raises(ValueError):
        convolve_multichannel(
            image,
            kernel,
            padding=-1,
        )


def test_multichannel_convolution_rejects_2d_image():

    image = np.zeros((4, 4))
    kernel = np.ones((2, 2, 1))

    with pytest.raises(ValueError):
        convolve_multichannel(
            image,
            kernel,
        )


def test_multichannel_convolution_rejects_2d_kernel():

    image = np.zeros((4, 4, 1))
    kernel = np.ones((2, 2))

    with pytest.raises(ValueError):
        convolve_multichannel(
            image,
            kernel,
        )


def test_multichannel_convolution_rejects_channel_mismatch():

    image = np.zeros((4, 4, 3))
    kernel = np.ones((2, 2, 2))

    with pytest.raises(ValueError):
        convolve_multichannel(
            image,
            kernel,
        )


def test_multichannel_convolution_rejects_large_kernel():

    image = np.zeros((3, 3, 2))
    kernel = np.ones((4, 4, 2))

    with pytest.raises(ValueError):
        convolve_multichannel(
            image,
            kernel,
        )


# ============================================================
# MULTI-FILTER CONVOLUTION
# ============================================================


def test_convolve_multifilter_output_shape():

    image = np.ones((5, 5, 2))

    filters = np.ones((3, 3, 2, 4))

    bias = np.zeros(4)

    result = convolve_multifilter(
        image,
        filters,
        bias,
    )

    assert result.shape == (3, 3, 4)


def test_convolve_multifilter_values():

    image = np.array(
        [
            [[1, 5], [2, 6]],
            [[3, 7], [4, 8]],
        ]
    )

    filter_0 = np.array(
        [
            [[1, 0], [0, 1]],
            [[0, 1], [1, 0]],
        ]
    )

    filter_1 = np.array(
        [
            [[1, -1], [1, 0]],
            [[1, 0], [1, -1]],
        ]
    )

    filters = np.stack(
        [
            filter_0,
            filter_1,
        ],
        axis=-1,
    )

    bias = np.array([0.0, 0.0])

    result = convolve_multifilter(
        image,
        filters,
        bias,
    )

    expected = np.array(
        [
            [[18, -3]],
        ]
    )

    assert np.array_equal(
        result,
        expected,
    )


# ============================================================
# BIAS
# ============================================================


def test_convolve_multifilter_bias():

    image = np.ones((3, 3, 1))

    filters = np.ones((2, 2, 1, 2))

    bias = np.array([10.0, -5.0])

    result = convolve_multifilter(
        image,
        filters,
        bias,
    )

    expected = np.array(
        [
            [
                [14.0, -1.0],
                [14.0, -1.0],
            ],
            [
                [14.0, -1.0],
                [14.0, -1.0],
            ],
        ]
    )

    assert np.array_equal(
        result,
        expected,
    )


def test_convolve_multifilter_rejects_invalid_bias_dimension():

    image = np.ones((3, 3, 1))

    filters = np.ones((2, 2, 1, 2))

    bias = np.zeros((2, 1))

    with pytest.raises(ValueError):
        convolve_multifilter(
            image,
            filters,
            bias,
        )


def test_convolve_multifilter_rejects_invalid_bias_size():

    image = np.ones((3, 3, 1))

    filters = np.ones((2, 2, 1, 2))

    bias = np.zeros(3)

    with pytest.raises(ValueError):
        convolve_multifilter(
            image,
            filters,
            bias,
        )


# ============================================================
# NUMBER OF FILTERS
# ============================================================


@pytest.mark.parametrize(
    "num_filters",
    [1, 2, 3, 5],
)
def test_convolve_multifilter_number_of_filters(
    num_filters,
):

    image = np.ones((6, 6, 2))

    filters = np.ones((3, 3, 2, num_filters))

    bias = np.zeros(num_filters)

    result = convolve_multifilter(
        image,
        filters,
        bias,
    )

    assert result.shape == (
        4,
        4,
        num_filters,
    )


# ============================================================
# STRIDE
# ============================================================


def test_convolve_multifilter_stride():

    image = np.ones((7, 7, 2))

    filters = np.ones((3, 3, 2, 2))

    bias = np.zeros(2)

    result = convolve_multifilter(
        image,
        filters,
        bias,
        stride=2,
    )

    assert result.shape == (
        3,
        3,
        2,
    )


# ============================================================
# PADDING
# ============================================================


def test_convolve_multifilter_padding():

    image = np.ones((5, 5, 2))

    filters = np.ones((3, 3, 2, 3))

    bias = np.zeros(3)

    result = convolve_multifilter(
        image,
        filters,
        bias,
        padding=1,
    )

    assert result.shape == (
        5,
        5,
        3,
    )


# ============================================================
# INVALID STRIDE
# ============================================================


def test_convolve_multifilter_invalid_stride():

    image = np.ones((5, 5, 2))

    filters = np.ones((3, 3, 2, 2))

    bias = np.zeros(2)

    with pytest.raises(ValueError):
        convolve_multifilter(
            image,
            filters,
            bias,
            stride=0,
        )


# ============================================================
# INVALID PADDING
# ============================================================


def test_convolve_multifilter_invalid_padding():

    image = np.ones((5, 5, 2))

    filters = np.ones((3, 3, 2, 2))

    bias = np.zeros(2)

    with pytest.raises(ValueError):
        convolve_multifilter(
            image,
            filters,
            bias,
            padding=-1,
        )


# ============================================================
# INVALID IMAGE
# ============================================================


def test_convolve_multifilter_invalid_image():

    image = np.ones((5, 5))

    filters = np.ones((3, 3, 1, 2))

    bias = np.zeros(2)

    with pytest.raises(ValueError):
        convolve_multifilter(
            image,
            filters,
            bias,
        )


# ============================================================
# INVALID FILTERS
# ============================================================


def test_convolve_multifilter_invalid_filters():

    image = np.ones((5, 5, 1))

    filters = np.ones((3, 3, 1))

    bias = np.zeros(2)

    with pytest.raises(ValueError):
        convolve_multifilter(
            image,
            filters,
            bias,
        )
