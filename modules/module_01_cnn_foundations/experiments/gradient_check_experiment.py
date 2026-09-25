"""Finite-difference checks for multi-filter convolution backpropagation."""

import numpy as np

from ..implementations.convolution import (
    convolve_multifilter,
    convolve_multifilter_backward,
)

# ============================================================
# Configuration
# ============================================================

EPSILON = 1e-5
RELATIVE_ERROR_THRESHOLD = 1e-7


# ============================================================
# Numerical gradient
# ============================================================


def numerical_gradient(parameter, loss_function, epsilon=EPSILON):
    """
    Calculate the numerical gradient of a scalar loss
    with respect to every element of a parameter.

    Uses the central finite-difference approximation:

        dJ/dtheta ≈
        [J(theta + epsilon) - J(theta - epsilon)]
        / (2 * epsilon)
    """

    gradient = np.zeros_like(parameter, dtype=float)

    for index in np.ndindex(parameter.shape):

        original_value = parameter[index]

        # theta + epsilon
        parameter[index] = original_value + epsilon
        loss_plus = loss_function()

        # theta - epsilon
        parameter[index] = original_value - epsilon
        loss_minus = loss_function()

        # Restore original parameter
        parameter[index] = original_value

        # Central difference
        gradient[index] = (loss_plus - loss_minus) / (2 * epsilon)

    return gradient


# ============================================================
# Relative error
# ============================================================


def relative_error(analytical, numerical):
    """
    Calculate relative error between two gradients.
    """

    numerator = np.linalg.norm(analytical - numerical)

    denominator = np.linalg.norm(analytical) + np.linalg.norm(numerical)

    if denominator == 0:
        return 0.0

    return numerator / denominator


# ============================================================
# Gradient check
# ============================================================


def check_gradient(
    analytical,
    numerical,
    name,
    threshold=RELATIVE_ERROR_THRESHOLD,
):
    """
    Compare analytical and numerical gradients.
    """

    error = relative_error(
        analytical,
        numerical,
    )

    print(f"\n{name}")
    print("-" * 60)

    print("Analytical gradient:")
    print(analytical)

    print("\nNumerical gradient:")
    print(numerical)

    print(f"\nRelative error: {error:.12e}")

    if error < threshold:
        print("Status: PASSED")
    else:
        print("Status: FAILED")

    return error


# ============================================================
# Test case
# ============================================================


def run_gradient_check(
    image,
    filters,
    bias,
    stride,
    padding,
    test_name,
):
    """
    Run a complete gradient check for convolution.

    Checks:
        dInput
        dFilters
        dBias
    """

    print("\n")
    print("=" * 70)
    print(test_name)
    print("=" * 70)

    # --------------------------------------------------------
    # Forward pass
    # --------------------------------------------------------

    def loss_function():
        output = convolve_multifilter(
            image,
            filters,
            bias,
            stride=stride,
            padding=padding,
        )

        # Simple scalar loss
        return np.sum(output)

    loss = loss_function()

    print("\nInput shape :", image.shape)
    print("Filter shape:", filters.shape)
    print("Bias shape  :", bias.shape)
    print("Stride      :", stride)
    print("Padding     :", padding)
    print("Loss        :", loss)

    # --------------------------------------------------------
    # Analytical gradients
    # --------------------------------------------------------

    output = convolve_multifilter(
        image,
        filters,
        bias,
        stride=stride,
        padding=padding,
    )

    dZ = np.ones_like(output)

    dInput, dFilters, dBias = convolve_multifilter_backward(
        dZ,
        image,
        filters,
        stride=stride,
        padding=padding,
    )

    # --------------------------------------------------------
    # Numerical gradient for input
    # --------------------------------------------------------

    numerical_dInput = numerical_gradient(
        image,
        loss_function,
    )

    input_error = check_gradient(
        dInput,
        numerical_dInput,
        "Input gradient (dInput)",
    )

    # --------------------------------------------------------
    # Numerical gradient for filters
    # --------------------------------------------------------

    numerical_dFilters = numerical_gradient(
        filters,
        loss_function,
    )

    filter_error = check_gradient(
        dFilters,
        numerical_dFilters,
        "Filter gradient (dFilters)",
    )

    # --------------------------------------------------------
    # Numerical gradient for bias
    # --------------------------------------------------------

    numerical_dBias = numerical_gradient(
        bias,
        loss_function,
    )

    bias_error = check_gradient(
        dBias,
        numerical_dBias,
        "Bias gradient (dBias)",
    )

    return (
        input_error,
        filter_error,
        bias_error,
    )


# ============================================================
# Main
# ============================================================


def main():

    # ========================================================
    # Test 1
    # Single channel, single filter
    # No padding, stride = 1
    # ========================================================

    image = np.array(
        [
            [[1.0], [2.0], [3.0]],
            [[4.0], [5.0], [6.0]],
            [[7.0], [8.0], [9.0]],
        ]
    )

    filters = np.array(
        [
            [
                [[0.1]],
                [[0.2]],
            ],
            [
                [[0.3]],
                [[0.4]],
            ],
        ]
    )

    bias = np.array([0.1])

    (
        error_input_1,
        error_filters_1,
        error_bias_1,
    ) = run_gradient_check(
        image=image,
        filters=filters,
        bias=bias,
        stride=1,
        padding=0,
        test_name="TEST 1: Single channel / Single filter",
    )

    # ========================================================
    # Test 2
    # Multiple channels, multiple filters
    # No padding, stride = 1
    # ========================================================

    image = np.array(
        [
            [
                [1.0, 2.0],
                [3.0, 4.0],
                [5.0, 6.0],
            ],
            [
                [7.0, 8.0],
                [9.0, 10.0],
                [11.0, 12.0],
            ],
            [
                [13.0, 14.0],
                [15.0, 16.0],
                [17.0, 18.0],
            ],
        ]
    )

    filters = np.array(
        [
            [
                [[0.1, 0.5], [0.2, 0.6]],
                [[0.3, 0.7], [0.4, 0.8]],
            ],
            [
                [[0.2, 0.9], [0.1, 0.3]],
                [[0.4, 0.5], [0.7, 0.2]],
            ],
        ]
    )

    bias = np.array([0.1, -0.2])

    (
        error_input_2,
        error_filters_2,
        error_bias_2,
    ) = run_gradient_check(
        image=image,
        filters=filters,
        bias=bias,
        stride=1,
        padding=0,
        test_name="TEST 2: Multiple channels / Multiple filters",
    )

    # ========================================================
    # Test 3
    # Padding = 1
    # ========================================================

    image = np.array(
        [
            [
                [1.0, 2.0],
                [3.0, 4.0],
                [5.0, 6.0],
            ],
            [
                [7.0, 8.0],
                [9.0, 10.0],
                [11.0, 12.0],
            ],
            [
                [13.0, 14.0],
                [15.0, 16.0],
                [17.0, 18.0],
            ],
        ]
    )

    filters = np.array(
        [
            [
                [[0.1, 0.5], [0.2, 0.6]],
                [[0.3, 0.7], [0.4, 0.8]],
            ],
            [
                [[0.2, 0.9], [0.1, 0.3]],
                [[0.4, 0.5], [0.7, 0.2]],
            ],
        ]
    )

    bias = np.array([0.3, -0.4])

    (
        error_input_3,
        error_filters_3,
        error_bias_3,
    ) = run_gradient_check(
        image=image,
        filters=filters,
        bias=bias,
        stride=1,
        padding=1,
        test_name="TEST 3: Padding = 1",
    )

    # ========================================================
    # Test 4
    # Stride = 2
    # ========================================================

    image = np.array(
        [
            [
                [1.0, 2.0],
                [3.0, 4.0],
                [5.0, 6.0],
                [7.0, 8.0],
                [9.0, 10.0],
            ],
            [
                [11.0, 12.0],
                [13.0, 14.0],
                [15.0, 16.0],
                [17.0, 18.0],
                [19.0, 20.0],
            ],
            [
                [21.0, 22.0],
                [23.0, 24.0],
                [25.0, 26.0],
                [27.0, 28.0],
                [29.0, 30.0],
            ],
            [
                [31.0, 32.0],
                [33.0, 34.0],
                [35.0, 36.0],
                [37.0, 38.0],
                [39.0, 40.0],
            ],
            [
                [41.0, 42.0],
                [43.0, 44.0],
                [45.0, 46.0],
                [47.0, 48.0],
                [49.0, 50.0],
            ],
        ]
    )

    filters = np.array(
        [
            [
                [[0.1, 0.5], [0.2, 0.6]],
                [[0.3, 0.7], [0.4, 0.8]],
            ],
            [
                [[0.2, 0.9], [0.1, 0.3]],
                [[0.4, 0.5], [0.7, 0.2]],
            ],
        ]
    )

    bias = np.array([0.2, -0.3])

    (
        error_input_4,
        error_filters_4,
        error_bias_4,
    ) = run_gradient_check(
        image=image,
        filters=filters,
        bias=bias,
        stride=2,
        padding=0,
        test_name="TEST 4: Stride = 2",
    )

    # ========================================================
    # Final summary
    # ========================================================

    all_errors = [
        error_input_1,
        error_filters_1,
        error_bias_1,
        error_input_2,
        error_filters_2,
        error_bias_2,
        error_input_3,
        error_filters_3,
        error_bias_3,
        error_input_4,
        error_filters_4,
        error_bias_4,
    ]

    maximum_error = max(all_errors)

    print("\n")
    print("=" * 70)
    print("FINAL GRADIENT CHECK SUMMARY")
    print("=" * 70)

    print(f"\nMaximum relative error: " f"{maximum_error:.12e}")

    if maximum_error < RELATIVE_ERROR_THRESHOLD:
        print("\nALL GRADIENT CHECKS PASSED")
        print("Convolution backward propagation is verified.")
        print("Input, filter, and bias gradients are verified.")
    else:
        print("\nGRADIENT CHECK FAILED")
        print("Inspect the test with the largest relative error.")


if __name__ == "__main__":
    main()
