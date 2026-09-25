import numpy as np


def convolve2d(image, kernel, stride=1, padding=0):
    """Compute a single-channel 2D cross-correlation.

    The kernel is applied without spatial reversal, matching the convention
    commonly used by CNN implementations. Zero-padding is added before the
    sliding-window computation.

    Parameters:
        image: Input with shape ``(H, W)``.
        kernel: Kernel with shape ``(KH, KW)``.
        stride: Number of pixels the kernel moves at each step.
        padding: Number of zero-padding pixels added around the image.

    Returns:
        Array with shape ``(H_out, W_out)``.
    """

    if stride <= 0:
        raise ValueError("Stride must be greater than 0")

    if padding < 0:
        raise ValueError("Negative padding is not allowed")

    if image.ndim != 2:
        raise ValueError("Image must be a 2D array.")

    if kernel.ndim != 2:
        raise ValueError("Kernel must be a 2D array.")

    img_height, img_width = image.shape
    kernel_height, kernel_width = kernel.shape

    padded_image = np.pad(
        image, ((padding, padding), (padding, padding)), mode="constant"
    )

    padded_height, padded_width = padded_image.shape

    if kernel_height > padded_height or kernel_width > padded_width:
        raise ValueError("Kernel cannot be larger than the image.")

    H_out = (img_height + 2 * padding - kernel_height) // stride + 1

    W_out = (img_width + 2 * padding - kernel_width) // stride + 1

    conv_img = np.zeros((H_out, W_out))

    for i in range(H_out):
        for j in range(W_out):

            region = padded_image[
                i * stride : i * stride + kernel_height,
                j * stride : j * stride + kernel_width,
            ]

            conv_img[i, j] = np.sum(region * kernel)

    return conv_img


def convolve_multichannel(image, kernel, stride=1, padding=0):
    """Compute a convolution over all channels of one image.

    Parameters:
        image: Input with shape ``(H, W, C)``.
        kernel: Kernel with shape ``(KH, KW, C)``.
        stride: Spatial step between neighboring windows.
        padding: Zero-padding width.

    Returns:
        A single feature map with shape ``(H_out, W_out)``.
    """
    if stride <= 0:
        raise ValueError("Stride must be greater than 0")

    if padding < 0:
        raise ValueError("Negative padding is not allowed")

    if image.ndim != 3:
        raise ValueError("Image must be a 3D array")

    if kernel.ndim != 3:
        raise ValueError("Kernel must be a 3D array")

    img_height, img_width, img_channel = image.shape
    kernel_height, kernel_width, kernel_channel = kernel.shape

    if img_channel != kernel_channel:
        raise ValueError("The image and kernel channels should be equal")

    padded_image = np.pad(
        image,
        (
            (padding, padding),
            (padding, padding),
            (0, 0),
        ),
        mode="constant",
    )

    padded_height, padded_width, _ = padded_image.shape

    if kernel_height > padded_height or kernel_width > padded_width:
        raise ValueError("Kernel cannot be larger than the image")

    H_out = (img_height + 2 * padding - kernel_height) // stride + 1

    W_out = (img_width + 2 * padding - kernel_width) // stride + 1

    conv_img = np.zeros((H_out, W_out))

    for i in range(H_out):
        for j in range(W_out):

            region = padded_image[
                i * stride : i * stride + kernel_height,
                j * stride : j * stride + kernel_width,
                :,
            ]

            conv_img[i, j] = np.sum(region * kernel)

    return conv_img


def convolve_multifilter(image, filters, bias, stride=1, padding=0):
    """
    Perform convolution using multiple filters and add one bias per filter.

    For each output position and filter ``n`` this computes:

    ``Z[i, j, n] = sum(X_region * filters[:, :, :, n]) + bias[n]``.

    Parameters
    ----------
    image : ndarray
        Shape: ``(H, W, C)``.

    filters : ndarray
        Shape: ``(KH, KW, C, N)``.

    bias : ndarray
        Shape: ``(N,)``.

    stride : int
        Convolution stride.

    padding : int
        Zero padding.

    Returns
    -------
    conv_img : ndarray
        Shape: ``(H_out, W_out, N)``.
    """

    if stride <= 0:
        raise ValueError("Stride must be greater than 0")

    if padding < 0:
        raise ValueError("Negative padding are not allowed")

    if image.ndim != 3:
        raise ValueError("Image must be a 3D array")

    if filters.ndim != 4:
        raise ValueError("Filters must be a 4D array")

    if bias.ndim != 1:
        raise ValueError("Bias must be a 1D array")

    if filters.shape[2] != image.shape[2]:
        raise ValueError("Image and filters must have the same number of channels")

    _, _, _, num_filters = filters.shape

    if bias.shape[0] != num_filters:
        raise ValueError("Bias size must match the number of filters")

    feature_maps = []

    for i in range(num_filters):

        kernel = filters[:, :, :, i]

        feature_map = convolve_multichannel(
            image,
            kernel,
            stride,
            padding,
        )

        # Add the bias for this filter
        feature_map += bias[i]

        feature_maps.append(feature_map)

    conv_img = np.stack(feature_maps, axis=-1)

    return conv_img


def convolve_multifilter_backward(dZ, image, filters, stride=1, padding=0):
    """
    Backward propagation for multi-filter convolution.

    ``dZ`` has shape ``(H_out, W_out, N)``. The returned gradients have
    shapes ``dX: (H, W, C)``, ``dW: (KH, KW, C, N)``, and ``db: (N,)``.

    Returns
    -------
        dX : ndarray
            Gradient with respect to input.

    dW : ndarray
        Gradient with respect to filters.

        db : ndarray
            Gradient with respect to bias.
    """

    if dZ.ndim != 3:
        raise ValueError("dZ must be a 3D array")

    if image.ndim != 3:
        raise ValueError("Image must be a 3D array")

    if filters.ndim != 4:
        raise ValueError("Filters must be a 4D array")

    if stride <= 0:
        raise ValueError("Stride must be greater than zero")

    if padding < 0:
        raise ValueError("Padding cannot be negative")

    if filters.shape[2] != image.shape[2]:
        raise ValueError("Image and filters must have the same number of channels")

    kernel_height, kernel_width, _, num_filters = filters.shape

    padded_image = np.pad(
        image, ((padding, padding), (padding, padding), (0, 0)), mode="constant"
    )

    padded_height, padded_width, _ = padded_image.shape

    if kernel_height > padded_height or kernel_width > padded_width:
        raise ValueError("Kernel cannot be larger than the padded image")

    output_height = (padded_height - kernel_height) // stride + 1

    output_width = (padded_width - kernel_width) // stride + 1

    if dZ.shape != (output_height, output_width, num_filters):
        raise ValueError("dZ shape does not match the convolution output shape")

    d_padded = np.zeros_like(padded_image, dtype=float)

    dW = np.zeros_like(filters, dtype=float)

    db = np.zeros(num_filters, dtype=float)

    for n in range(num_filters):

        kernel = filters[:, :, :, n]

        for i in range(output_height):
            for j in range(output_width):

                region = padded_image[
                    i * stride : i * stride + kernel_height,
                    j * stride : j * stride + kernel_width,
                    :,
                ]

                gradient = dZ[i, j, n]

                # Gradient with respect to filter
                dW[:, :, :, n] += gradient * region

                # Gradient with respect to input
                d_padded[
                    i * stride : i * stride + kernel_height,
                    j * stride : j * stride + kernel_width,
                    :,
                ] += (
                    gradient * kernel
                )

                # Gradient with respect to bias
                db[n] += gradient

    if padding > 0:
        dX = d_padded[padding:-padding, padding:-padding, :]
    else:
        dX = d_padded

    return dX, dW, db
