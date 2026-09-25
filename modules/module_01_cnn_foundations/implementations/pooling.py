import numpy as np


def max_pool2D(image, pool_size=2, stride=1, padding=0):
    """Apply max pooling to a 2D image and return ``(H_out, W_out)``."""
    if image.ndim != 2 :
        raise ValueError("Image must be a 2D array")
    if pool_size <= 0 :
        raise ValueError("The pooling size must be greater than zero")
    if stride <= 0 :
        raise ValueError("The stride must be greater than zero")
    if padding < 0:
        raise ValueError("The padding must not be negative")

    img_height, img_width = image.shape

    padded_image = np.pad(image, ((padding,padding), (padding,padding)), mode="constant")
    padded_height, padded_width = padded_image.shape

    if pool_size > padded_height or pool_size > padded_width:
        raise ValueError("Pooling window cannot be larger than the image")

    H_out = (img_height + 2*padding - pool_size) // stride + 1
    W_out = (img_width + 2*padding - pool_size) // stride + 1

    pool_img = np.zeros((H_out, W_out))

    for i in range(H_out):
        for j in range(W_out):

            region = padded_image[
                i * stride : i * stride + pool_size,
                j * stride : j * stride + pool_size
            ]

            pool_img[i,j] = np.max(region)

    return pool_img


def max_pool_multichannel(image, pool_size=2, stride=1, padding=0):
    """Apply independent max pooling to each channel of ``(H, W, C)``."""

    if image.ndim != 3:
        raise ValueError("Image must be a 3D array")

    _, _, C = image.shape

    feature_maps = []

    for i in range(C):

        channel = image[:, :, i]

        feature_map = max_pool2D(
            channel,
            pool_size=pool_size,
            stride=stride,
            padding=padding
        )

        feature_maps.append(feature_map)

    pool_img = np.stack(feature_maps, axis=-1)

    return pool_img


def average_pool2D(image, pool_size=2, stride=1, padding=0):
    """Apply average pooling to a 2D image and return ``(H_out, W_out)``."""
    if image.ndim != 2:
        raise ValueError("Image must be a 2D array")
    if pool_size <= 0:
        raise ValueError("The pooling size must be greater than zero")
    if stride <= 0:
        raise ValueError("The stride must be greater than zero")
    if padding < 0:
        raise ValueError("The padding must not be negative")

    img_height, img_width = image.shape

    padded_image = np.pad(
        image, ((padding, padding), (padding, padding)), mode="constant"
    )
    padded_height, padded_width = padded_image.shape

    if pool_size > padded_height or pool_size > padded_width:
        raise ValueError("Pooling window cannot be larger than the image")

    H_out = (img_height + 2 * padding - pool_size) // stride + 1
    W_out = (img_width + 2 * padding - pool_size) // stride + 1

    pool_img = np.zeros((H_out, W_out))

    for i in range(H_out):
        for j in range(W_out):

            region = padded_image[
                i * stride : i * stride + pool_size, j * stride : j * stride + pool_size
            ]

            pool_img[i, j] = np.mean(region)

    return pool_img


def average_pool_multichannel(image, pool_size=2, stride=1, padding=0):
    """Apply independent average pooling to each channel of ``(H, W, C)``."""

    if image.ndim != 3:
        raise ValueError("Image must be a 3D array")

    _, _, C = image.shape

    feature_maps = []

    for i in range(C):

        channel = image[:, :, i]

        feature_map = average_pool2D(
            channel, pool_size=pool_size, stride=stride, padding=padding
        )

        feature_maps.append(feature_map)

    pool_img = np.stack(feature_maps, axis=-1)

    return pool_img


def max_pool_backward(dA, input, pool_size=2, stride=1, padding=0):
    """Route each pooled gradient to the selected maximum input element."""
    if dA.ndim != 2:
        raise ValueError("dA must be a 2D array")

    if input.ndim != 2:
        raise ValueError("input must be a 2D array")

    if pool_size <= 0:
        raise ValueError("pool_size must be greater than zero")

    if stride <= 0:
        raise ValueError("stride must be greater than zero")

    if padding < 0:
        raise ValueError("padding cannot be negative")

    padded_input = np.pad(
        input, ((padding, padding), (padding, padding)), mode="constant"
    )

    padded_height, padded_width = padded_input.shape

    if pool_size > padded_height or pool_size > padded_width:
        raise ValueError("pooling window cannot be larger than the padded input")

    output_height = (padded_height - pool_size) // stride + 1
    output_width = (padded_width - pool_size) // stride + 1

    if dA.shape != (output_height, output_width):
        raise ValueError("dA shape does not match the pooling output shape")

    d_padded = np.zeros_like(padded_input, dtype=float)

    for i in range(output_height):
        for j in range(output_width):

            region = padded_input[
                i * stride : i * stride + pool_size, j * stride : j * stride + pool_size
            ]

            max_position = np.unravel_index(np.argmax(region), region.shape)

            max_row, max_col = max_position

            d_padded[i * stride + max_row, j * stride + max_col] += dA[i, j]

    if padding > 0:
        dX = d_padded[padding:-padding, padding:-padding]
    else:
        dX = d_padded

    return dX


def max_pool_multichannel_backward(dA, input, pool_size=2, stride=1, padding=0):
    """Apply max-pooling backpropagation independently across channels."""
    if dA.ndim != 3:
        raise ValueError("dA must be a 3D array")

    if input.ndim != 3:
        raise ValueError("input must be a 3D array")

    if dA.shape[2] != input.shape[2]:
        raise ValueError("dA and input must have the same number of channels")

    gradients = []

    for i in range(input.shape[2]):
        channel_input = input[:, :, i]
        channel_dA = dA[:, :, i]

        channel_dX = max_pool_backward(
            channel_dA,
            channel_input,
            pool_size=pool_size,
            stride=stride,
            padding=padding,
        )

        gradients.append(channel_dX)

    dX = np.stack(gradients, axis=-1)

    return dX
