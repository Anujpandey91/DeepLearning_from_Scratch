import numpy as np

from .activations import relu, relu_backward, softmax
from .convolution import convolve_multifilter, convolve_multifilter_backward
from .layers import dense_backward, dense_forward, flatten, flatten_backward
from .losses import categorical_cross_entropy, categorical_cross_entropy_backward
from .pooling import max_pool_multichannel, max_pool_multichannel_backward


class CNN:
    """A small NumPy CNN used to expose the mechanics of image classification.

    The fixed architecture is::

        Input -> Conv (3x3, 2 filters) -> ReLU -> MaxPool (2x2, stride 2)
              -> Flatten -> Dense (10 units) -> Softmax

    Inputs use image layout ``(H, W, C)``. Dense, loss, and softmax operations
    use the convention ``(features/classes, examples)``; ``fit`` supplies one
    example at a time, so its dense tensors have shape ``(n, 1)``.
    """

    def __init__(self, learning_rate=0.01, epochs=10):
        """Create a CNN configured for sample-by-sample SGD training."""
        self.learning_rate = learning_rate
        self.epochs = epochs

        self.parameters = {}
        self.loss_history = []

        from .optimizers import GradientDescent

        self.optimizer = GradientDescent(learning_rate=self.learning_rate)

    def _initialize_parameters(self, input_shape):
        """Initialize weights and biases for one input shape.

        ``input_shape`` is ``(H, W, C)``. The convolution weights have shape
        ``(3, 3, C, 2)``, and the dense output has ten classes. The flattened
        size is derived from the convolution and pooling output dimensions.
        """

        if len(input_shape) != 3:
            raise ValueError("input_shape must contain (height, width, channels)")

        input_height, input_width, input_channels = input_shape

        if input_height <= 0 or input_width <= 0:
            raise ValueError("Input height and width must be greater than zero")

        if input_channels <= 0:
            raise ValueError("Number of input channels must be greater than zero")

        # Reset parameters
        self.parameters = {}

        # --------------------------------------------------------
        # Convolution parameters
        # --------------------------------------------------------

        num_filters = 2
        kernel_height = 3
        kernel_width = 3

        self.parameters["W1"] = (
            np.random.randn(
                kernel_height,
                kernel_width,
                input_channels,
                num_filters,
            )
            * 0.01
        )

        self.parameters["b1"] = np.zeros(num_filters)

        # --------------------------------------------------------
        # Convolution configuration
        # --------------------------------------------------------

        conv_stride = 1
        conv_padding = 0

        conv_height = (
            input_height + 2 * conv_padding - kernel_height
        ) // conv_stride + 1

        conv_width = (input_width + 2 * conv_padding - kernel_width) // conv_stride + 1

        if conv_height <= 0 or conv_width <= 0:
            raise ValueError(
                "Input dimensions are too small " "for the convolution kernel"
            )

        # --------------------------------------------------------
        # Pooling configuration
        # --------------------------------------------------------

        pool_size = 2
        pool_stride = 2

        pool_height = (conv_height - pool_size) // pool_stride + 1

        pool_width = (conv_width - pool_size) // pool_stride + 1

        if pool_height <= 0 or pool_width <= 0:
            raise ValueError(
                "Convolution output is too small " "for the pooling window"
            )

        # --------------------------------------------------------
        # Flatten size
        # --------------------------------------------------------

        flatten_size = pool_height * pool_width * num_filters

        # --------------------------------------------------------
        # Dense parameters
        # --------------------------------------------------------

        num_classes = 10

        self.parameters["W2"] = (
            np.random.randn(
                num_classes,
                flatten_size,
            )
            * 0.01
        )

        self.parameters["b2"] = np.zeros((num_classes, 1))

    def _forward(self, X):
        """Run the CNN forward pass and return probabilities plus a cache.

        ``X`` has shape ``(H, W, C)`` and the returned probabilities have
        shape ``(10, 1)``. The cache stores intermediate activations needed by
        ``_backward``: convolution output, ReLU output, pooled activations,
        flattened features, dense logits, and softmax probabilities.
        """

        # --------------------------------------------------------
        # Get parameters
        # --------------------------------------------------------

        W1 = self.parameters["W1"]
        b1 = self.parameters["b1"]

        W2 = self.parameters["W2"]
        b2 = self.parameters["b2"]

        # --------------------------------------------------------
        # Convolution
        # --------------------------------------------------------

        Z1 = convolve_multifilter(
            X,
            W1,
            b1,
            stride=1,
            padding=0,
        )

        # --------------------------------------------------------
        # ReLU
        # --------------------------------------------------------

        A1 = relu(Z1)

        # --------------------------------------------------------
        # Max Pooling
        # --------------------------------------------------------

        A_pool = max_pool_multichannel(
            A1,
            pool_size=2,
            stride=2,
            padding=0,
        )

        # --------------------------------------------------------
        # Flatten
        # --------------------------------------------------------

        A2_flat = flatten(A_pool)

        X_dense = A2_flat.reshape(-1, 1)

        # --------------------------------------------------------
        # Dense
        # --------------------------------------------------------

        Z2 = dense_forward(
            X_dense,
            W2,
            b2,
        )

        # --------------------------------------------------------
        # Softmax
        # --------------------------------------------------------

        A2 = softmax(Z2)

        # --------------------------------------------------------
        # Cache
        # --------------------------------------------------------

        cache = {
            "X": X,
            "Z1": Z1,
            "A1": A1,
            "A_pool": A_pool,
            "A_flat": A2_flat,
            "X_dense": X_dense,
            "Z2": Z2,
            "A2": A2,
        }

        return A2, cache

    def _backward(self, Y, cache):
        """Backpropagate the loss through the cached forward pass.

        ``Y`` has shape ``(10, 1)``. The returned gradient dictionary contains
        ``dW1``, ``db1``, ``dW2``, and ``db2`` matching the parameter shapes.
        The softmax and categorical cross-entropy derivative is simplified to
        ``dZ = (A - Y) / m`` before passing through the dense layer.
        """

        # --------------------------------------------------------
        # Get cached values
        # --------------------------------------------------------

        X = cache["X"]
        Z1 = cache["Z1"]
        A1 = cache["A1"]
        A_pool = cache["A_pool"]
        X_dense = cache["X_dense"]
        A2 = cache["A2"]

        # --------------------------------------------------------
        # Get parameters
        # --------------------------------------------------------

        W1 = self.parameters["W1"]
        W2 = self.parameters["W2"]

        # --------------------------------------------------------
        # Softmax + Categorical Cross Entropy
        # --------------------------------------------------------

        dZ2 = categorical_cross_entropy_backward(
            Y,
            A2,
        )

        # --------------------------------------------------------
        # Dense backward
        # --------------------------------------------------------

        dW2, db2, dX_dense = dense_backward(
            dZ2,
            X_dense,
            W2,
        )

        # --------------------------------------------------------
        # Flatten backward
        # --------------------------------------------------------

        dA_pool = flatten_backward(
            dX_dense.reshape(-1),
            A_pool.shape,
        )

        # --------------------------------------------------------
        # Max Pool backward
        # --------------------------------------------------------

        dA1 = max_pool_multichannel_backward(
            dA_pool,
            A1,
            pool_size=2,
            stride=2,
            padding=0,
        )

        # --------------------------------------------------------
        # ReLU backward
        # --------------------------------------------------------

        dZ1 = relu_backward(
            dA1,
            Z1,
        )

        # --------------------------------------------------------
        # Convolution backward
        # --------------------------------------------------------

        dX, dW1, db1 = convolve_multifilter_backward(
            dZ1,
            X,
            W1,
            stride=1,
            padding=0,
        )

        # --------------------------------------------------------
        # Store gradients
        # --------------------------------------------------------

        grads = {
            "dW1": dW1,
            "db1": db1,
            "dW2": dW2,
            "db2": db2,
        }

        return grads

    def _update_parameters(self, grads):
        """Apply the optimizer to the current parameters and gradients."""
        self.parameters = self.optimizer.update(
            self.parameters,
            grads,
        )

    def fit(self, X, Y):
        """Train on one-hot labels using sample-by-sample SGD.

        ``X`` has shape ``(m, H, W, C)`` and ``Y`` has shape
        ``(10, m)``. Parameters are initialized from the first image, then
        each example is processed with forward propagation, cross-entropy,
        backpropagation, and an immediate gradient-descent update. The method
        records one mean epoch loss per epoch and returns ``self``.
        """

        if X.ndim != 4:
                raise ValueError(
                "X must be a 4D array with shape (m, height, width, channels)"
            )

        if Y.ndim != 2:
            raise ValueError("Y must be a 2D array with shape (num_classes, m)")

        if X.shape[0] != Y.shape[1]:
            raise ValueError(
                "Number of samples in X must match number of examples in Y"
            )

        # Initialize parameters using one input image
        self._initialize_parameters(X[0].shape)

        self.loss_history = []

        m = X.shape[0]

        for epoch in range(self.epochs):

            epoch_loss = 0.0

            for i in range(m):

                # ------------------------------------------------
                # Select one training example
                # ------------------------------------------------

                image = X[i]
                y = Y[:, i : i + 1]

                # ------------------------------------------------
                # Forward pass
                # ------------------------------------------------

                probabilities, cache = self._forward(image)

                # ------------------------------------------------
                # Loss
                # ------------------------------------------------

                loss = categorical_cross_entropy(
                    y,
                    probabilities,
                )

                epoch_loss += loss

                # ------------------------------------------------
                # Backward pass
                # ------------------------------------------------

                grads = self._backward(
                    y,
                    cache,
                )

                # ------------------------------------------------
                # Update parameters
                # ------------------------------------------------

                self._update_parameters(grads)

            epoch_loss /= m

            self.loss_history.append(epoch_loss)

            print(f"Epoch {epoch + 1}/{self.epochs} " f"- Loss: {epoch_loss:.6f}")

        return self

    def predict(self, X):
        """Return the predicted class index for each image in ``X``."""
        if X.ndim != 4:
            raise ValueError(
                "X must be a 4D array with shape (m, height, width, channels)"
            )

        predictions = []

        for i in range(X.shape[0]):

            probabilities, _ = self._forward(X[i])

            predicted_class = np.argmax(probabilities, axis=0)[0]

            predictions.append(predicted_class)

        return np.array(predictions)

    def evaluate(self, X, Y):
        """Return classification accuracy for images ``X`` and one-hot ``Y``."""
        predictions = self.predict(X)

        true_labels = np.argmax(Y, axis=0)

        accuracy = np.mean(predictions == true_labels)

        return accuracy
