import numpy as np

from .base import Layer
from modules.module_01_cnn_foundations.implementations.convolution import (
    convolve_multifilter,
    convolve_multifilter_backward,
)
from modules.module_01_cnn_foundations.implementations.pooling import (
    max_pool_multichannel,
    max_pool_multichannel_backward,
    average_pool_multichannel,
    average_pool_multichannel_backward,
)


class ReLU(Layer):

    def forward(self, X):
        self.X = X
        return np.maximum(0, X)

    def backward(self, dY):
        return dY * (self.X > 0)


class Conv2D(Layer):

    def __init__(self, filters, kernel_size, padding=1, stride=1):
        self.filters = filters
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride

        self.W = None
        self.b = None
        self.parameters = {}
        self.gradients = {}

    def _initialize_parameters(self, input_channels):
        self.W = (
            np.random.randn(
                self.kernel_size, self.kernel_size, input_channels, self.filters
            )
            * 0.01
        )

        self.b = np.zeros(self.filters)

        self.parameters["W"] = self.W
        self.parameters["b"] = self.b

    def forward(self, X):
        if self.W is None:
            input_channels = X.shape[-1]
            self._initialize_parameters(input_channels)

        self.X = X

        return convolve_multifilter(
            image=X,
            filters=self.W,
            bias=self.b,
            stride=self.stride,
            padding=self.padding,
        )

    def backward(self, dY):
        dX, dW, db = convolve_multifilter_backward(
            dZ=dY,
            image=self.X,
            filters=self.W,
            stride=self.stride,
            padding=self.padding,
        )

        self.gradients["dW"] = dW
        self.gradients["db"] = db

        return dX


class MaxPool2D(Layer):

    def __init__(self, pool_size=2, stride=2, padding=0):
        self.pool_size = pool_size
        self.stride = stride
        self.padding = padding

        self.parameters = {}
        self.gradients = {}

    def forward(self, X):
        self.X = X

        return max_pool_multichannel(
            image=X, pool_size=self.pool_size, stride=self.stride, padding=self.padding
        )

    def backward(self, dY):
        return max_pool_multichannel_backward(
            dA=dY,
            input=self.X,
            pool_size=self.pool_size,
            stride=self.stride,
            padding=self.padding,
        )


class AveragePool2D(Layer):
    def __init__(self, pool_size=2, stride=2, padding=0):
        self.pool_size = pool_size
        self.stride = stride
        self.padding = padding

        self.parameters = {}
        self.gradients = {}

    def forward(self, X):
        self.X = X

        return average_pool_multichannel(
            image=X, pool_size=self.pool_size, stride=self.stride, padding=self.padding
        )

    def backward(self, dY):
        return average_pool_multichannel_backward(
            dA=dY,
            input=self.X,
            pool_size=self.pool_size,
            stride=self.stride,
            padding=self.padding,
        )


class Flatten(Layer):

    def __init__(self):
        self.parameters = {}
        self.gradients = {}

    def forward(self, X):
        self.input_shape = X.shape
        if X.ndim == 4:
            X = X.reshape(X.shape[0], -1)            
        elif X.ndim == 3:
            X = X.reshape(-1)
        else:
            raise ValueError("The input should be of 3D or 4D")

        return X
        
    def backward(self, dY):
        return dY.reshape(self.input_shape)

class Dense(Layer):

    def __init__(self, units):
        self.units = units
        self.W = None
        self.b = None
        self.parameters = {}
        self.gradients = {}

    def forward(self, X):
        self.input_features = X.shape[-1]

        if self.W is None:
            self.W = np.random.randn(self.input_features, self.units) * np.sqrt(
                2.0 / self.input_features
            )

            self.b = np.zeros(self.units)

            self.parameters["W"] = self.W
            self.parameters["b"] = self.b

        self.X = X

        return np.dot(X, self.W) + self.b

    def backward(self, dZ):
        dW = np.dot(self.X.T , dZ)
        dX = np.dot(dZ, self.W.T)

        if dZ.ndim == 1:
            db = dZ
        elif dZ.ndim == 2:
            db = np.sum(dZ, axis=0)
        else:
            raise ValueError("dZ should be 1D or 2D")

        self.gradients["dW"] = dW
        self.gradients["db"] = db

        return dX


class Softmax(Layer):

    def __init__(self):
        self.parameters = {}
        self.gradients = {}

    def forward(self, X):
        if X.ndim == 1:
            X_shifted = X - np.max(X)
            exp_X = np.exp(X_shifted)
            return exp_X / np.sum(exp_X)

        elif X.ndim == 2:
            X_shifted = X - np.max(X, axis=1, keepdims=True)
            exp_X = np.exp(X_shifted)
            return exp_X / np.sum(exp_X, axis=1, keepdims=True)

        else:
            raise ValueError("Softmax input should be 1D or 2D")
