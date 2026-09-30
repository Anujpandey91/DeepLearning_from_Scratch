import numpy as np

from modules.module_02_deep_cnn_models.implementations.sequential import Sequential
from modules.module_02_deep_cnn_models.implementations.layers import Dense, ReLU

def create_test_model():
    return Sequential(
        [
            Dense(5),
            ReLU(),
            Dense(3),
        ]
    )


def test_sequential_forward():
    model = create_test_model()

    X = np.random.randn(4, 10)

    output = model.forward(X)

    assert output.shape == (4, 3)


def test_sequential_backward():
    model = create_test_model()

    X = np.random.randn(4, 10)

    output = model.forward(X)

    dY = np.random.randn(*output.shape)

    dX = model.backward(dY)

    assert dX.shape == X.shape


def test_sequential_parameters():
    model = create_test_model()

    X = np.random.randn(4, 10)

    model.forward(X)

    parameters = model.parameters()

    assert 0 in parameters
    assert 2 in parameters

    assert "W" in parameters[0]
    assert "b" in parameters[0]

    assert "W" in parameters[2]
    assert "b" in parameters[2]

    assert parameters[0]["W"].shape == (10, 5)
    assert parameters[0]["b"].shape == (5,)

    assert parameters[2]["W"].shape == (5, 3)
    assert parameters[2]["b"].shape == (3,)


def test_sequential_gradients():
    model = create_test_model()

    X = np.random.randn(4, 10)

    output = model.forward(X)

    dY = np.random.randn(*output.shape)

    model.backward(dY)

    gradients = model.gradients()

    assert 0 in gradients
    assert 2 in gradients

    assert "dW" in gradients[0]
    assert "db" in gradients[0]

    assert "dW" in gradients[2]
    assert "db" in gradients[2]

    assert gradients[0]["dW"].shape == (10, 5)
    assert gradients[0]["db"].shape == (5,)

    assert gradients[2]["dW"].shape == (5, 3)
    assert gradients[2]["db"].shape == (3,)
