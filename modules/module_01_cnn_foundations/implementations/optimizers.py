class GradientDescent:
    """Vanilla gradient descent used by the educational CNN."""

    def __init__(self, learning_rate=0.01):
        if learning_rate <= 0:
            raise ValueError("learning_rate must be greater than zero")

        self.learning_rate = learning_rate

    def update(self, parameters, grads):
        """Update each ``W_l`` and ``b_l`` using its matching gradient."""

        L = len(parameters) // 2

        for l in range(1, L + 1):

            parameters["W" + str(l)] -= (
                self.learning_rate * grads["dW" + str(l)]
            )

            parameters["b" + str(l)] -= (
                self.learning_rate * grads["db" + str(l)]
            )

        return parameters