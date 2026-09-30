class Layer:
    def forward(self, X):
        raise NotImplementedError

    def backward(self, dY):
        raise NotImplementedError

    def parameters(self):
        return {}

    def gradients(self):
        return {}