class Sequential:

    def __init__(self, layers):
        self.layers = layers

    def forward(self, X):

        for layer in self.layers:
            X = layer.forward(X)

        return X

    def backward(self, dY):

        for layer in reversed(self.layers):
            dY = layer.backward(dY)

        return dY


    def parameters(self):
        parameters = {}

        for index, layer in enumerate(self.layers):
            if layer.parameters:
                parameters[index] = layer.parameters

        return parameters


    def gradients(self):
        gradients = {}
        
        for index, layer in enumerate(self.layers):
            if layer.gradients:
                gradients[index] = layer.gradients
                
        return gradients
