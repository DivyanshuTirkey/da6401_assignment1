import numpy as np
from .. import Model


class NAG:
    def __init__(self, model: Model, loss, lr=0.1, beta=0.5):
        self.model = model
        self.loss = loss
        self.beta = beta
        self.lr = lr

        self.weights = []
        self.bias = []
        
        self.u = []
        self.u_bias = []

        for layer in self.model.layers:
            if hasattr(layer, 'weights'):
                self.u.append(np.zeros_like(layer.weights))
                self.weights.append(layer.weights)
            else:
                self.u.append(None)
                self.weights.append(None)

            if hasattr(layer, 'bias'):
                self.u_bias.append(np.zeros_like(layer.bias))
                self.bias.append(layer.bias)
            else:
                self.u_bias.append(None)
                self.bias.append(None)

    def step(self, x, y):

        for id, layer in enumerate(self.model.layers):
            if hasattr(layer, 'weights'):
                self.weights[id] = layer.weights
                layer.weights -= self.beta * self.u[id]
            if hasattr(layer, "bias"):
                self.bias[id] = layer.bias
                layer.bias -= self.beta * self.u_bias[id]
        
        loss = self.loss.forward(self.model.forward(x), y)
        self.model.backward(self.loss)

        for id, layer in enumerate(self.model.layers):
            if hasattr(layer, 'weights'):
                self.u[id] = self.beta * self.u[id] + layer.grads
                layer.weights = self.weights[id] - self.u[id] * self.lr

            if hasattr(layer, 'grads_bias'):
                self.u_bias[id] = self.beta * self.u_bias[id] + layer.grads_bias
                layer.bias = self.bias[id] - self.u_bias[id] * self.lr

        self.model.update_params()

