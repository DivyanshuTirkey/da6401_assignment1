import numpy as np
from .. import Model


class NAG:
    def __init__(self, model: Model, lr=0.1, beta=0.5):
        self.model = model
        self.beta = beta
        self.lr = lr

        self.weights = []
        self.bias = []
        
        self.u = []
        self.u_bias = []

        for layer in self.model.layers:
            self.u.append(np.zeros_like(layer.weights) if hasattr(layer, 'weights') else None)
            self.weights.append(np.zeros_like(layer.weights) if hasattr(layer, 'weights') else None)

            self.u_bias.append(np.zeros_like(layer.bias) if hasattr(layer, 'bias') else None)
            self.bias.append(np.zeros_like(layer.bias) if hasattr(layer, 'bias') else None)

    def pre_step(self):
        for id, layer in enumerate(self.model.layers):
            if hasattr(layer, 'weights'):
                self.weights[id] = layer.weights.copy()
                layer.weights -= self.beta * self.u[id]

            if hasattr(layer, "bias"):
                self.bias[id] = layer.bias.copy()
                layer.bias -= self.beta * self.u_bias[id]

    def step(self):
        for id, layer in enumerate(self.model.layers):
            if hasattr(layer, 'weights'):

                self.u[id] = self.beta * self.u[id] + (1 - self.beta) * layer.grads
                layer.weights = self.weights[id] - self.u[id] * self.lr

            if hasattr(layer, 'grads_bias'):

                self.u_bias[id] = self.beta * self.u_bias[id] + (1 - self.beta) * layer.grads_bias
                layer.bias = self.bias[id] - self.u_bias[id] * self.lr

