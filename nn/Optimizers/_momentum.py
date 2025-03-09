import numpy as np
from .. import Model


class MomentumSGD:
    def __init__(self, model: Model, lr=0.1, beta=0.5):
        self.model = model
        self.beta = beta
        self.lr = lr

        self.u = []
        self.u_bias = []
        for layer in self.model.layers:
            if hasattr(layer, 'weights'):
                self.u.append(np.zeros_like(layer.weights))
            else:
                self.u.append(None)

            if hasattr(layer, 'bias'):
                self.u_bias.append(np.zeros_like(layer.bias))
            else:
                self.u_bias.append(None)
            

    def step(self):
        for id, layer in enumerate(self.model.layers):
            if hasattr(layer, 'weights'):

                self.u[id] = self.beta * self.u[id] + (1 - self.beta) * layer.grads
                layer.weights -= self.lr * self.u[id]

            if hasattr(layer, 'grads_bias'):
                
                self.u_bias[id] = self.beta * self.u_bias[id] + (1 - self.beta) * layer.grads_bias
                layer.bias -= self.lr * self.u_bias[id]
