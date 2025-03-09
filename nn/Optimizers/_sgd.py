import numpy as np
from .. import Model


class SGD:
    def __init__(self, model: Model, lr=0.1):
        self.model = model
        self.lr = lr

    def step(self):
        for layer in self.model.layers:
            if hasattr(layer, 'weights'):
                layer.weights -= self.lr * layer.grads 

            if hasattr(layer, 'grads_bias'):
                layer.bias -= self.lr * layer.grads_bias
    