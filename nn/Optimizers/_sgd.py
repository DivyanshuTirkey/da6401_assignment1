import numpy as np
from .. import Model


class SGD:
    def __init__(self, model: Model, lr=0.1):
        self.model = model
        self.lr = lr

    def step(self):
        for layer in self.model.layers:
            if hasattr(layer, 'weights'):
                layer.weights -= layer.grads * self.lr
            if hasattr(layer, 'grads_bias'):
                    layer.bias -= layer.grads_bias * self.lr

        self.model.update_params()
    