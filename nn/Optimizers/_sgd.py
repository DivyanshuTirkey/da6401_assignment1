import numpy as np
from .. import Model


class SGD:
    def __init__(self, model: Model, lr=0.1, weight_decay=0.0):
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay

    def step(self):
        for layer in self.model.layers:
            if hasattr(layer, 'weights'):
                grad_w_reg = layer.grads + self.weight_decay * layer.weights
                layer.weights -= self.lr * grad_w_reg

            if hasattr(layer, 'grads_bias'):
                layer.bias -= self.lr * layer.grads_bias
        self.model.update_params()