import numpy as np
from .. import Model


class RMSProp:
    def __init__(self, model: Model, lr=0.01, beta=0.5, weight_decay=0.0, eps=1e-8):
        self.model = model
        self.lr = lr
        self.beta = beta
        self.eps = eps
        self.weight_decay = weight_decay

        self.v = []
        self.v_bias = []

        for layer in self.model.layers:
            self.v.append(np.zeros_like(layer.weights) if hasattr(layer, 'weights') else None)
            self.v_bias.append(np.zeros_like(layer.bias) if hasattr(layer, 'bias') else None)

    def step(self):
        for id, layer in enumerate(self.model.layers):
            if hasattr(layer, 'grads') and hasattr(layer, 'weights'):

                grad_w_reg = layer.grads + self.weight_decay * layer.weights

                self.v[id] = self.beta * self.v[id] + (1 - self.beta) * grad_w_reg**2
                layer.weights -= self.lr / np.sqrt(self.v[id] + self.eps) * grad_w_reg

            if hasattr(layer, 'grads_bias') and hasattr(layer, 'bias'):
                
                self.v_bias[id] = self.beta * self.v_bias[id] + (1 - self.beta) * layer.grads_bias**2
                layer.bias -= self.lr / np.sqrt(self.v_bias[id] + self.eps) * layer.grads_bias
        self.model.update_params()