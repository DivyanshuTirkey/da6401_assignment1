import numpy as np
from .. import Model


class Adam:
    def __init__(self, model: Model, lr=0.1, beta1=0.5, beta2=0.5, eps=1e-8, weight_decay=0.0):
        self.eps = eps
        self.model = model
        self.beta1 = beta1
        self.beta2 = beta2
        self.lr = lr
        self.weight_decay = weight_decay
        self.t = 0

        self.m = []
        self.m_bias = []

        self.v = []
        self.v_bias = []

        for layer in self.model.layers:
            self.m.append(np.zeros_like(layer.weights) if hasattr(layer, 'weights') else None)
            self.v.append(np.zeros_like(layer.weights) if hasattr(layer, 'weights') else None)

            self.m_bias.append(np.zeros_like(layer.bias) if hasattr(layer, 'bias') else None)
            self.v_bias.append(np.zeros_like(layer.bias) if hasattr(layer, 'bias') else None)
            

    def step(self):
        self.t += 1
        for id, layer in enumerate(self.model.layers):
            if hasattr(layer, 'grads') and hasattr(layer, 'weights'):

                # Compute biased first moment estimate
                self.m[id] = self.beta1 * self.m[id] + (1 - self.beta1) * layer.grads
                self.v[id] = self.beta2 * self.v[id] + (1 - self.beta2) * layer.grads**2

                # Bias-corrected moment estimates
                m_ = self.m[id] / (1 - self.beta1 ** self.t)
                v_ = self.v[id] / (1 - self.beta2 ** self.t)

                # Apply weight decay (L2 regularization)
                layer.weights -= self.lr * (m_ / (np.sqrt(v_) + self.eps) + self.weight_decay * layer.weights)

            if hasattr(layer, 'grads_bias') and hasattr(layer, 'bias'):
                
                self.m_bias[id] = self.beta1 * self.m_bias[id] + (1 - self.beta1) * layer.grads_bias
                self.v_bias[id] = self.beta2 * self.v_bias[id] + (1 - self.beta2) * layer.grads_bias**2

                m_bias_  = self.m_bias[id] / (1 - self.beta1 ** self.t)
                v_bias_  = self.v_bias[id] / (1 - self.beta2 ** self.t)

                layer.bias -= self.lr * m_bias_  / (np.sqrt(v_bias_) + self.eps)

        self.model.update_params()
