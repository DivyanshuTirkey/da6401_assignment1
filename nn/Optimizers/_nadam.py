import numpy as np
from .. import Model

class Nadam:
    def __init__(self, model: Model, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0):
        self.model = model
        self.beta1 = beta1
        self.beta2 = beta2
        self.lr = lr
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0  # Initialize timestep

        self.m = []
        self.v = []
        self.m_bias = []
        self.v_bias = []

        for layer in self.model.layers:
            self.m.append(np.zeros_like(layer.weights) if hasattr(layer, 'weights') else None)
            self.v.append(np.zeros_like(layer.weights) if hasattr(layer, 'weights') else None)
            self.m_bias.append(np.zeros_like(layer.bias) if hasattr(layer, 'bias') else None)
            self.v_bias.append(np.zeros_like(layer.bias) if hasattr(layer, 'bias') else None)

    def step(self):
        self.t += 1  # Increment timestep at the beginning

        for id, layer in enumerate(self.model.layers):
            if hasattr(layer, 'grads') and hasattr(layer, 'weights'):
                # Update biased first moment estimate (momentum)
                self.m[id] = self.beta1 * self.m[id] + (1 - self.beta1) * layer.grads
                # Update biased second raw moment estimate (RMS term)
                self.v[id] = self.beta2 * self.v[id] + (1 - self.beta2) * (layer.grads ** 2)

                # Compute bias-corrected moment estimates
                m_hat = self.m[id] / (1 - self.beta1 ** self.t)
                v_hat = self.v[id] / (1 - self.beta2 ** self.t)

                # Nadam modification: Nesterov accelerated gradient
                m_nadam = self.beta1 * m_hat + (1 - self.beta1) * layer.grads / (1 - self.beta1 ** self.t)

                # Update weights with weight decay
                layer.weights -= self.lr * (m_nadam / (np.sqrt(v_hat) + self.eps) + self.weight_decay * layer.weights)

            if hasattr(layer, 'grads_bias') and hasattr(layer, 'bias'):
                # Update biased first moment estimate for bias
                self.m_bias[id] = self.beta1 * self.m_bias[id] + (1 - self.beta1) * layer.grads_bias
                # Update biased second raw moment estimate for bias
                self.v_bias[id] = self.beta2 * self.v_bias[id] + (1 - self.beta2) * (layer.grads_bias ** 2)

                # Compute bias-corrected moment estimates
                m_hat_bias = self.m_bias[id] / (1 - self.beta1 ** self.t)
                v_hat_bias = self.v_bias[id] / (1 - self.beta2 ** self.t)

                # Nadam modification for bias
                m_nadam_bias = self.beta1 * m_hat_bias + (1 - self.beta1) * layer.grads_bias / (1 - self.beta1 ** self.t)

                # Update bias (No weight decay on biases)
                layer.bias -= self.lr * m_nadam_bias / (np.sqrt(v_hat_bias) + self.eps)
