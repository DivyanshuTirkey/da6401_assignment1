import numpy as np

class Linear:
    def __init__(self, in_neurons, out_neurons, bias=True):
        self.in_neurons = in_neurons
        self.out_neurons = out_neurons

        self.weights = np.zeros((out_neurons, in_neurons))
        if bias:
            self.bias = np.ones(out_neurons)
        else:
            self.bias = np.zeros(out_neurons)

    def forward(self, x):
        self.input = np.clip(x, -1e10, 1e10)
        return self.input @ self.weights.T + self.bias
    
    def back(self, delta):
        self.grads = (delta.T @ self.input) / self.input.shape[0]
        self.grads_bias = np.sum(delta, axis=0) / self.input.shape[0]
        delta = delta @ self.weights

        return delta





