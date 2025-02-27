import numpy as np

class Linear:
    def __init__(self, in_neurons, out_neurons, bias=True):
        self.in_neurons = in_neurons
        self.out_neurons = out_neurons

        self.weights = np.zeros((in_neurons, out_neurons))
        if bias:
            self.bias = np.ones(out_neurons)
        else:
            self.bias = np.zeros(out_neurons)

    def forward(self, x):
        return self.weights.T @ x + self.bias