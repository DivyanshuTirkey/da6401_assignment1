import numpy as np


class Model:
    def __init__(self, layers: list, initializer=None):
        self.layers = layers
        self.params = {'weights': [], 'bias': []}
        self.initializer = initializer

        self._initialize()
        self._collect_params()

    def _initialize_layer(self, layer):
        if self.initializer == 'random':
            layer.weights = np.random.rand(*layer.weights.shape) * 2 - 1

        elif self.initializer == 'Xavier':
            layer.weights = np.random.normal(0, np.sqrt(2 / (layer.in_neurons + layer.out_neurons)), layer.weights.shape)
    
    def _initialize(self):
        for layer in self.layers:
            if hasattr(layer, 'weights') and layer.weights is not None:
                self._initialize_layer(layer)

    def _collect_params(self):
        for layer in self.layers:
            self.params['weights'].append(layer.weights.view() if hasattr(layer, 'weights') else np.nan)
            self.params['bias'].append(layer.bias.view() if hasattr(layer, 'bias') else np.nan)
    

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, loss):
        delta = loss.back()
        for layer in reversed(self.layers):
            delta = layer.back(delta)

    def update_params(self):
        for id, layer in enumerate(self.layers):
            self.params['weights'][id] = layer.weights.view() if hasattr(layer, 'weights') else np.nan
            self.params['bias'][id] = layer.bias.view() if hasattr(layer, 'bias') else np.nan
    
    def load_params(self, weights):
        for id, layer in enumerate(self.layers):
            if hasattr(layer, 'weights') and weights['weights'][id] != np.nan and layer.weights.shape == weights['weights'][id].shape:
                layer.weights = weights['weights'][id]
            if hasattr(layer, 'bias') and weights['bias'][id] != np.nan and layer.bias.shape == weights['bias'][id].shape:
                layer.bias = weights['bias'][id]

        self._collect_params()