import numpy as np


class Model:
    def __init__(self, layers: list):
        self.layers = layers
        self.params = []

    def _collect_params(self):
        for layer in self.layers:
            if hasattr(layer, 'weights'):
                self.params.append((layer.weights, layer.bias if hasattr(layer, 'bias') else None))
            else:
                self.params.append(None)
    
    def update_params(self):
        for id in range(len(self.params)):
            if self.params[id] is not None:
                self.params[id] = self.layers[id].weights

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, loss):
        delta = loss.back()
        for layer in reversed(self.layers):
            delta = layer.back(delta)


