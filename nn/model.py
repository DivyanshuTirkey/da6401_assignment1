import numpy as np


class Model:
    def __init__(self, layers: list):
        self.layers = layers

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)

        return x

    def backward(self, loss):
        delta = loss.back()
        for layer in reversed(self.layers):
            delta = layer.back(delta)


