import numpy as np


class SGD:
    def __init__(self, parameters, lr=10e-2):
        self.parameters = parameters
        self.lr = lr

    def step(self):
        grads = grads