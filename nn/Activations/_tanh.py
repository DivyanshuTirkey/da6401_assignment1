import numpy as np

class Tanh:
    def __init__(self):
        pass

    def forward(self, x):
        x = np.array(x)
        return np.tanh(x)
    
    def back(self, delta):
        delta = np.array(delta)
        self.grads = 1 - np.tanh(delta)**2
        return self.grads