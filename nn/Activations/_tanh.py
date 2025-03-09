import numpy as np

class Tanh:
    def __init__(self):
        pass

    def forward(self, x):
        self.x = x
        self.tanh_out = np.tanh(self.x)
        return self.tanh_out
    
    def back(self, delta):
        return delta * (1 - self.tanh_out**2)