import numpy as np
class Sigmoid:
    def __init__(self):
        pass

    def forward(self, x):
        x = np.array(x)
        return 1 / (1 + np.exp(-x))
    
    def back(self, delta):
        delta = np.array(delta)
        return self.forward(delta)*(1 - self.forward(delta))
