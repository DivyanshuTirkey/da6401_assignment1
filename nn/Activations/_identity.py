import numpy as np

class Identity:
    def __init__(self):
        pass

    def forward(self, x):
        x = np.array(x)
        return x
    
    def back(self, delta):
        delta = np.array(delta)
        return np.ones(delta.shape)