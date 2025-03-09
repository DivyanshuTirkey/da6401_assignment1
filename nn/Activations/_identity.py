import numpy as np

class Identity:
    def __init__(self):
        pass

    def forward(self, x):
        return np.array(x)
    
    def back(self, delta):
        return delta