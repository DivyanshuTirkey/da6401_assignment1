import numpy as np

class ReLU:
    def __init__(self):
        pass

    def forward(self, x):
        x = np.array(x)

        return np.maximum(0, x)
    
    def back(self, delta):
        delta = np.array(delta)

        return delta > 0

        
