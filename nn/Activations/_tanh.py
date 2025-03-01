import numpy as np

class Tanh:
    def __init__(self):
        pass

    def forward(self, x):
        x = np.array(x)
        return np.tanh(x)
    
    def back(self, x):
        x = np.array(x)
        return 1 - np.tanh(x)**2