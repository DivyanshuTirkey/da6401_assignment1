import numpy as np

class Identity:
    def __init__(self):
        pass

    def forward(self, x):
        x = np.array(x)
        return x
    
    def back(self, x):
        x = np.array(x)
        return np.ones(x.shape)