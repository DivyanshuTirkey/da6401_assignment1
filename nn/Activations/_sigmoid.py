import numpy as np
class Sigmoid:
    def __init__(self):
        pass

    def forward(self, x):
        x = np.array(x)
        return 1 / (1 + np.exp(-x))
    
    def back(self, x):
        x = np.array(x)
        return self.forward(x)*(1 - self.forward(x))
