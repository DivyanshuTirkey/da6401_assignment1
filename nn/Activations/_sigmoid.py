import numpy as np

class Sigmoid:
    def __init__(self):
        pass

    def forward(self, x):
        self.x = x
        self.sigmoid = np.where(
            x >= 0,
            1 / (1 + np.exp(-x)),
            np.exp(x) / (1 + np.exp(x))
        )
        return self.sigmoid
    
    def back(self, delta):
        return delta * self.sigmoid * (1 - self.sigmoid)