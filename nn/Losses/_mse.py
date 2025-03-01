import numpy as np

class MSE:
    def __init__(self):
        pass

    def forward(self, x, y):
        return np.mean((y - x)**2)

    def back(self, x, y):
        batch_size = x.shape[0]
        return -2 * (y-x) / batch_size
