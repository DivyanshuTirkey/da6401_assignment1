import numpy as np

class MSE:
    def __init__(self):
        pass

    def forward(self, x, y):
        self.x = x
        self.y = y

        self.val = np.mean((y - x)**2)
        return self.val

    def back(self):
        batch_size = self.x.shape[0]
        return -2 * (self.y - self.x) / batch_size
