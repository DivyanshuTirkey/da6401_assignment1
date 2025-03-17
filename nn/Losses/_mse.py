import numpy as np

class MSE:
    def __init__(self):
        pass

    def forward(self, x, y):
        self.x = np.clip(x, -1e10, 1e10)  # Clip to prevent overflow
        one_hot_labels = np.zeros_like(x)
        one_hot_labels[np.arange(len(y)), y] = 1
        self.y = one_hot_labels

        self.val = np.mean((self.y - self.x) ** 2) / 2
        return self.val

    def back(self):
        batch_size = self.x.shape[0]
        return -(self.y - self.x) / batch_size