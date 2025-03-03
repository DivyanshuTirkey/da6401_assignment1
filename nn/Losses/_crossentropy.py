import numpy as np

from ..Activations import Softmax

class CrossEntropy:
    def __init__(self):
        self.softmax = Softmax()

    def forward(self, x, y):
        self.x = x
        self.y = y

        softmax_vals = self.softmax.forward(x)
        epsilon = 1e-10
        self.softmax_vals = np.clip(softmax_vals, epsilon, 1 - epsilon)

        one_hot_labels = np.zeros_like(softmax_vals)
        one_hot_labels[np.arange(len(y)), y] = 1

        self.val = - np.sum(one_hot_labels * np.log(softmax_vals))
        
        return self.val

    def back(self):
        one_hot_labels = np.zeros_like(self.softmax_vals)
        one_hot_labels[np.arange(len(self.y)), self.y] = 1

        return self.softmax_vals - one_hot_labels