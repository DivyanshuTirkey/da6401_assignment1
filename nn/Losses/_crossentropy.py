import numpy as np

from ..Activations import Softmax

class CrossEntropy:
    def __init__(self):
        self.softmax = Softmax()

    def forward(self, x, y):
        softmax_vals = self.softmax.forward(x)
        epsilon = 1e-10
        softmax_vals = np.clip(softmax_vals, epsilon, 1 - epsilon)

        one_hot_labels = np.zeros_like(softmax_vals)
        
        one_hot_labels[np.arange(len(y)), y] = 1

        loss = - np.sum(one_hot_labels * np.log(softmax_vals))
        return loss

    def back(self, x, y):
        softmax_vals = self.softmax.forward(x)
        epsilon = 1e-10
        softmax_vals = np.clip(softmax_vals, epsilon, 1 - epsilon)

        one_hot_labels = np.zeros_like(softmax_vals)
        one_hot_labels[np.arange(len(y)), y] = 1

        return softmax_vals - one_hot_labels