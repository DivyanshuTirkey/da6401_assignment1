import numpy as np
from ..Activations import Softmax

class CrossEntropy:
    def __init__(self):
        self.softmax = Softmax()

    def forward(self, x, y):
        self.x = x
        self.y = y

        # Numerically stable softmax calculation
        logits_shifted = x - np.max(x, axis=1, keepdims=True)
        exp_logits = np.exp(logits_shifted)
        self.softmax_vals = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        # Clip to avoid log(0)
        epsilon = 1e-12
        clipped_softmax_vals = np.clip(self.softmax_vals, epsilon, 1. - epsilon)

        correct_logprobs = -np.log(clipped_softmax_vals[np.arange(len(y)), y])
        
        self.val = np.mean(correct_logprobs)
        
        return self.val

    def back(self):
        batch_size = len(self.y)

        grad = self.softmax_vals.copy()
        grad[np.arange(batch_size), self.y] -= 1
        grad /= batch_size
        
        return grad