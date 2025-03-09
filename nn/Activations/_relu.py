import numpy as np

class ReLU:
    def __init__(self):
        pass

    def forward(self, x):
        x = np.array(x)
        self.x = x

        return np.maximum(0, x)
    
    def back(self, delta):
        mask = self.x > 0
        self.grads = delta * mask
        return self.grads

        
