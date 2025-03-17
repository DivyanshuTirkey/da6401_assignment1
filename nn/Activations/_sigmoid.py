import numpy as np

class Sigmoid:
    def __init__(self, clip_value=500):
        # clip_value is the maximum value for x before applying exp, to prevent overflow
        self.clip_value = clip_value

    def forward(self, x):
        self.x = x
        # Clip the values to the range [-clip_value, clip_value]
        clipped_x = np.clip(x, -self.clip_value, self.clip_value)
        
        self.sigmoid = 1 / (1 + np.exp(-clipped_x))
        return self.sigmoid
    
    def back(self, delta):
        return delta * self.sigmoid * (1 - self.sigmoid)