import numpy as np

class Flatten:
    def __init__(self):
        pass

    def forward(self, x):
        x = np.array(x)
        batch_size = x.shape[0]
        out_shape = np.prod(x.shape[1:])

        out = x.reshape(batch_size, out_shape)

        return out