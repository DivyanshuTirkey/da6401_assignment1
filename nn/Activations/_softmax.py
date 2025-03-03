import numpy as np

class Softmax:
    def __init__(self):
        pass

    def forward(self, x):
        x = np.array(x)
        shifted_out = x - np.max(x, axis = 1, keepdims=True)
        exp_out = np.exp(shifted_out)
        exp_out /= np.sum(exp_out, axis=1, keepdims=True)
        return exp_out

    def back(self, delta):
        delta = np.array(delta)

        batch_size, nclasses = delta.shape

        s = self.forward(delta)

        jacobian = np.zeros((batch_size, nclasses, nclasses))

        for id in range(batch_size):
            for i in range(nclasses):
                for j in range(nclasses):
                    if i == j:
                        jacobian[id, i, j] = s[id, i] - s[id, i] * s[id, j]
                    else:
                        jacobian[id, i, j] = - s[id, i] * s[id, j]
        
        return jacobian



