import numpy as np

class Softmax:
    def __init__(self):
        pass

    def forward(self, x):
        x = np.array(x)
        shifted_out = x - np.max(x, axis=1, keepdims=True)
        exp_out = np.exp(shifted_out)
        self.softmax_out = exp_out / np.sum(exp_out, axis=1, keepdims=True)

        return self.softmax_out

    def back(self, delta):
        batch_size, nclasses = delta.shape
        grad_input = np.zeros_like(delta)

        for i in range(batch_size):
            softmax_vector = self.softmax_out[i].reshape(-1, 1) 
            jacobian = np.diagflat(softmax_vector) - np.dot(softmax_vector, softmax_vector.T)  
            
            grad_input[i] = np.dot(jacobian, delta[i])

        return grad_input



