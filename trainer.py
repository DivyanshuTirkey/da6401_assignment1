import numpy as np
from keras.datasets import fashion_mnist, mnist

from nn import Model
from nn.Layers import *
from nn.Activations import *
from nn.Losses import *
from nn.Activations import *
from nn.Optimizers import *


class Trainer:
    def __init__(self, num_layers: int, hidden_size: list[int], activation: str, init: str|None, loss: str, optimizer: str, epochs: int, lr: float|None, batch_size: int, x_train, y_train, x_val, y_val, num_classes, momentum: float|None=None, beta: float|None=None, beta1: float|None=None, beta2: float|None=None, weight_decay: float|None=0.0, eps: float|None=None, class_names = None):

        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.activation = activation
        self.loss = loss
        self.optimizer = optimizer
        self.epochs = epochs
        self.batch_size = batch_size
        self.momentum = momentum
        self.beta = beta
        self.beta1 = beta1
        self.beta2 = beta2
        self.weight_decay = weight_decay
        self.lr = lr
        self.eps = eps
        self.init = init
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val

        self.input_shape = self.x_train[0].shape
        self.num_classes = num_classes
        self.class_names = class_names

        self.activation_layer = self.get_activation()
        self.model = self.create_model()
        self.loss_fn = self.get_loss()
        self.optimizer_fn = self.get_optimizer()

    def get_optimizer(self):
        if self.optimizer == 'sgd':
            return SGD(model=self.model, lr=self.lr)
        if self.optimizer == 'momentum':
            return MomentumSGD(model=self.model, lr=self.lr, beta=self.momentum)
        if self.optimizer == 'nag':
            return NAG(model=self.model, lr=self.lr, beta=self.momentum)
        if self.optimizer == 'rmsprop':
            print(self.beta)
            return RMSProp(model=self.model, lr=self.lr, beta=self.beta, eps=self.eps)
        if self.optimizer == 'adam':
            return Adam(model=self.model, lr=self.lr, beta1=self.beta1, beta2=self.beta, eps=self.eps, weight_decay=self.weight_decay)
        if self.optimizer == 'nadam':
            return Nadam(model=self.model, lr=self.lr, beta1=self.beta1, beta2=self.beta, eps=self.eps, weight_decay=self.weight_decay)

    def get_loss(self):
        return MSE() if self.loss == 'mse' else CrossEntropy()

    def get_activation(self):
        if self.activation == 'ReLU':
            return ReLU
        elif self.activation == 'sigmoid':
            return Sigmoid
        elif self.activation == 'tanh':
            return Tanh
        elif self.activation == 'identity':
            return Identity
        

    def create_model(self):
        layers = [Flatten()]
        out = np.prod(self.input_shape)

        for i in range(self.num_layers):
            layers.append(Linear(out, self.hidden_size[i]))
            layers.append(self.activation_layer())
            out = self.hidden_size[i]

        layers.append(Linear(out, self.num_classes))

        if self.loss == 'mse':
            layers.append(Identity())
        
        model = Model(layers=layers, initializer=self.init)
        return model
    
    def create_batches(self, x, y):
        indices = np.random.permutation(x.shape[0])

        # Generate batches
        for i in range(0, len(indices), self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            yield x[batch_indices], y[batch_indices]
    
    def train(self):
        
        for epoch in range(self.epochs):
            train_itr = self.create_batches(self.x_train, self.y_train)
            val_itr = self.create_batches(self.x_val, self.y_val)

            train_loss = 0
            val_loss = 0
            
            for x_batch, y_batch in train_itr:
                # NAG pre-step to calculate and assign lookahead-weights
                if self.optimizer == 'nag':
                    self.optimizer_fn.pre_step()

                out = self.model.forward(x_batch)
                loss_val = self.loss_fn.forward(out, y_batch)

                self.model.backward(self.loss_fn)
                self.optimizer_fn.step()
                train_loss +=  loss_val
            
            for x_valb, y_valb in val_itr:
                out = self.model.forward(x_valb)
                loss_val = self.loss_fn.forward(out, y_valb)

                val_loss += loss_val

            print(f"EPOCH:{epoch + 1} Train loss:{train_loss} Val loss:{val_loss}")

def split_data(x,y, split=0.1):
    split_size = int(x.shape[0] * split)

    ids = np.random.choice(x.shape[0], size=split_size, replace=False)

    split_x = x[ids]
    split_y = y[ids]
    remaining_x = np.delete(x, ids, axis=0)
    remaining_y = np.delete(y, ids, axis=0)

    return (remaining_x, remaining_y), (split_x, split_y)

def get_dataset(dataset):
    if dataset == 'fashion_mnist':
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
        class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
    elif dataset == 'mnist':
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        class_names = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    return x_train, y_train, x_test, y_test, class_names
    


if __name__ == '__main__':
    x_train, y_train, x_test, y_test, class_names = get_dataset('fashion_mnist')
    (x_train, y_train), (x_val, y_val) = split_data(x_train, y_train)

    trainer = Trainer(3,[40,60,80], 'ReLU', 'random', 'crossentropy', 'momentum', 100, 0.001, 64, x_train, y_train, x_val, y_val, 10, 0.9, 0.9, 0.9, 0.9, 0.2, 1e-8, class_names=class_names)

    trainer.train()