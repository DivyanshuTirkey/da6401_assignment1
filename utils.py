import wandb
import numpy as np
from keras.datasets import fashion_mnist, mnist

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

def accuracy(y_pred, y_true):
    # Convert logits to predicted class indices
    y_pred_labels = np.argmax(y_pred, axis=1)

    # If y_true is one-hot encoded, convert to class indices
    if len(y_true.shape) > 1:
        y_true = np.argmax(y_true, axis=1)

    return np.mean(y_pred_labels == y_true)

def display_images(x_train, y_train, class_names):
    selected_images = {}
    for img, label in zip(x_train, y_train):
        if label not in selected_images:
            selected_images[label] = img
        if len(selected_images) == 10:
            break
    
    wandb.log({
        "Examples": [wandb.Image(selected_images[label], caption=class_names[label]) for label in sorted(selected_images.keys())]
    })
