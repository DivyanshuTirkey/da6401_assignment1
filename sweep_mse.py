import wandb
import wandb.agents
import random
import pickle
import os

from nn import Model
from nn.Layers import *
from nn.Activations import *
from nn.Losses import *
from nn.Optimizers import *

from utils import *
from trainer import Trainer

# Define sweep configuration
sweep_config = {
    'name': 'sweep_f_mnist_random_mse',
    'method': 'random',
    'metric': {
        'name': 'val_accuracy',  # Fixed typo in metric name
        'goal': 'maximize'   
    },
    'parameters': {
        'epochs': {
            'values': [5, 10]
        },
        'hidden_layers': {
            'values': [3, 4, 5]
        },
        'weight_decay': {
            'values': [0, 0.0005, 0.5]
        },
        'learning_rate': {
            'values': [1e-3, 1e-4]
        },
        'optimizer': {
            'values': ['sgd', 'momentum', 'nag', 'rmsprop', 'adam', 'nadam']
        },
        'batch_size': {
            'values': [16, 32, 64]
        },
        'weight_initialization': {
            'values': ['random', 'Xavier']
        },
        'activation_function': {
            'values': ['sigmoid', 'tanh', 'ReLU']
        }
    }
}

# Load and prepare the dataset
x_train, y_train, x_test, y_test, class_names = get_dataset('fashion_mnist')
(x_train, y_train), (x_val, y_val) = split_data(x_train, y_train)

best_val_acc = -float('inf')
best_model = None 

def save_best_model(trainer, val_acc):
    global best_val_acc, best_model

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_model = trainer.model

def sweep():

    
    run = wandb.init()
    # Access hyperparameters from wandb.config
    epochs = wandb.config.epochs
    hidden_layers = wandb.config.hidden_layers
    hidden_size = {f'hl_{i+1}':random.sample([32, 64, 128],1)[0] for i in range(hidden_layers)}
    weight_decay = wandb.config.weight_decay
    learning_rate = wandb.config.learning_rate
    optimizer = wandb.config.optimizer
    batch_size = wandb.config.batch_size
    weight_initialization = wandb.config.weight_initialization
    activation_function = wandb.config.activation_function

    wandb.config.update({
        'hidden_size':hidden_size
    })
    name = f'hl_{hidden_layers}_ep_{epochs}_lr_{learning_rate}_opt_{optimizer}_bt_{batch_size}_init_{weight_initialization}_act_{activation_function}'
    
    run.name = name

    hidden_size = [val for val in hidden_size.values()]
    trainer = Trainer(hidden_layers, hidden_size, activation_function, weight_initialization, 'mse', optimizer, epochs, learning_rate, batch_size, x_train, y_train, x_val, y_val, 10, 0.9, 0.9, 0.9, 0.9, weight_decay, 1e-8, class_names)


    history = trainer.train()

    for result in history:
        epoch, train_loss, val_loss, train_acc, val_acc = result
        wandb.log({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_accuracy': train_acc,
            'val_accuracy': val_acc 
        })

        save_best_model(trainer, val_acc)

    wandb.finish()
        
def cm():
    global best_val_acc, best_model
    wandb.init(name="evaluation_run")
    pred = best_model.forward(x_test)
    pred = np.argmax(pred, axis=1)
    wandb.init(entity=entity, project=project)
    test_acc = np.count_nonzero(y_test == pred) / x_test.shape[0]
    cm = wandb.plot.confusion_matrix(y_true=y_test, preds=pred, class_names=class_names, title="Confusion_Matrix on Test set")
    wandb.log({
        'confusion_matrix': cm,
        'test_acc': test_acc
    })
    wandb.finish()

if __name__ == '__main__':
    entity = 'da24m005-iit-madras'
    project = "Project1"
    sweep_id = wandb.sweep(sweep_config, project=project, entity=entity)

    wandb.agent(sweep_id, function=sweep, count=120)
    wandb.teardown()
    wandb.agent(sweep_id,function=cm, count=1) 