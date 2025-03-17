import argparse
from trainer import Trainer
from utils import *



def create_arg_parser():
    parser = argparse.ArgumentParser(description="Train a neural network with the specified parameters.")

    # Adding arguments
    parser.add_argument('-wp', '--wandb_project', type=str,
                        help='Project name used to track experiments in Weights & Biases dashboard.')
    parser.add_argument('-we', '--wandb_entity', type=str,
                        help='Wandb Entity used to track experiments in the Weights & Biases dashboard.')
    parser.add_argument('-d', '--dataset', type=str, choices=['mnist', 'fashion_mnist'], default='fashion_mnist',
                        help='Dataset to use for training the neural network.')
    parser.add_argument('-e', '--epochs', type=int, default=10,
                        help='Number of epochs to train the neural network.')
    parser.add_argument('-b', '--batch_size', type=int, default=64,
                        help='Batch size used to train neural network.')
    parser.add_argument('-l', '--loss', type=str, choices=['mean_squared_error', 'cross_entropy'], default='cross_entropy',
                        help='Loss function to use during training.')
    parser.add_argument('-o', '--optimizer', type=str, choices=['sgd', 'momentum', 'nag', 'rmsprop', 'adam', 'nadam'],
                        default='nadam', help='Optimizer to use for training.')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001,
                        help='Learning rate used to optimize model parameters.')
    parser.add_argument('-m', '--momentum', type=float, default=0.9,
                        help='Momentum used by momentum and nag optimizers.')
    parser.add_argument('-beta', '--beta', type=float, default=0.9,
                        help='Beta used by rmsprop optimizer.')
    parser.add_argument('-beta1', '--beta1', type=float, default=0.9,
                        help='Beta1 used by adam and nadam optimizers.')
    parser.add_argument('-beta2', '--beta2', type=float, default=0.9,
                        help='Beta2 used by adam and nadam optimizers.')
    parser.add_argument('-eps', '--epsilon', type=float, default=0.000001,
                        help='Epsilon used by optimizers.')
    parser.add_argument('-w_d', '--weight_decay', type=float, default=0.5,
                        help='Weight decay used by optimizers.')
    parser.add_argument('-w_i', '--weight_init', type=str, choices=['random', 'Xavier'], default='Xavier',
                        help='Weight initialization method.')
    parser.add_argument('-nhl', '--num_layers', type=int, default=4,
                        help='Number of hidden layers used in the feedforward neural network.')
    parser.add_argument('-sz', '--hidden_size', type=str, default='128,128,64,64',
                        help='Number of hidden neurons in a feedforward layer.')
    parser.add_argument('-a', '--activation', type=str, choices=['identity', 'sigmoid', 'tanh', 'ReLU'], default='ReLU',
                        help='Activation function to use in the neural network.')

    return parser

def save_best_model(trainer, val_acc):
    global best_val_acc, best_model

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_model = trainer.model

def test_evaluate():
    global best_val_acc, best_model

    pred = best_model.forward(x_test)
    pred = np.argmax(pred, axis=1)

    test_acc = np.count_nonzero(y_test == pred) / x_test.shape[0]

    cm = wandb.plot.confusion_matrix(y_true=y_test, preds=pred, class_names=class_names, title="Confusion_Matrix on Test set")
    
    wandb.log({
        'confusion_matrix': cm,
        'test_acc': test_acc
    })

if __name__ == "__main__":
    parser = create_arg_parser()
    args = parser.parse_args()

    wandb_project = args.wandb_project
    wandb_entity = args.wandb_entity
    dataset = args.dataset
    epochs = args.epochs
    batch_size = args.batch_size
    loss = args.loss
    optimizer = args.optimizer
    lr = args.learning_rate
    momentum = args.momentum
    beta = args.beta
    beta1 = args.beta1
    beta2 = args.beta2
    eps = args.epsilon
    weight_decay = args.weight_decay
    weight_init = args.weight_init
    num_layers = args.num_layers
    hidden_size = [int(i) for i in args.hidden_size.split(',')]
    activation = args.activation

    run = wandb.init(
        entity=wandb_entity,
        project=wandb_project,
    )
    name=f'Trainer_{dataset}_{loss}_{lr}_{optimizer}'

    run.name = name

    x_train, y_train, x_test, y_test, class_names = get_dataset(dataset)

    display_images(x_train, y_train, class_names)

    (x_train, y_train), (x_val, y_val) = split_data(x_train, y_train, 0.1)

    
    best_val_acc = -float('inf') 
    best_model = None



    trainer = Trainer(
        num_layers=num_layers,
        hidden_size=hidden_size,
        activation=activation,
        init=weight_init,
        loss=loss,
        optimizer=optimizer,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        x_train=x_train,
        x_val=x_val,
        y_train=y_train,
        y_val=y_val,
        num_classes=len(class_names),
        beta=beta,
        momentum=momentum,
        beta1=beta1,
        beta2=beta2,
        weight_decay=weight_decay,
        eps=eps,
        class_names=class_names
    )

    # Train the model and get history
    history = trainer.train()

    # Log metrics to wandb
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

    test_evaluate()
    
    wandb.finish()
    
        