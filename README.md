# Fashion MNIST Neural Network with Wandb Hyperparameter Sweeps

## **Overview**
This project implements a custom feedforward neural network for the **Fashion MNIST** dataset, designed to support modularity, customizability, and extensibility. It allows users to train and evaluate the model, explore different architectures, and optimize hyperparameters using **Weights & Biases (wandb)** sweeps. The project is built with flexibility in mind, enabling users to experiment with various activation functions, optimizers, loss functions, and weight initialization methods.

---

## **Features**
- Customizable feedforward neural network architecture.
- Modular implementation of layers, activation functions, optimizers, and loss functions.
- Support for hyperparameter tuning using wandb sweeps.
- Automatic logging of metrics (accuracy, loss) and visualizations (e.g., confusion matrix).
- Easy integration of new components (e.g., layers, activation functions).

---

## **Directory Structure**
```
.
├── nn
│   ├── Activations
│   │   ├── __init__.py
│   │   ├── _identity.py
│   │   ├── _relu.py
│   │   ├── _sigmoid.py
│   │   ├── _softmax.py
│   │   └── _tanh.py
│   ├── Layers
│   │   ├── __init__.py
│   │   ├── _flatten.py
│   │   └── _linear.py
│   ├── Losses
│   │   ├── __init__.py
│   │   ├── _crossentropy.py
│   │   └── _mse.py
│   ├── Optimizers
│   │   ├── __init__.py
│   │   ├── _adam.py
│   │   ├── _momentum.py
│   │   ├── _nadam.py
│   │   ├── _nag.py
│   │   ├── _rmsprop.py
│   │   └── _sgd.py
│   ├── __init__.py
│   └── model.py
├── sweep.py
├── sweep_mse.py
├── train.py
├── trainer.py
└── utils.py
```

### **Key Components**
1. **`nn/`**: Contains the core neural network components:
   - **Activations**: Implements activation functions like ReLU, sigmoid, tanh, etc.
   - **Layers**: Includes basic layers such as `Linear` and `Flatten`.
   - **Losses**: Provides loss functions like `CrossEntropy` and `MeanSquaredError`.
   - **Optimizers**: Implements optimizers like SGD, Adam, RMSProp, etc.
   - **model.py**: Defines the `Model` class for forward and backward propagation.

2. **`train.py`**: Script for training the neural network with configurable options via command-line arguments.

3. **`trainer.py`**: Implements the `Trainer` class to manage the training and validation process.

4. **`utils.py`**: Contains utility functions for dataset loading, data splitting, accuracy calculation, and visualization.

5. **`sweep.py` & `sweep_mse.py`**: Scripts for performing wandb sweeps:
   - `sweep.py`: Uses the cross-entropy loss function.
   - `sweep_mse.py`: Uses the mean squared error (MSE) loss function.

---

## **Installation**

1. Clone the repository:
   ```bash
   git clone https://github.com/DivyanshuTirkey/da6401_assignment1.git
   cd da6401_assignment1
   ```

2. Install required dependencies:
   ```bash
   pip install wandb numpy keras
   ```

---

## **Usage**

### **Training**
Run the `train.py` script with configurable options:
```bash
python train.py --wandb_project  --wandb_entity  \
    --dataset fashion_mnist --epochs 10 --batch_size 64 \
    --loss cross_entropy --optimizer adam --learning_rate 0.001 \
    --num_layers 4 --hidden_size 128,128,64,64 --activation ReLU \
    --weight_init Xavier
```

### **Hyperparameter Sweeps**
To perform hyperparameter tuning using wandb sweeps:
1. Configure sweep parameters in `sweep.py` or `sweep_mse.py`.
2. Run the script:
   ```bash
   python sweep.py
   ```
3. View results and visualizations on your wandb dashboard.

---

## **Arguments for `train.py`**

The following arguments can be passed to configure training:

- `--wandb_project (-wp)`: Name of the wandb project.
- `--wandb_entity (-we)`: Name of the wandb entity.
- `--dataset (-d)`: Dataset to use (`mnist`, `fashion_mnist`). Default: `fashion_mnist`.
- `--epochs (-e)`: Number of epochs (default: 10).
- `--batch_size (-b)`: Batch size (default: 64).
- `--loss (-l)`: Loss function (`mse`, `cross_entropy`). Default: `cross_entropy`.
- `--optimizer (-o)`: Optimizer (`sgd`, `momentum`, `nag`, `rmsprop`, `adam`, `nadam`). Default: `nadam`.
- `--learning_rate (-lr)`: Learning rate (default: 0.001).
- `--momentum`: Momentum (default: 0.9).
- `--beta`: Beta used by RMSprop (default: 0.9).
- `--beta1`: Beta1 used by adam and nadam(default: 0.9).
- `--beta2`: Beta2 used by adam and nadam(default: 0.9).
- `--epsilon`: Epsilon (default: 1e-6).
- `--weight_decay`: Weight decay (default: 0.5).
- `--weight_init (-w_i)`: Weight initialization method (`random`, `Xavier`). Default: Xavier.
- `--num_layers (-nhl)`: Number of hidden layers (default: 4).
- `--hidden_size (-sz)`: Comma-separated list of hidden layer sizes (e.g., `'128,128,64'`). Default: `'128,128,64'`.
- `--activation (-a)`: Activation function (`identity`, `sigmoid`, `tanh`, `ReLU`). Default: ReLU.

---

## **Results & Visualizations**

Weights & Biases automatically generates insightful visualizations during training and sweeps:
1. Hyperparameter importance plots.
2. Parallel coordinate plots showing correlations between parameters.
3. Confusion matrix for test set evaluation.

These visualizations are logged to your wandb dashboard for easy analysis.

---

## **Extensibility**

The modular design allows easy integration of new components:
1. Add new activation functions in the folder `nn/Activations`.
2. Implement custom layers in the folder `nn/Layers`.
3. Define new optimizers in the folder `nn/Optimizers`.
4. Extend loss functions in the folder `nn/Losses`.

---