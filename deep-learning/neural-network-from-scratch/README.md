# Neural Network from Scratch

A neural network that recognizes handwritten digits (MNIST), written in **plain NumPy**: no Keras, no PyTorch. Every step that frameworks hide (forward pass, error backpropagation and weight updates) is coded by hand.

## How it works

`mnist_numpy_network.py` defines a `NeuralNetworks` class with 3 layers:

```
784 inputs (28×28 pixels) → 200 hidden nodes → 10 outputs (digits 0 to 9)
```

- **Weights** start as small random numbers, scaled by `1/√(number of incoming links)`.
- **Activation:** sigmoid (`scipy.special.expit`).
- **Inputs** are rescaled from 0–255 to 0.01–1.0, and targets are 0.01 for wrong digits and 0.99 for the right one (avoiding the exact 0 and 1 that sigmoid can never reach).
- **Training** (`train`): forward pass, output error = target − output, hidden error = output error passed back through the weights, then gradient descent on both weight matrices (learning rate 0.1, 5 epochs).
- **Testing** (`query`): runs each test image through the network and keeps a scorecard of right and wrong answers, then prints the accuracy.

## Data

MNIST in CSV format (each row = label + 784 pixel values). The full training file is about 110 MB, over GitHub's file limit, so it isn't included.

Download `mnist_train.csv` and `mnist_test.csv` (for example from [MNIST in CSV](https://pjreddie.com/projects/mnist-in-csv/)) and put them in a `dataset/` folder next to the script.

## Run it

```bash
pip install numpy scipy matplotlib
python mnist_numpy_network.py
```

## What I learned

- What actually happens inside `model.fit()`: matrix multiplication, activation, error and weight update.
- How backpropagation splits the output error between hidden nodes in proportion to their weights.
- Why input scaling and weight initialization matter for training to work at all.
