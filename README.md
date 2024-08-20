
# Project: Custom Neural Network with Manual Autograd and Backpropagation

This project is a simple implementation of a neural network from scratch, designed primarily for educational purposes. It includes the manual computation of forward passes, backpropagation, and gradient updates without relying on high-level libraries such as PyTorch's autograd. The goal is to gain a deeper understanding of how neural networks operate at a fundamental level.

## Project Overview

This project implements a basic neural network with a single hidden layer, a Tanh activation function, and a cross-entropy loss function. The network is trained on the MNIST dataset to verify that the manual backpropagation and gradient computation are functioning correctly. This training serves as a proof of concept rather than a high-performance model.

The neural network architecture includes:
- **Input Layer**: 784 nodes (28x28 pixels)
- **Hidden Layer**: 100 nodes with Tanh activation
- **Output Layer**: 10 nodes (one for each digit class)

## Dependencies

- `torch` (only for tensor operations, without autograd)
- `torchvision` (for loading the MNIST dataset)

To install the necessary dependencies, run:
```bash
pip install torch torchvision
```

## Usage

To train the neural network, simply run:
```bash
python main.py
```

The script will download the MNIST dataset, initialize the model, and start the training process for the specified number of epochs. The training loss will be printed for each epoch.

## Credits

This project was inspired and guided by the following resources:

- [The spelled-out intro to neural networks and backpropagation: building micrograd](https://youtu.be/VMj-3S1tku0?si=Wm-nFiP3cI0QNXvj) by Andrej Karpathy.
- [Building makemore Part 4: Becoming a Backprop Ninja](https://youtu.be/q8SA3rM6ckI?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ) by Andrej Karpathy.
