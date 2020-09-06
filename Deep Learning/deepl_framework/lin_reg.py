#!/usr/bin/python3
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

from autograd import *


def load_data(filename):
    """load the data from the given file, returning a matrix for X and a vector for y"""
    xy = np.loadtxt(filename, delimiter=',')
    x = xy[:, 0:2]
    y = xy[:, 2]
    return x, y


def plot_data(inputs, targets, ax=None, cols=('blue', 'red')):
    """ plots the data to a (possibly new) ax """
    new_ax = ax is None
    if new_ax:
        # set up a new plot ax if we don't have one yet
        ax = plt.axes()
        ax.grid(True)
        plt.xlabel('x_0')
        plt.ylabel('x_1')
        plt.title('data')

    # separate classes arrays
    class0 = np.array([inputs[i] for i in range(len(inputs)) if targets[i] <= 0.5])
    class1 = np.array([inputs[i] for i in range(len(inputs)) if targets[i] > 0.5])

    # use scatter plots to visualize the data
    ax.scatter(class0[:, 1], class0[:, 2], c=cols[0], label='class 0')
    ax.scatter(class1[:, 1], class1[:, 2], c=cols[1], label='class 1')
    ax.legend()
    return ax


def accuracy(inputs, targets, weights_trained):
    _, result = forward(weights_trained, inputs)
    return np.mean((result.data >= 0.5) == targets)


def add_boundary(ax, weights_trained):
    xv, yv = np.meshgrid(np.linspace(0, 10, num=101), np.linspace(0, 10, num=101))
    x_grid = np.column_stack([np.ones(101 * 101), xv.reshape(-1), yv.reshape(-1)])
    _, result = forward(weights_trained, x_grid)
    z = result.data.reshape(101, 101)
    ax.contour(xv, yv, z, [0.5])


def forward(weights: [np.array], inputs: np.array) -> ([Variable], [Tensor]):
    all_vars, vx = [], Variable('x', inputs)
    for i, w in enumerate(weights):
        w2 = Variable('W%d' % i, w)
        all_vars.append(w2)
        if i > 0:
            vx = ReLU(vx)
        vx = MatMul(vx, w2)
    vx = Sigmoid(vx)
    return all_vars, vx


def apply_grads(weights: [np.array], all_vars: [Variable], outputs: Tensor, targets: np.array, lr: float):
    loss = mse(outputs, Variable('targets', targets))
    loss.backward()
    for w, v in zip(weights, all_vars):
        w -= lr * v.grad


def gradient_descent(weights: [np.array], lr: float, steps: int, inputs: np.array, targets: np.array) -> np.array:
    """
    Args:
      weights: initial value for the parameters theta
      lr: learning rate
      steps: total number of iterations to perform
      inputs: training inputs
      targets: training targets
    returns the final value for theta
    """
    for i in range(steps):
        v, outputs = forward(weights, inputs)
        apply_grads(weights, v, outputs, targets, lr)
        Tensor.reset_tensors()
    return weights


def main():
    # load training and test sets
    train_inputs, train_targets = load_data('data_train.csv')
    test_inputs, test_targets = load_data("data_test.csv")
    train_targets = np.expand_dims(train_targets, -1)
    test_targets = np.expand_dims(test_targets, -1)

    # extend the input data in order to add a bias term to the dot product with theta
    train_inputs = np.column_stack([np.ones(len(train_targets)), train_inputs])
    test_inputs = np.column_stack([np.ones(len(test_targets)), test_inputs])

    print('-' * 100, '\ninputs\n', train_inputs)
    print('-' * 100, '\ntargets\n', train_targets, '\n', '-' * 100)

    ax = plot_data(train_inputs, train_targets, cols=('blue', 'red'), ax=None)

    weights = []
    num_neurons = [len(train_inputs[0, :]), 100, 1]
    for n0, n1 in zip(num_neurons[:-1], num_neurons[1:]):
        weights.append(np.random.randn(n0, n1)*0.01)

    weights_trained = gradient_descent(weights=weights,
                                       lr=0.1,
                                       steps=10000,
                                       inputs=train_inputs,
                                       targets=train_targets)

    ax = plot_data(test_inputs, test_targets, cols=('lightblue', 'orange'), ax=ax)
    print("Weights:\n  %s" % '\n  '.join(['%s' % w.reshape(-1) for w in weights_trained]))
    print("Test accuracy: %.2f" % (accuracy(test_inputs, test_targets, weights_trained)))

    add_boundary(ax=ax, weights_trained=weights_trained)
    plt.show()


if __name__ == '__main__':
    main()
