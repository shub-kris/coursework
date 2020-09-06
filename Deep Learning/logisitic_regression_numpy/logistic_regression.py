#!/usr/bin/python3
# -*- coding: utf-8 -*-
import csv
import numpy as np
import matplotlib.pyplot as plt


"""
logistic regression, assignment sheet 5

Authors: Shubham Krishna

complete the code sections marked with TODO
"""


def load_data(filename):
    """load the data from the given file, returning a matrix for X and a vector for y"""
    xy = np.loadtxt(filename, delimiter=',')
    x = xy[:, 0:2]
    y = xy[:, 2]
    return x, y


# (a)
def plot_data(inputs, targets, ax=None, cols=('blue', 'red')):
    """ plots the data to a (possibly new) ax """
    if ax is None:
        # set up a new plot ax if we don't have one yet, otherwise we can plot to the existing one
        ax = plt.axes()
        # TODO set sensible x/y limits
        # TODO set x/y labels and ticks
        plt.title('Student Admissions by Exam Scores')
        
        maxval = np.max(inputs[:,1:]) + 0.1
        plt.xlabel('X1')
        plt.xticks(np.arange(0, maxval+0.1, step=1))
        ax.set_xlim(left=0, right=maxval)
        
        plt.ylabel('X2')
        plt.yticks(np.arange(0, maxval+0.1, step=1))
        ax.set_ylim(bottom=0, top=maxval)

    # TODO plot the data (e.g. via scatter)
    # TODO legend
    plt.scatter(inputs[targets==0][:,1], inputs[targets==0][:,2], marker='x', label='not admitted', color=cols[1])
    plt.scatter(inputs[targets==1][:,1], inputs[targets==1][:,2], marker='o', label='admitted', color=cols[0])
    plt.legend()

    return ax


# (b)
def sigmoid(x):
    # TODO sigmoid function
    return 1/(1 + np.exp(-x))


# (c)
def cost(theta, inputs, targets, epsilon=1e-10):
    """ compute the cost function from the parameters theta """
    # TODO cost function
    # TODO add epsilon to the log expressions for numerical stability
    sigmoid_val = sigmoid(np.matmul(inputs, theta))
    log_sigmoid = np.log(sigmoid_val)
    log1_sigmoid = np.log(1 - sigmoid_val)
    return np.sum(targets * log_sigmoid + ((1- targets)* log1_sigmoid))


# (c)
def gradient(theta, inputs, targets):
    """ compute the derivative of the cost function with respect to theta """
    # TODO compute the gradient with respect to the targets
    sigmoid_val = sigmoid(np.matmul(inputs, theta))
    error = sigmoid_val - targets
    grad = np.matmul(inputs.T,error)
    return grad


# (d)
def gradient_descent(theta_0, lr, steps, inputs, targets):
    """
    Args:
      theta_0: initial value for the parameters theta
      lr: learing rate
      steps: total number of iterations to perform
      inputs: training inputs
      targets: training targets
    returns the final value for theta
    """
    # TODO iteratively update theta for 'steps' iterations
    for i in range(steps):
        theta_0 = theta_0 - lr * gradient(theta_0,inputs,targets)
    return theta_0


# (e), (f)
def accuracy(inputs, targets, theta):
    # TODO calculate the accuracy
    sigmoid_val = np.matmul(inputs, theta)
    predicted_score = [1 if val >= 0.5 else 0 for val in sigmoid_val]
    return np.sum(np.equal(predicted_score, targets))/len(targets)


# (e), (f)
def add_boundary(ax, theta_trained, polynomial_degree):
    # TODO (e) plot a decision boundary on an existing plot ax (hint: use contour plot)
    # TODO (f) consider the polynomial extension
    n = 500
    ylist = np.linspace(*(ax.get_ylim()), n)
    xlist = np.linspace(*(ax.get_xlim()), n)
    xm, ym = np.meshgrid(xlist, ylist)
    
    F = np.stack([np.ones(xm.shape), xm, ym], axis=-1)
    pivoted = F.reshape(-1,3)
    pivoted_poly = polynomial_extension(pivoted, polynomial_degree)
    stacked_poly = pivoted_poly.reshape(-1, n, len(theta_trained))
    print(f'stacked: {stacked_poly.shape}')

    sigmoid_val = sigmoid(np.matmul(stacked_poly, theta_trained))
    print(f'sigmoid : {sigmoid_val.shape}')
    zm = (sigmoid_val >= 0.5)

    print(xm.shape)
    print(ym.shape)
    print(zm.shape)

    cp = ax.contour(xm, ym, zm) 

# (f)
def polynomial_extension(inputs, degree):
    # TODO calculate the polynomial extension
    new = np.copy(inputs) 
    for i in range(degree+1):
        for j in range(degree+1):
            if i + j <= degree and (i != 0 or j != 0) and (i != 0 or j != 1) and (i != 1 or j != 0):
                new = np.column_stack([new, inputs[:,2]**i * inputs[:,1]**j])
    
    return new


def main():

    # TODO (f) test different values
    polynomial_degree = 5

    # load training and test sets
    train_inputs, train_targets = load_data('data_train.csv')
    test_inputs, test_targets = load_data("data_test.csv")

    # extend the input data in order to add a bias term to the dot product with theta
    train_inputs = np.column_stack([np.ones(len(train_targets)), train_inputs])
    test_inputs = np.column_stack([np.ones(len(test_targets)), test_inputs])

    print('-'*100, '\ninputs\n', train_inputs)
    print('-'*100, '\ntargets\n', train_targets)

    # (a) visualization
    ax = plot_data(train_inputs, train_targets, cols=('blue', 'red'), ax=None)

    train_inputs = polynomial_extension(train_inputs, degree=polynomial_degree)
    print('-'*100, '\ninputs (polynomial extension)\n', train_inputs, '\n', '-'*100)

    # (d) use these parameters for training the model
    theta_trained = gradient_descent(theta_0=np.zeros(len(train_inputs[0, :])),
                                     lr=1e-4,
                                     steps=100000,
                                     inputs=train_inputs,
                                     targets=train_targets)

    # (e) evaluation
    test_inputs = polynomial_extension(test_inputs, degree=polynomial_degree)
    
    ax = plot_data(test_inputs, test_targets, cols=('lightblue', 'orange'), ax=ax)
    print("Accuracy: " + str(accuracy(test_inputs, test_targets, theta_trained)))

    # (f) boundary plot
    add_boundary(ax=ax, theta_trained=theta_trained, polynomial_degree=polynomial_degree)
    plt.show()


if __name__ == '__main__':
    main()
