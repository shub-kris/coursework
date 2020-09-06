import random
import numpy as np

import mnist

# Load the raw MNIST
X_train, y_train = mnist.read(dataset='training')
X_test, y_test = mnist.read(dataset='testing')

# As a sanity check, we print out the size of the training and test data.
print('Training data shape: ', X_train.shape)
print('Training labels shape: ', y_train.shape)
print('Test data shape: ', X_test.shape)
print('Test labels shape: ', y_test.shape)

# Reshape the image data into rows
# Datatype float allows you to subtract images (is otherwise uint8)
X_train = np.reshape(X_train, (X_train.shape[0], -1)).astype('float')
X_test = np.reshape(X_test, (X_test.shape[0], -1)).astype('float')
print(X_train.shape, X_test.shape)


# normalizing the input to make training easier
X_mean = np.mean(X_train)
X_stddev = np.sqrt(np.var(X_train)+1e-4)

X_train = (X_train - X_mean)/X_stddev
X_test = (X_test - X_mean)/X_stddev


def batch(num):
    idx = np.random.randint(60000, size=num)
    return X_train[idx,:], y_train[idx]

def sigmoid(x):
    return 1/(1+np.exp(-x))

# from Assignment 5
def softmax(a):
    an = (a.T - np.amax(a, axis=-1)).T
    expa = np.exp(an)
    return (expa.T/np.sum(expa, axis=-1)).T

def softmax_crossentropy(a, y):
    an = (a.T - np.amax(a, axis=-1)).T
    return np.log(np.sum(np.exp(an), axis=-1)) - an[np.arange(a.shape[0]),y]

def grad_softmax_crossentropy(a, y):
    s = softmax(a)
    s[np.arange(s.shape[0]),y] -= 1
    return s


# Parameters
training_steps = 10000
batch_size = 128
lr = 0.1


# set up variables and initialize them
lim1 = np.sqrt(6/(768 + 100))
W1 = np.random.uniform(low=-lim1, high=lim1, size=(X_train.shape[-1], 100))
b1 = np.zeros(100)

lim2 = np.sqrt(6/(100+10))
W2 = np.random.uniform(low=-lim2, high=lim2, size=(100, 10))
b2 = np.zeros(10)

def evaluate():
    u1 = X_test @ W1 + b1
    z1 = sigmoid(u1)
    u2 = z1 @ W2 + b2
    pred = np.argmax(u2, axis=-1)
    acc = np.mean(pred == y_test)
    return acc


print()
for step in range(training_steps):
    X_batch, y_batch = batch(batch_size)

    # note: @ is matrix multiplication (np.matmul())
    
    u1 = X_batch @ W1 + b1
  
    z1 = sigmoid(u1)
    
    u2 = z1 @ W2 + b2

    loss = softmax_crossentropy(u2, y_batch) # value not actually needed, for debugging only

    dLdu2 = grad_softmax_crossentropy(u2, y_batch)
    
    #save z1.T 
    z1_T = z1.T

    dLdW2 = z1_T @ dLdu2

    dLdb2 = np.sum(dLdu2,axis = 0)
    
    du2dz1 = W2.T
    
    dz1du1 = z1 * (1 - z1) # Element wise multiplicaion
  
    du1dW1 = X_batch.T

  
    temp1 = (dLdu2 @ du2dz1) * dz1du1
    
    dLdW1 = du1dW1@ temp1

    dLdb1 = np.sum(temp1,axis = 0)

    #TODO implement the remaining backpropagation steps here
    # the network should reach above 90% test accuracy

    # update the weights
    assert(W1.shape == dLdW1.shape)
    assert(b1.shape == dLdb1.shape)
    assert(W2.shape == dLdW2.shape)
    assert(b2.shape == dLdb2.shape)
    W1 -= lr * dLdW1
    b1 -= lr * dLdb1
    W2 -= lr * dLdW2
    b2 -= lr * dLdb2

    if step % 500 == 0:
        print("step: {}, loss = {}, test acc = {}".format(step, np.mean(loss), evaluate()))

    # reduce learning rate at half and 3/4 of training steps
    if step == training_steps/2:
        lr /= 10

print()

# evaluate
print("Test Accuracy: {}".format(evaluate()))
