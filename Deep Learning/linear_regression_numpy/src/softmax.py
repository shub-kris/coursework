#!/usr/bin/python3
import numpy as np
#
# (b)
#
# compute the softmax for the preactivations a.
# a is a numpy array
#


def softmax(b):
    #TODO
    b = b - np.max(b) #For numerical stability
    b = np.exp(b)
    sum = np.sum(b)
    b = b/sum
    return b

#
# compute the softmax-cossentropy between the preactivations a and the
# correct class y.
# y is an integer indicating the correct class, 0 <= y < np.size(a, axis=-1).
#
def softmax_crossentropy(a, y):
    #TODO
    y_h = np.zeros(len(a))
    y_h[y] = 1
    res = -y_h * np.log(softmax(a))
    return np.sum(res)
    
#
# (c)
#    
# compute the gradient of the softmax-cossentropy between the
# preactivations a and the correct class y with respect to the preactivations
# a.
# y is an integer indicating the correct class, 0 <= y < np.size(a, axis=-1).
#
def grad_softmax_crossentropy(a, y):
    #TODO
    z = np.zeros(len(a))
    z[y] = 1
    return(softmax(a) -  z)
#
# (d)
#

# To compute the numerical gradient at a point (a,y), for component i compute
# '(ce(a+da,y)-ce(a,y))/e' where 'da[i] = e' and the other entries of 'da' are
# zero and e is a small number, e.g. 0.0001 (i.e. use the finite differences
# method for each component of the gradient separately).
#
# implemented correctly, the difference between analytical and numerical
# gradient should be of the same magnitude as e
def numerical_gradient(a, y, e):
    #TODO
    E = np.zeros(len(a))
    E[y] = e
    return ((softmax_crossentropy(a + E,y) - softmax_crossentropy(a,y))/e)


#TODO compare analytical and numerical gradient

n = 3
for i in range(3):
    np.random.seed(i)
    a = np.random.rand(n)
    e = 0.0001
    y = i
    num_gradient = np.zeros(n)
    num_gradient[y] = numerical_gradient(a,y,e)
    anlyt_grafient = grad_softmax_crossentropy(a,y)
    print((num_gradient - anlyt_grafient)[y])


