#!/usr/bin/env python

import numpy as np
import random

from q1_softmax import softmax
from q2_sigmoid import sigmoid, sigmoid_grad
from q2_gradcheck import gradcheck_naive


def forward_backward_prop(data, labels, params, dimensions):
    """
    Forward and backward propagation for a TWO-layer sigmoidal network

    Compute the forward propagation and for the cross entropy cost,
    and backward propagation for the gradients for all parameters.

    Arguments:
    data -- M x Dx matrix, where each row is a training example.
    labels -- M x Dy matrix, where each row is a one-hot vector.
    params -- Model parameters, these are unpacked for you.
    dimensions -- A tuple of input dimension, number of hidden units
                  and output dimension
    """

    ### Unpack network parameters (do not modify)
    ofs = 0
    Dx, H, Dy = (dimensions[0], dimensions[1], dimensions[2])

    W1 = np.reshape(params[ofs:ofs+ Dx * H], (Dx, H))
    ofs += Dx * H
    b1 = np.reshape(params[ofs:ofs + H], (1, H))
    ofs += H
    W2 = np.reshape(params[ofs:ofs + H * Dy], (H, Dy))
    ofs += H * Dy
    b2 = np.reshape(params[ofs:ofs + Dy], (1, Dy))

    ### YOUR CODE HERE: forward propagation
    #I use the same notation as the written hw.
    z1 = np.dot(data,W1)+b1     #(20,5)
    h = sigmoid(z1)             #
    z2 = np.dot(h,W2)+b2        #(20,10)
    y_hat = softmax(z2)         #

    cost = np.sum( -labels * np.log(y_hat) ) 

    ### YOUR CODE HERE: backward propagation   
    delta3 = y_hat - data               #(20,10)
    
    # gradW2 means you take the derivative of the cost function w.r.t W2.
    gradW2 = h.T.dot(delta3)          #(5,10)
    gradb2 = np.sum(delta3)             #scalar

    delta2 = delta3.dot(W2.T) * sigmoid_grad(z1)  #(20,10)

    gradW1 = delta2.T.dot(data)                       #()
    gradb1 = np.sum(delta2)  

    ### Stack gradients (do not modify)
    grad = np.concatenate((gradW1.flatten(), gradb1.flatten(),
        gradW2.flatten(), gradb2.flatten()))

    return cost, grad


def sanity_check():
    """
    Set up fake data and parameters for the neural network, and test using
    gradcheck.
    """
    print "Running sanity check..."

    N = 20
    dimensions = [10, 5, 10]
    data = np.random.randn(N, dimensions[0])   # each row will be a datum
    labels = np.zeros((N, dimensions[2]))
    for i in xrange(N):
        labels[i, random.randint(0,dimensions[2]-1)] = 1

    params = np.random.randn((dimensions[0] + 1) * dimensions[1] + (
        dimensions[1] + 1) * dimensions[2], )

    gradcheck_naive(lambda params:
        forward_backward_prop(data, labels, params, dimensions), params)
    #gradcheck_naive(quad, np.array(123.456))   
    #Gradient check passed!
    #forward_backward_prop(data, labels, params, dimensions) returns cost and grad

def your_sanity_checks():
    """
    Use this space add any additional sanity checks by running:
        python q2_neural.py
    This function will not be called by the autograder, nor will
    your additional tests be graded.
    """
    print "Running your sanity checks..."



if __name__ == "__main__":
    sanity_check()
    your_sanity_checks()
