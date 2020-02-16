# Neural Network
"""
Created on Mon Feb 11 08:37:47 2019

@author: Chad Gouws
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def sigmoid(x):
    return 1/(1+np.exp(-x))


def sigmoid_derivative(x):
    return x*(1-x)


class NeuralNetwork:

    def __init__(self, x, y):
        self.input = x
        self.weights1 = np.random.rand(self.input.shape[1], 4)
        self.weights2 = np.random.rand(4, 1)
        self.y = y
        self.output = np.zeros(self.y.shape)

    def feed_forward(self):
        self.layer1 = sigmoid(np.dot(self.input, self.weights1))
        self.output = sigmoid(np.dot(self.layer1, self.weights2))

    def back_propagate(self):
        d_weights2 = np.dot(self.layer1.T, (2 * (self.y - self.output) * sigmoid_derivative(self.output)))
        d_weights1 = np.dot(self.input.T, (np.dot(2 * (self.y - self.output) * sigmoid_derivative(self.output), self.weights2.T) * sigmoid_derivative(self.layer1)))
        
        self.weights1 += d_weights1
        self.weights2 += d_weights2
        

X = np.array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
y = np.array([0, 1, 1, 0])

weights1 = np.random.rand(X.shape[1], 4)
weights2 = np.random.rand(4, 1)
layer1 = sigmoid(np.dot(X[0], weights1))
print(X[0])
print(weights1)
print(layer1)


# nn = NeuralNetwork(X, y)
#
# for _ in range(1500):
#
#     nn.feed_forward()
#     nn.back_propagate()
#
# print(y)