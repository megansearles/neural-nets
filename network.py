# a simple neural network I am implementing for practice

import numpy as np

np.random.seed(2) #for now

def sigmoid(x) :
    return (1 / (1 + np.exp(-x)))

def dsigmoid(x) :
    return sigmoid(x) * (1 - sigmoid(x))

class Network :

    def __init__(self, layers) :
        self.size = len(layers)
        self.layers = tuple(layers)

        self.weights = np.random.randn(m,n) for m, n in zip(layers[1:], layers[:-1])
        self.biases = np.random.randn(n, 1) for n in layers[1:]

    #def __init__(self) :

    #    self.size = 3
    #    self.layers = (2,3,1)

    #    self.biases = [ np.array((1,1,1)).T, np.array((1,)).T]
    #    self.weights = [ np.array(((1,1,1), (1,1,1))), np.array((1,1,1)).T ]



    def feedforward(self, inputs) :
        a = np.array(inputs).T

        for i in range(len(self.biases)) :
            a = sigmoid(np.dot(self.weights[i].T, a) + self.biases[i])
            #a = np.dot(self.weights[i].T, a) + self.biases[i]

        return a
