
import numpy as np
import random
import math
import idx2numpy
import abc

np.seterr(over='raise')

def sigmoid(x) :
        return (1/(1 + np.exp(-1.0 *x)))

def d_sigmoid(x) :
    return sigmoid(x) * (1 - sigmoid(x))

def cost(expected, real) :
    return np.sum((expected - real) ** 2) / (2 * expected.shape[1])
 
def d_cost( expected, real) :
    return  real - expected

def create_data(amount) :
    data = []
    for i in range(amount) :
        x = ( 10 * random.random())  
        data.append((x, np.abs(math.sin(x))))
    return data

def randomize(data, labels) :
    perm = np.random.permutation(len(data))

    data.data = data.data[perm]
    labels.data = labels.data[perm]

class dataset :
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def get(self, start, end) :
        """returns the data from the range [start,end) from the set as an numpy array"""
        return

    @abc.abstractmethod
    def __len__(self) :
        return

def print_array(name, array) :
    result = name + " : [" 
    for a in array :
        result += " {}, ".format(a.shape)

    result += "]"

    print(result)

class idx_dataset (dataset) :

    def __init__(self, filepath, labels=0) :
        self.data = idx2numpy.convert_from_file(filepath)

        if labels > 0 :
            temp = np.zeros((self.data.shape[0], labels))

            for i in range(self.data.shape[0]) :
                temp[i, self.data[i]] = 1

            self.data=temp

        else :
            self.data = self.data / 256


    def get(self, start, end) :
        result = self.data.view()[start:end]

        if (len(self.data.shape) == 3) :
            result.shape = (end - start, self.data.shape[1] * self.data.shape[2])

        if (len(self.data.shape) == 1) :
            result.shape = (end - start, 1)

        return result

    def __len__(self) :
        return self.data.shape[0]

class Network :

    def __init__(self, dimensions, learning_rate =0.01) :
        self.dimensions = dimensions
        self.learning_rate = learning_rate

        self.weights = []
        self.biases = []
        self.raw_activations = [] #not sure if I should use this or not, I will try it with
        self.activations = []
        for i in range(len(dimensions) - 1) :
            self.weights.append(np.random.randn(dimensions[i + 1], dimensions[i]))
            #self.weights[-1].fill(0.5)

        for i in range(1,len( dimensions))  :
            self.biases.append(np.random.randn(dimensions[i], 1))
            #self.biases[-1].fill(1)


    def submit_data(self, training_data, testing_data, training_labels, testing_labels) :
        self.training_data = training_data
        self.testing_data = testing_data
        self.training_labels = training_labels
        self.testing_labels = testing_labels

    def train_test(self, batch_size = 50, epochs = 10) :
        print("starting train/test loop")

        for e in range(epochs) :
            #randomize(self.training_data, self.training_labels)
            #randomize(self.testing_data, self.testing_labels)
            i = 0
            loss = 0
            while (i < len(self.training_data)) :
                self.feed_forward(self.training_data.get(i,i+batch_size))
                loss = self.back_propagate(self.training_labels.get(i,i+batch_size))
                i += batch_size

            accuracy = self.test(batch_size = batch_size)
            print("end of epoch {} accuracy: {} loss: {}".format(e, accuracy, loss))

    def test(self, batch_size=50) :

        i = 0
        total = 0
        while (i < len(self.testing_data)) :
            results = self.feed_forward(self.testing_data.get(i,i+batch_size))

                
            #print("--")
            #print(results.argmax(0))
            #print(self.training_labels.get(i, i+batch_size).T.argmax(0))

            results = (results.argmax(0) == self.testing_labels.get(i,i+batch_size).T.argmax(0))

            total += np.sum(results)
            i += batch_size

        return (total/len(self.testing_data))

    ##The shape of the inputs coming in should be that each row is a set of inputs
    #
    def feed_forward (self, inputs, debug=False) :
        self.activations = []
        self.raw_activations = []

        a = np.array(inputs).T

        self.activations.append(a) 

        for i in range(len(self.weights)) :
            z = np.dot(self.weights[i], a) + self.biases[i]
            self.raw_activations.append(z)
            a = sigmoid(z)
            if (debug == True) :
                print("{} dot {} + {} = {}".format(self.weights[i].T.shape, self.activations[-1].shape, self.biases[i].shape, a.shape))
            self.activations.append(a)

        return a


    ## The outputs shape should be that each row is a set of outputs
    def back_propagate (self, outputs) :
        delta = []

        #first computer the last layer's error
        delta.append(d_cost(outputs.T, self.activations[-1]) * d_sigmoid(self.raw_activations[-1]))

        #back propagate through each of the next layers error
        for i in reversed(range(len(self.weights) - 1)) :
            delta.append(np.dot(self.weights[i + 1].T, delta[-1]) * d_sigmoid(self.raw_activations[i]))


        delta.reverse()

        delta_bias = []

        for d in delta :
            delta_bias.append(np.sum(d, axis=1).reshape(d.shape[0], 1) / d.shape[1])

        delta_weights = []
        for i,d in enumerate(delta) :
            delta_weights.append(np.dot(d, self.activations[i].T) / d.shape[1])

        
        #print("--")
        #print("output: {}".format(outputs.shape))
        #print_array("weights", self.weights)
        #print_array("biases", self.biases)
        #print_array("raw_activations", self.raw_activations)
        #print_array("activations", self.activations)
        #print_array("delta", delta)
        #print_array("delta_weights", delta_weights)
        #print_array("delta_biases", delta_bias)

        for i in range(len(self.biases)) :
            self.biases[i] -= self.learning_rate * delta_bias[i]
            self.weights[i] -= self.learning_rate * delta_weights[i]

        return cost(outputs.T, self.activations[-1])

def test_back_propagation() :
    network = Network((2,3,3,2), learning_rate=1.0)

    input = np.array([[1,0],[0,1],[1,1],[0,0]])
    output = np.array([[1,0],[1,0],[1,1],[0,0]])

    print("before : ") 
    print("input: {}; {}".format(input.shape, input))
    print("weights: ") 
    for w in network.weights :
        print(w.shape, w)
    print("biases:") 
    for b in network.biases :
        print(b.shape, b)
    print()

    

def test_feed_forward() :
    network = Network((2,3,3,2))
    print(network.feed_forward(np.array([1,2])))
    print(network.feed_forward(np.array([[1,2],[2,3],[5,7]])))

def test_dimensions() :
    network = Network((2,3,3,2))
    print(network.weights)
    for w in network.weights :
        print(w.shape)
    print(network.biases)

def test_sin() :
    network = Network((1,10,1), learning_rate=10)
    network.submit_data(create_data(50000),create_data(10000)) 
    network.train_test(batch_size = 25, epochs=100)
    print(network.feed_forward(np.array([0]).reshape(1,1) ))
    print(network.feed_forward(np.array([np.pi/2]).reshape(1,1) ))

def test_mnist() :
    network = Network((784,30,10), learning_rate=0.1)
    network.submit_data(idx_dataset('data/train_images'), idx_dataset('data/test_images'), idx_dataset('data/train_labels',labels=10), idx_dataset('data/test_labels', labels=10))
    network.train_test(batch_size = 10, epochs=30)

test_mnist()
