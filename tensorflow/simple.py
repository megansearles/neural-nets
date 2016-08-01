##A simple neural network that does not work very well right now
# uses tflearn and has one conv layer and one fully connected layer
import tflearn

from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression

import tflearn.datasets.mnist as mnist
import h5py

data = h5py.File('forms_out.h5', 'r')


network = input_data(shape = [None, 256,256, 3], name='input')
network = conv_2d(network, 32, 3, activation='relu', regularizer="L2")
network = max_pool_2d(network, 2)
network = fully_connected(network, 30, activation='sigmoid')
network = fully_connected(network, 11, activation='sigmoid')
network = regression(network, optimizer='sgd', learning_rate=0.01, loss='categorical_crossentropy', name='target', batch_size=50)

model = tflearn.DNN(network, tensorboard_verbose=3)
model.fit({'input': data['train_image']}, {'target' : data['train_label']}, n_epoch=300, validation_set=({'input' : data['test_image']}, {'target' : data['test_label']}), run_id='convnet_bad', show_metric=True)
