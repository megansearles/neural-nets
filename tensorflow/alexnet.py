#network based off of alexnet, specfically the bvlc_alexnet found in caffe
import tflearn

from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression

import h5py

data = h5py.File('forms_out.h5', 'r')

train_input = data['train_image']
test_input = data['test_image']
train_label = data['train_label']
test_label = data['test_label']

network = input_data(shape = [None, 256, 256, 3], name='data')
network = conv_2d(network, 96, 11, strides=4, activation='relu', regularizer='L2', name='conv1')
network = local_response_normalization(network, depth_radius = 5, alpha = 0.0001, beta = 0.75, name='norm1')
network = max_pool_2d(network, 3, strides=2, name= 'pool1')
network = conv_2d(network, 256, 5, strides=1, activation = 'relu', regularizer='L2', name='conv2')
network = local_response_normalization(network, depth_radius = 5, alpha=0.0001, beta=0.75, name = 'norm2')
network = max_pool_2d(network, 3, strides=2, name='pool2')
network = conv_2d(network, 384, 3, strides=1, activation = 'relu', regularizer='L2', name='conv3')
network = conv_2d(network, 384, 3, strides=1, activation = 'relu', regularizer='L2', name='conv4')
network = conv_2d(network, 256, 3, strides=1, activation = 'relu', regularizer='L2', name='conv5')
network = max_pool_2d(network, 3, strides=2, name='pool5')
network = fully_connected(network, 4096, activation='relu', name='fc6')
network = dropout(network, 0.5, name='drop6')
network = fully_connected(network, 4096, activation='relu', name='fc7')
network = dropout(network, 0.5, name='drop7')
network = fully_connected(network, 11, activation='softmax', name='final')

network = regression(network, optimizer='sgd', learning_rate=0.01, loss='softmax_categorical_crossentropy', batch_size=50, name='target')

model = tflearn.DNN(network)

model.fit({'data': train_input}, {'target' : train_label}, n_epoch=30, validation_set=({'data' : test_input}, {'target' : test_label}), run_id='alexnet', show_metric=True)
