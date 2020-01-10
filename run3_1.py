# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 16:02:31 2018

@author: lenovo
"""
import numpy as np
import network3
from network3 import Network
from network3 import ConvPoolLayer, FullyConnectedLayer, SoftmaxLayer

training_data, validation_data, test_data = network3.load_data_shared()

a = validation_data
print(a[1].get_value())
"""
mini_batch_size = 10

net = Network([ConvPoolLayer(image_shape=(mini_batch_size, 1, 28, 28),
                             filter_shape=(20, 1, 5, 5),
                             poolsize=(2, 2)),
                FullyConnectedLayer(n_in=20*12*12, n_out=100),
               SoftmaxLayer(n_in=100, n_out=10)], mini_batch_size)

net.SGD(training_data, 50, mini_batch_size, 0.1,
        validation_data, test_data)
np.save('biase_cnn.npy',[net.layers[0].b,net.layers[1].b,net.layers[2].b])
np.save('weights_cnn.npy',[net.layers[0].w,net.layers[1].w,net.layers[2].w])

"""