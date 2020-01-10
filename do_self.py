# -*- coding: utf-8 -*-
"""
Created on Sat Apr 28 09:51:14 2018

@author: lenovo
"""
import numpy as np
import theano
import theano.tensor as T
from theano.tensor.nnet import conv
import theano.tensor.signal.pool as downsample
import matplotlib.pyplot as plt
import mnist_loader
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

class lyers():
    def __init__(self,inpts,in_size,out_size,activation=None):
        self.inpts = inpts
        self.W = theano.shared(np.random.randn(in_size,out_size))
        self.B = theano.shared(np.random.randn(out_size,))
        self.params = [self.W,self.B]
        self.z = theano.dot(self.inpts,self.W)+self.B
        self.activation = activation
        if activation is None:
            self.out = self.z
        if activation == "sigm":
            self.out = T.nnet.sigmoid(self.z)
        if activation == "relu":
            self.out = T.nnet.relu(self.z)
        if activation == "tanh":
            self.out = T.nnet(self.z)

class convlyer():
    def __init__(self,inpts,filter_shape,image_shape,pool_size=(2,2)):
        self.image_shape = image_shape
        self.inpts = inpts.reshape(self.image_shape)
        self.W = theano.shared(np.random.randn(5,5))
        self.B = theano.shared(np.random.randn(12,))
        self.params = [self.W,self.B]
        self.z = theano.dot(self.inpts,self.W)+self.B
        
        self.pool_size = pool_size
        
        self.filter_shape = filter_shape
        conv_out = conv.conv2d(self.inpts,image_shape=self.image_shape,
                               filters = self.W,filter_shape = self.filter_shape)
        pool_out = downsample.pool_2d(input = conv_out , ds =self.pool_size,
                                      ignore_border=True )
        self.out = T.nnet.sigmoid(pool_out)

x = T.matrices("x")
y = T.matrices("y")

ly1 = convlyer(x,filter_shape=(5,5),
               image_shape = (28,28),pool_size=(2,2))
ly2 = lyers(ly1.out,12,100,activation="relu")
ly3 = lyers(ly2.out,100,10,activation="sigm")
lamda = 0.1
cost = T.mean(1/2*T.square(ly3.outputs-y))
params = [param for ly in [ly1,ly2,ly3] for param in ly.params]
grads = T.grad(cost,params)
update = [(param,param-lamda*grad) for param,grad in zip(params,grads)]

train = theano.function([x,y],cost,updates=update)
predict = theano.function([x],ly3.outputs)

for t_data in training_data[0:10]:
    err = train(t_data[0],t_data[1])
prediction = predict(test_data[0][0])
print(prediction)