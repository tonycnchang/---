# -*- coding: utf-8 -*-
"""
Created on Sat Sep  1 14:02:57 2018

@author: lenovo
"""

import numpy as np
import theano
import theano.tensor as T
from theano.tensor.nnet import conv
import theano.tensor.signal.pool as downsample
import matplotlib.pyplot as plt

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

def hardlim(y):
    if y < 0 :
        return 0
    else:
        return 1
    
x1 ,y1 = np.array([[1],[2]]) , 1
x2 ,y2 = np.array([[-1],[2]]) , 0
x3 ,y3 = np.array([[0],[-1]]) , 0
w = np.array([[1.0],[1.0]])
for j in range(5):
    cal_y1 = hardlim(np.dot(w.T,x1))
    w = w + x1
    j += 1
#    print(cal_y1)
for x,y in zip([x2,x3],[y2,y3]):
    for i in range(10):
        w = w-x
        cal_y = hardlim(np.dot(w.T,x))
#        print(cal_y)
#        cost = 1/2*(y - cal_y)**2
#        delt = (y-cal_y)*cal_y
#        w = w - delt
    i += 1
print(w,hardlim(np.dot(w.T,x1)),hardlim(np.dot(w.T,x2)),hardlim(np.dot(w.T,x3)),sep = "\n")