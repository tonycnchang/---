import theano
import theano.tensor as T
from theano.tensor.nnet import conv
from theano.tensor.nnet import softmax
from theano.tensor import shared_randomstreams
import theano.tensor.signal.pool as downsample
from PIL import Image
import numpy as np

import network3_cnn
from network3_cnn import Network
from network3_cnn import ConvPoolLayer, FullyConnectedLayer, SoftmaxLayer

im = Image.open('y3.jpg').convert('L')
im1 = im.resize((28,28),Image.ANTIALIAS)
im1.show()
t_data = (1-(np.array(im1)/255)).reshape(784,1)

W = np.load("weights_cnn.npy")
B = np.load("biase_cnn.npy")
"""
training_data, validation_data, test_data = network3_cnn.load_data_shared()
im = Image.open('7.jpg').convert('L')
im1 = im.resize((28,28),Image.ANTIALIAS)

t_data = (1-(np.array(im1,dtype=np.float32)/255)).reshape(28,28)

mini_batch_size = 1
net = Network([ConvPoolLayer(image_shape=(mini_batch_size, 1, 28, 28),
                             filter_shape=(20, 1, 5, 5),
                             poolsize=(2, 2)),
                FullyConnectedLayer(n_in=20*12*12, n_out=100),
               SoftmaxLayer(n_in=100, n_out=10)], mini_batch_size)
prediction = theano.function([net.x],net.output)

print("******",np.argmax(prediction(t_data)),"******",sep = "\n")
"""
#im = Image.open('y2.jpg').convert('L')
#im1 = im.resize((28,28),Image.ANTIALIAS)
#im1.show()
#t_data = (1-(np.array(im1)/255)).reshape(784,1)

def sigmoid(z):

    #The sigmoid function.

    return 1.0/(1.0+np.exp(-z))
def relu(z):

    #The sigmoid function.

    return np.where(z<0,0,z)


A = t_data
for b,w in zip(B,W):
    
    A = sigmoid(np.dot(w, A)+b)

recog = np.argmax(A)
print(A)
print("我猜这个数是：",recog)
