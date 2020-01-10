import mnist_loader
import network_img
from PIL import Image
import numpy as np
import scipy
from PIL import Image
"""
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
im = Image.open('3.jpg').convert('L')

t_data = (1-(np.float32(np.array(im))/255)).reshape(784,1)

net = network_img.Network([784, 30, 10])
net.SGD(training_data, 30, 10, 5, test_data=t_data)

"""
W = np.load("weight2.npy")
B = np.load("biase2.npy")

#自己编的数据
#im = Image.open('1.jpg').convert('L')
#im1 = im.resize((28,28),Image.ANTIALIAS)
#im1.show()
#t_data = (1-(np.array(im1)/255)).reshape(784,1)

#test数据
"""
im = Image.open('7.jpg').resize((28,28),Image.ANTIALIAS).convert('L')
im = np.absolute(255-np.array(im))
t_data = np.array(im).reshape(784,1)

def sigmoid(z):

   ##The sigmoid function.##

    return 1.0/(1.0+np.exp(-1*z))
def relu(z):

    ##The sigmoid function.##

    return np.where(z<0,0,z)


A = t_data
for b,w in zip(B,W):
    
    A = sigmoid(np.dot(w, A)+b)

recog = np.argmax(A)
print(A)
print("我猜这个数是：")
"""
y = np.array([[0.01],[0.91],[0.01],[0.01],[0.01],[0.01],[0.01],[0.01],[0.01],[0.01]])

y1 = np.log(y/(1-y))
x1 = np.dot(np.dot(np.linalg.pinv(np.dot(W[1].T,W[1])),W[1].T),(y1-B[1]))

#x1[x1<0] = 0.0001
#x1[x1>1] = 0.9999
y0 = np.log(x1/(1-x1))
#y0[y0<0] = 0.0001
#y0[y0>1] = 0.9999
x0 = np.dot(np.dot(np.linalg.pinv(np.dot(W[0].T,W[0])),W[0].T),(y0-B[0]))

x0 = x0.reshape(28,28)
print(np.sum(y))
im = Image.fromarray(np.uint8(x0))
im.show()
