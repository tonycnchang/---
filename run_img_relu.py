import mnist_loader
import network_img_relu
from PIL import Image
import numpy as np

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
im = Image.open('3.jpg').convert('L')
im1 = im.resize((28,28),Image.ANTIALIAS)
t_data = (1-(np.float32(np.array(im1))/255)).reshape(784,1)

net = network_img_relu.Network([784, 30, 10])
net.SGD(training_data, 3, 10, 5, test_data=test_data)

