import mnist_loader
import network_img
from PIL import Image
import numpy as np

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
im = Image.open('3.jpg').convert('L')

t_data = (1-(np.float32(np.array(im))/255)).reshape(784,1)

net = network_img.Network([784, 30, 10])
net.SGD(training_data, 30, 10, 5, test_data=t_data)

