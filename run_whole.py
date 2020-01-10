import mnist_loader
import network_whole
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

net = network2.Network([784, 30, 10])
net.SGD(training_data, 50, 10, 5, test_data=test_data)
