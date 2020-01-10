import mnist_loader
import network_print
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

net = network_print.Network([784, 30, 10])
net.SGD(training_data, 2, 10, 5, test_data=test_data)
