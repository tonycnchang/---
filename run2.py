import mnist_loader
import network2
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

net = network2.Network([784, 30, 10],cost = network2.CrossEntropyCost)
result = net.SGD(training_data, 50, 10, 0.5,lmbda = 5.0,evaluation_data=test_data,monitor_evaluation_accuracy=True)

#evaluation_cost, evaluation_accuracy,training_cost, training_accuracy = result
