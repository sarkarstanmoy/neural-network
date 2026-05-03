import numpy as np
from dense_layer import DenseLayer

inputs = [[1, 2, 3, 2.5], [2., 5., -1., 2], [-1.5, 2.7, 3.3, -0.8]]

weights = [[0.2, 0.8, -0.5, 1],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]
print(np.array(weights).T)
biases = [2, 3, 0.5]

weights2 = [[0.1, -0.14, 0.5],
            [-0.5, 0.12, -0.33],
            [-0.44, 0.73, -0.13]]
biases2 = [-1, 2, -0.5]

layer1 = DenseLayer(4, 3)
layer1.weights = np.array(weights).T
layer1.biases = biases

layer2 = DenseLayer(3, 3)
layer2.weights = np.array(weights2).T
layer2.biases = biases2

layer1.forward(inputs)
print(layer1.output)

layer2.forward(layer1.output)
print(layer2.output)
