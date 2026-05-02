import numpy as np


class DenseLayer:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        print('-------weights-------')
        print(self.weights)
        self.biases = np.zeros((1, n_neurons))
        print('-------biases-------')
        print(self.biases)

    def forward(self, inputs):
        print('-------inputs-------')
        print(inputs)
        self.output = np.dot(inputs, self.weights) + self.biases
       