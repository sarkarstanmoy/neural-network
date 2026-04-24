import numpy as np

inputs = [1.0,2.0,3.0]
weights = [0.2,0.4,0.6]
bias = 2.0

outputs = np.dot(inputs,weights) + bias
print(outputs)