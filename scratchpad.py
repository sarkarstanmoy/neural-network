from nnfs.datasets import spiral_data
import numpy as np
import nnfs

nnfs.init()
X, y = spiral_data( samples = 100 , classes = 3 )
print (X.shape) 
print (y.shape)
