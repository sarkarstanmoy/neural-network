import numpy as np
a  = [1,2,3]
b = [2,3,4]
arr = np.array([a])
print(arr)
arr1 = np.expand_dims(arr, axis=0)
print(arr1)
arr2 = np.expand_dims(arr, axis=1)
print(arr2)

#transpose
print('------------------')
arr3 = np.transpose(arr)
print(arr3)

print(np.dot(a, b))
print(np.dot(np.array([a]),np.array([b]).transpose()))