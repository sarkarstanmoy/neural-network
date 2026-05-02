import numpy as np
a = [[1,2,3,4,5],[6,7,8,9,10],[11,12,13,14,15]]
b = np.sum(a)
print(b)
b1 = np.sum(a, axis=0)
print(b1)
b2 = np.sum(a, axis=1)
print(b2)
b3 = np.sum(a, axis=0, keepdims=True)
print(b3)
b4 = np.sum(a, axis=1, keepdims=True)
print(b4)