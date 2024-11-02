# Matrix Multiplication with random 2 matrices

import numpy as np
import time

dim = 100

# create 2 matrices
a = np.random.random((dim, dim))
b = np.zeros((dim, dim))

# first option
time_start = time.time()
result_a = np.dot(a,b)
time_end = time.time()
print("time consumption: ", time_end - time_start)

# second option
result_b = np.zeros((dim, dim))
time_start = time.time()
for i in range(dim):
    for j in range(dim):
        for k in range(dim):
            result_b[i,j] += a[i,k] * b[k,j]
time_end = time.time()
print("time consumption: ", time_end - time_start)