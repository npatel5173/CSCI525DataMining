# Hands-On Lab Unit 2

import numpy as np

a1 = [[1,0,0], [23,34,3], [41,12,2]]
b1 = [[4,23,5], [6,43,7], [98,67,6]]
print("Result1: ")
result1add = np.add(a1,b1)
print(result1add)
result1dot = np.dot(a1,b1)
print(result1dot)

a2 = [[1,0,0], [3,4,3], [1,2,2]]
b2 = [[4,2,3], [6,3,7], [2,2,2]]
print("Result2: ")
result2dot = np.dot(a2,b2)
print(result2dot)

a3 = [[1,0,1], [2,3,4], [4,1,2]]
b3 = [[2,1,4], [3,1,2], [1,1,5]]
print("Result3: ")
result3dot = np.dot(a3,b3)
print(result3dot)

