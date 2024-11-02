# Lab 2 - Linear Regression

# import libraries
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

# define training samples
x = np.array([[6,99], [7,86], [8,70], [7,88], [2,111], [17,86],
              [2,103], [17,86], [2,103], [9,87], [4,94], [11,78]])
target = np.array([1,100,2,1,2,1,2,100,2,2,2,2])

# define the testing samples
x_test = np.array([[3,93], [17,26]])
y_test = np.array([1,100])

# splitting the data into training and testing data
lg = LinearRegression()

# train the model
lg.fit(x, target)

# print the testing score
# the best possible score is 1.0. It can be negative because the model can be arbitrarily worse
print(lg.score(x_test, y_test))

# data scatter of predicted values
y_predict = lg.predict(x_test)
print("The predicted labels: ", y_predict)
print("True labels: ", y_test)

# plot the training samples and testing samples
plt.scatter(x_test[:,0], x_test[:,1], color='b')
plt.scatter(x[:,0], x[:,1], color='black')
plt.xlabel("x0")
plt.ylabel("x1")
plt.show()