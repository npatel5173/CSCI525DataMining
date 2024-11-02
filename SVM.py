import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

x = np.array([[1.6, 2.4], [1.1, 2.3],
              [3.2, 3.2], [2.9, 3.2], [2.3, 6.8],[4.3, 2.4],[2.3, 5.4],
              [6.4, 2.3], [5.1, 5.4], [5.3, 5.9]])

y = np.array([0, 0, 1, 1, 1, 1, 1, 2, 2, 2]) #define the label/class
plt.scatter(x[0:2,0], x[0:2,1], color='red')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.1)

from sklearn.svm import SVC
model = SVC(kernel='linear')
model.fit(X_train, y_train)
print("labels of the test data: ", y_test)
print(model.predict(X_test))
print(model.score(X_test, y_test))
