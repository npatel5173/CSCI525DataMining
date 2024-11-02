# Lab 1 - Iris dataset

from sklearn import datasets
iris = datasets.load_iris() # load the iris dataset from the library

# iris include the features and classes/labels
# iris.data refers to feature
# iris.target refers to classes/label
x = iris.data[:, 0:2] # colon represents anything. 0:2 returns the indices from 0 to 1
y = iris.target

import matplotlib.pyplot as plt
plt.plot(x[:,0], x[:,1], 'r*')
plt.show() # plot the samples on a 2D coordinate

# import the pca algorithm
from sklearn.decomposition import PCA
pca = PCA(n_components=2)

# perform the pca analysis
# pca.fit(x)
# print(pca.explained_variance_ratio_) # print out the importance of different features
# print(pca.fit_transform(x)) # returns the reduced dataset






# __________________________________________________________________________
# import numpy as np
# import matplotlib.pyplot as plt
# X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
#  = np.array([[1, 2, 3], [2, 1, 1], [3, 8, 5], [0, 2, 0]])

# plt.plot(X[:,0], X[:,1], 'r*')
# plt.show()

# Create the Principal component analysis (PCA)
# and assign the number of components to 2.
# pca = PCA(n_components=3)

# Fit the model with X.
# pca.fit(X)

# Percentage of variance explained by
# each of the selected components.
# print(pca.explained_variance_ratio_)


# Fit the model with X and apply the dimensionality reduction on X.
# X_reduced = PCA(n_components=2).fit_transform(X)
# print(X_reduced)