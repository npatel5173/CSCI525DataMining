# Lab 1 - Iris dataset

# identify 3 different irises from sepal and petal length and width
# use PCA on the iris dataset

# load the iris dataset from the library
# import the pca algorithm
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


# load the iris dataset from the library
iris = load_iris()

# iris include the features and classes/labels
# iris.data refers to feature
x = iris.data
plt.plot(x[:,0],x[:,1],'r*')  # colon represents anything. 0:2 returns the indices from 0 to 1
plt.show()  # plot the samples on a 2D coordinate


# perform the pca analysis
pca = PCA(n_components=2)
pca.fit(x)
print(pca.explained_variance_ratio_)  # print out the importance of different features
print(pca.fit_transform(x))  # returns the reduced dataset
