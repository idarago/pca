# # # # # # # # # # # # # # #
# Example with Iris dataset #
# # # # # # # # # # # # # # #
from pca import *
import matplotlib.pyplot as plt
from sklearn import datasets

iris = datasets.load_iris()
X = iris.data
y = iris.target

# Project the data onto the 2 primary principal components so that we can plot it
pca = PCA(2)
pca.fit(X)
X_projected = pca.transform(X)

# Plot the data
labelled = [[],[],[]]
for i in range(len(X_projected)):
    labelled[y[i]].append(X_projected[i])

colors = ["red","green","blue"]
for color, idx, target_name in zip(colors, [0,1,2], iris.target_names):
    plt.scatter([x[0] for x in labelled[idx]], [x[1] for x in labelled[idx]], c = color, label=target_name)
plt.legend()
plt.show()