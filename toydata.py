from pca import *
import matplotlib.pyplot as plt

x = np.random.randn(1000)
y = np.random.randn(1000)
C = [[10,10],[0,1]]

# Create linearly scaled data
X = np.array([np.dot(C,[x[i],y[i]]) for i in range(1000)])

# PCA gives the directions that explains most of the variance in order
pca = PCA(2)
pca.fit(X)
X_projected = pca.transform(X)

ax = plt.figure(figsize=(20,5))
plt.scatter([x[0] for x in X], [x[1] for x in X])
origin = np.array([[0,0],[0,0]])
plt.quiver(*origin, pca.components[:,0], pca.components[:,1], color=["red","green"], scale=10, minshaft=5)
plt.show()