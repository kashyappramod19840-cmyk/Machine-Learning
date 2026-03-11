
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import k_means
X = np.array([
    [1, 2],
    [1.5, 1.8],
    [5, 8],
    [8, 8],
    [1, 0.6],
    [9, 11],
    [8, 2],
    [10, 2],
    [9, 3]
])

kmeans = KMeans(n_clusters=3)

kmeans.fit(X)

labels = kmeans.labels_


centroids = kmeans.cluster_centers_


plt.scatter(X[:,0], X[:,1], c=labels)
plt.scatter(centroids[:,0], centroids[:,1], marker='X', s=200)

plt.title("K-Means Clustering (Unsupervised Learning)")
plt.show()
