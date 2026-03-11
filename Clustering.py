import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

np.random.seed(42)
n = 200

income = np.random.uniform(20000, 100000, n)
spending_score = np.random.uniform(1, 100, n)

df = pd.DataFrame({
    "Income": income,
    "Spending_Score": spending_score
})

X = df[["Income", "Spending_Score"]]

kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)

labels = kmeans.labels_
centroids = kmeans.cluster_centers_

df["Cluster"] = labels

plt.figure(figsize=(6,4))
plt.scatter(df["Income"], df["Spending_Score"], c=df["Cluster"])
plt.scatter(
    centroids[:,0],
    centroids[:,1],
    marker="X",
    s=200
)
plt.xlabel("Income")
plt.ylabel("Spending Score")
plt.title("Customer Clustering using K-Means")
plt.show()
