import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.cluster import KMeans, AgglomerativeClustering
from scipy.cluster import hierarchy as shc

dataset = pd.read_csv('data/Mall.csv')

X = dataset.iloc[:, [3, 4]].values

#within cluster sum of squares
# km_classifier.inertia_ --> sum of squares
wcss = []
for i in range(1, 11):
    km_classifier = KMeans(n_clusters=i, init='k-means++',max_iter=300, n_init=10)
    km_classifier.fit(X)
    wcss.append(km_classifier.inertia_)

plt.plot(range(1, 11), wcss)
plt.title("The Elbow method")
plt.xlabel("Number of clusters")
plt.ylabel("WCSS")
plt.show()

km_classifier = KMeans(n_clusters=5, init='k-means++',max_iter=300, n_init=10)
y_means = km_classifier.fit_predict(X)

plt.scatter(X[y_means == 0, 0], X[y_means == 0, 1], s=100, c='orange', label='Careless')
plt.scatter(X[y_means == 1, 0], X[y_means == 1, 1], s=100, c='magenta', label='Standard')
plt.scatter(X[y_means == 2, 0], X[y_means == 2, 1], s=100, c='red', label='Target')
plt.scatter(X[y_means == 3, 0], X[y_means == 3, 1], s=100, c='blue', label='Careful')
plt.scatter(X[y_means == 4, 0], X[y_means == 4, 1], s=100, c='green', label='Sensible')
plt.scatter(km_classifier.cluster_centers_[:, 0], km_classifier.cluster_centers_[:, 1], s=300, c='yellow', label='centroid')
plt.title("Clusters of Clients")
plt.xlabel("Annual Income(k$)")
plt.ylabel("Spending Score (1-100)")
plt.legend()
plt.show()


# Taking all the features into consideration

encoder = LabelEncoder()
data = dataset.iloc[:, :].values
data[:, 1] = encoder.fit_transform(data[:, 1])

wcss = []
for i in range(1, 11):
    km_classifier = KMeans(n_clusters=i, init='k-means++',max_iter=300, n_init=10)
    km_classifier.fit(data)
    wcss.append(km_classifier.inertia_)

plt.plot(range(1, 11), wcss)
plt.title("The Elbow method")
plt.xlabel("Number of clusters")
plt.ylabel("WCSS")
plt.show()


## Hierarchial Clustering

dd = shc.dendrogram(shc.linkage(X, method='ward'))
# `ward` method tries to minimize the VARIANCE in each of the clusters
plt.title("Dendrogram")
plt.xlabel("Customers")
plt.ylabel("Eucleadian Distance")
plt.show()

hc = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')
y_hc = hc.fit_predict(X)

plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], s=50, c='orange', label='Careless')
plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], s=50, c='magenta', label='Standard')
plt.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1], s=50, c='red', label='Target')
plt.scatter(X[y_hc == 3, 0], X[y_hc == 3, 1], s=50, c='blue', label='Careful')
plt.scatter(X[y_hc == 4, 0], X[y_hc == 4, 1], s=50, c='green', label='Sensible')
plt.title("Clusters of Clients")
plt.xlabel("Annual Income(k$)")
plt.ylabel("Spending Score (1-100)")
plt.legend()
plt.show()
