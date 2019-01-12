## K-means

dataset <- read.csv('data/Mall.csv')
X <- dataset[4:5]

# Elbow method to find optimal number of clusters
set.seed(30)

wcss <- vector()

for(i in 1:10) wcss[i] <- sum(kmeans(X, i)$withinss)

plot(1:10, wcss, type = 'b', main = paste("Clusters of Clients"), xlab = "Number of Clusters", ylab = "WCSS")

km = kmeans(X, 5, iter.max = 300, nstart = 10)

library(cluster)
clusplot(X, km$cluster, line = 0,
         shade = TRUE, color = TRUE, labels = 2, 
         plotchar = FALSE, span = TRUE,
         main = paste("Clusters of Client"), 
         xlab = "Annual Income", ylab = "Spending Score")


## Hierarchial Clusterinig

dendrogram = hclust(d = dist(X, method = 'euclidean'), method = 'ward.D')
plot(dendrogram,
     main = paste("Dendrogram"),
     xlab = "Customers", ylab = "Euclidean Distance")

# Fitting Data
hc = hclust(d = dist(X, method = 'euclidean'), method = 'ward.D')
y_hc = cutree(hc, 5)

# Cluster Visualization
library(cluster)
clusplot(X, y_hc, line = 0,
         shade = TRUE, color = TRUE, labels = 2, 
         plotchar = FALSE, span = TRUE,
         main = paste("Clusters of Client"), 
         xlab = "Annual Income", ylab = "Spending Score")