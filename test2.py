import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist


def divisive_clustering_by_dissimilarity(X, num_clusters):
    """
    Implement the Divisive Clustering algorithm based on point dissimilarity.
    Parameters:
    X (numpy.ndarray): A 2D array where each row represents a data point.
    num_clusters (int): The desired number of clusters.
    Returns:
    numpy.ndarray: An array of cluster labels for each data point.
    """
    # Initialize the clusters with two random points
    np.random.seed(42)
    initial_centers_idx = np.random.choice(len(X), 2, replace=False)
    clusters = [[X[initial_centers_idx[0]]], [X[initial_centers_idx[1]]]]

    while len(clusters) < num_clusters:
        # Calculate Manhattan distance matrix
        distance_matrix = pdist(np.concatenate(clusters), metric='cityblock')

        # Calculate the average distance of each point to all points in its cluster
        avg_distances = np.mean(distance_matrix, axis=0)

        # Find the index of the most dissimilar point within the cluster
        max_avg_distance_idx = np.argmax(avg_distances)

        # Split the cluster into two at the most dissimilar point
        new_cluster = clusters[max_avg_distance_idx].copy()
        split_point = new_cluster.pop(max_avg_distance_idx)
        clusters[max_avg_distance_idx] = new_cluster
        clusters.append([split_point])

    # Assign cluster labels to each data point
    labels = np.zeros(len(X), dtype=int)
    for i, cluster in enumerate(clusters):
        cluster_indices = np.where(np.all(X[:, np.newaxis] == cluster, axis=-1))[0]
        labels[cluster_indices] = i

    return labels


# Test the algorithm with the provided dataset
np.random.seed(42)
X = np.random.normal(loc=[5, 5], scale=0.5, size=(10, 2))
labels = divisive_clustering_by_dissimilarity(X, num_clusters=2)
print("Cluster labels:", labels)

# Create a dendrogram
Z = linkage(X, 'single')
plt.figure(figsize=(8, 5))
dendrogram(Z)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Data points')
plt.ylabel('Distance')
plt.show()