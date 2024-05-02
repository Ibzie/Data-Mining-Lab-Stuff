import numpy as np
from scipy.spatial.distance import cdist


def divisive_clustering_by_dissimilarity(X, num_clusters):
    """
    Implement the Divisive Clustering algorithm based on point dissimilarity.
    Parameters:
    X (numpy.ndarray): A 2D array where each row represents a data point.
    num_clusters (int): The desired number of clusters.
    Returns:
    numpy.ndarray: An array of cluster labels for each data point.
    """
    # Initialize the clusters with two random points (best way in my opinion)
    np.random.seed(42)
    initial_centers_idx = np.random.choice(len(X), 2, replace=False)
    clusters = [[X[initial_centers_idx[0]]], [X[initial_centers_idx[1]]]]
    print("PRINTING CLUTSTERS:")
    print(clusters)

    while len(clusters) < num_clusters:
        # Calculate Manhattan distance matrix
        distance_matrix = cdist(X, np.concatenate(clusters), metric='cityblock')

        avg_distances = np.mean(distance_matrix, axis=1)

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


# Plz worK, i need you to work
np.random.seed(42)
X = np.random.normal(loc=[5, 5], scale=0.5, size=(10, 2))
labels = divisive_clustering_by_dissimilarity(X, num_clusters=2)
print("Cluster labels:", labels)

# Format the distance matrix for better understanding (It prints the entire data matrix)
formatted_distance_matrix = np.zeros((len(X), len(X)), dtype=float)
for i in range(len(X)):
    for j in range(len(X)):
        formatted_distance_matrix[i, j] = cdist([X[i]], [X[j]], metric='cityblock')

print("Distance Matrix:")
print(formatted_distance_matrix)

