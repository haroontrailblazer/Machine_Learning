from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np

# Sample data: list of (x, y) coordinates
data = [
    [1, 2], [1.5, 1.8], [5, 8],
    [8, 8], [1, 0.6], [9, 11],
    [8, 2], [10, 2], [9, 3]
]

# Convert to a NumPy array
X = np.array(data)

# KMeans + Yellowbrick
plt.style.use('dark_background')
plt.grid(True, linestyle='--', alpha=0.1)
model = KMeans(random_state=42)
visualizer = KElbowVisualizer(model, k=(1, 10))
visualizer.fit(X)
visualizer.show()

# Create and fit the model
kmeans = KMeans(n_clusters=visualizer.elbow_value_, random_state=42)
kmeans.fit(data)

# Predict cluster labels
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

# Plotting
for idx, point in enumerate(data):
    plt.scatter(point[0], point[1], c=f"C{labels[idx]}", label=f"Point {idx+1}")

# Plot centroids
plt.scatter(centroids[:, 0], centroids[:, 1], s=200, c='red', marker='X', label='Centroids')
plt.title("K-Means Clustering Example")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.1)
plt.show()