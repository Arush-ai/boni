# Importing necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# Step 1: Load the dataset (Iris dataset)
iris = load_iris()
X = iris.data  # Features
y = iris.target  # True labels (for comparison, not used in unsupervised learning)

# Step 2: Data Preprocessing
# Standardize the data to ensure each feature contributes equally to the model
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 3: Apply K-Means Clustering
# Define the model (choosing 3 clusters based on the fact that there are 3 species in the Iris dataset)
kmeans = KMeans(n_clusters=3, random_state=42)
y_kmeans = kmeans.fit_predict(X_scaled)

# Step 4: Visualize the clusters
# Use PCA to reduce the data to 2D for visualization purposes
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Plotting the clusters
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_kmeans, cmap='viridis')
plt.title("K-Means Clustering of Iris Dataset")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar(label='Cluster')
plt.show()

# Step 5: Evaluate the clustering performance using silhouette score
sil_score = silhouette_score(X_scaled, y_kmeans)
print(f"Silhouette Score: {sil_score:.4f}")
#  Silhouette Score provides a measure of 
# how similar each point is to its own cluster compared to other clusters


