import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import os

print("--- Starting Machine Learning Clustering Phase ---")

# 1. Load the raw numeric data matrix you saved in the last step
if not os.path.exists("ndvi_matrix.npy"):
    if os.path.exists("sf_ndvi_matrix.npy"):
        ndvi = np.load("sf_ndvi_matrix.npy")
    else:
        raise FileNotFoundError("Could not find your saved NDVI matrix file!")
else:
    ndvi = np.load("ndvi_matrix.npy")

print(f"Loaded feature matrix shape: {ndvi.shape}")

# 2. Reshape the data for Machine Learning
flat_ndvi = ndvi.reshape(-1, 1)
print(f"Reshaped data into a flat column of {flat_ndvi.shape[0]} pixels.")

# 3. Train the K-Means Model
print("Training K-Means model to find 3 distinct terrain clusters...")
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans.fit(flat_ndvi)

# 4. Predict the zones for every pixel
print("Categorizing pixels...")
cluster_labels = kmeans.labels_

# 5. Reshape back into the original 2D map shape
clustered_map = cluster_labels.reshape(ndvi.shape)

# 6. Save the Machine Learning classification map
print("Saving your classified ML map...")
plt.figure(figsize=(10, 10))
plt.imshow(clustered_map, cmap="terrain")
plt.colorbar(ticks=[0, 1, 2], label="AI Discovered Clusters")
plt.title("Satellite Land Classification Map Using K-Means ML", fontsize=14)
plt.axis("off")

# Save the final product image
plt.savefig("ml_classified_map.png", dpi=300, bbox_inches='tight')

print("🎉 SUCCESS! Machine learning classification complete.")
print("Open 'ml_classified_map.png' to see what the AI discovered!")