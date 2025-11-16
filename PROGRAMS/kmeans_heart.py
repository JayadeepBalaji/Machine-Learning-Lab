# Import required libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D

# Step 1: Load the dataset
# (Ensure heart.csv is in the same folder as this Python file)
data = pd.read_csv("C:\\Users\\pc\\OneDrive\\Desktop\\ML LAB\\DATASETS\\heart.csv")

# Step 2: Select relevant features for clustering
# You can change these features for experimentation
X = data[['age', 'chol', 'thalach', 'trestbps']]

# Step 3: Standardize the features (important for clustering)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 4: Apply K-Means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_scaled)

# Step 5: Add cluster labels to the dataset
data['Cluster'] = kmeans.labels_

# Step 6: Display cluster centers (after scaling)
print("Cluster Centers (Standardized):\n", kmeans.cluster_centers_)
print("\nCluster Distribution:\n", data['Cluster'].value_counts())

# Step 7: Visualize the clusters using 3D plot (Age vs Chol vs Thalach)
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(data['age'], data['chol'], data['thalach'],
                    c=data['Cluster'], cmap='viridis', edgecolors='black')
ax.set_title('K-Means Clustering on Heart Disease Data (3D)')
ax.set_xlabel('Age')
ax.set_ylabel('Cholesterol')
ax.set_zlabel('Max Heart Rate (Thalach)')
plt.colorbar(scatter, label='Cluster')
plt.show()

# Step 8: Optional – Visualize Age vs Max Heart Rate (Thalach)
plt.figure(figsize=(8,6))
plt.scatter(data['age'], data['thalach'], c=data['Cluster'], cmap='plasma', edgecolors='black')
plt.title('Clusters (Age vs Max Heart Rate)')
plt.xlabel('Age')
plt.ylabel('Max Heart Rate (Thalach)')
plt.colorbar(label='Cluster')
plt.show()

# Step 9: Optional – Visualize chol vs Max Heart Rate (Thalach)
plt.figure(figsize=(8,6))
plt.scatter(data['chol'], data['thalach'], c=data['Cluster'], cmap='plasma', edgecolors='black')
plt.title('Clusters (Cholesterol vs Max Heart Rate)')
plt.xlabel('Cholesterol')
plt.ylabel('Max Heart Rate (Thalach)')
plt.colorbar(label='Cluster')
plt.show()