import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

# 1. Load dataset
df = pd.read_csv("C:\\Users\\pc\\OneDrive\\Desktop\\ML LAB\\DATASETS\\winequality-red.csv", sep=';')

# 2. Convert 'quality' to binary labels
df["quality_label"] = df["quality"].apply(lambda x: 1 if x >= 6 else 0)  # 1 = Good, 0 = Bad

# 3. Split features and target
X = df.drop(["quality", "quality_label"], axis=1)
y = df["quality_label"]

# 4. Standardize feature values (Important for KNN!)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 5. Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y
)

# k = 5  # number of neighbors (low values => more flexible (overfitting), high values => smoother decision boundary)

ks = range(2, 26)
accuracies = []

for k in ks:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    acc = accuracy_score(y_test, knn.predict(X_test))
    accuracies.append(acc)

# Plot accuracy vs k and mark best k
plt.figure(figsize=(8,5))
plt.plot(list(ks), accuracies, marker='o', linestyle='-')
plt.xlabel('k (number of neighbors)')
plt.ylabel('Accuracy')
plt.title('KNN: Accuracy vs k')
plt.grid(True)

plt.show()
optimal_k_index = np.argmax(accuracies)
optimal_k = ks[optimal_k_index]
max_accuracy = accuracies[optimal_k_index]

print(f"Optimal k value: {optimal_k}")
print(f"Maximum accuracy achieved: {max_accuracy:.4f}")

# 6. Create and train KNN model
k = 8  # number of neighbors
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)

# 7. Make predictions
y_pred = knn.predict(X_test)

# 8. Evaluate performance
print(f"Accuracy with k={k}:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# 9. Predict on a new sample (example)
sample = np.array([[7.4,0.7,0.0,1.9,0.076,11.0,34.0,0.9978,3.51,0.56,9.4]])  # one wine sample
sample_scaled = scaler.transform(sample)
prediction = knn.predict(sample_scaled)
print("\nPrediction for sample:", "Good" if prediction[0] == 1 else "Bad")