# Import necessary libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
# 1. Load dataset
iris = load_iris()

X = iris.data      # features (sepal length, width, petal length, width)
# print(X[:, 0])  # Print sepal length
y = iris.target    # target labels (0,1,2 corresponding to species)

# 2. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 3. Create KNN model
k = 3  # number of neighbors
knn = KNeighborsClassifier(n_neighbors=k)

# 4. Train the model
knn.fit(X_train, y_train)

# 5. Make predictions
y_pred = knn.predict(X_test)

# 6. Evaluate performance
print(f"Accuracy with k={k}:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 7. Predict a new sample
sample = np.array([[5.1, 3.5, 1.4, 0.2]])  # example input
predicted_class = knn.predict(sample)
print("\nPrediction for sample", sample, ":", iris.target_names[predicted_class][0])