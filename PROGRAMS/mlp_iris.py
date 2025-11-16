# # Neural Network using sklearn's MLPClassifier
# from sklearn.datasets import load_iris
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.neural_network import MLPClassifier
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# import matplotlib.pyplot as plt
# import seaborn as sns
# import pandas as pd

# # Step 1: Load dataset
# X = iris.data       # features
# y = iris.target     # labels (0, 1, 2 for species)

# # Step 2: Split into train and test sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# # Step 3: Feature scaling (important for neural networks)
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

# # Step 4: Define the MLP model
# model = MLPClassifier(hidden_layer_sizes=(10, 8),  # 2 hidden layers (10 and 8 neurons)
#                       activation='relu',            # ReLU activation
#                       solver='adam',                # Adam optimizer
#                       max_iter=1000,                # Number of training epochs
#                       random_state=42)

# # Step 5: Train the model
# model.fit(X_train, y_train)

# # Step 6: Make predictions
# y_pred = model.predict(X_test)

# # Step 7: Evaluate model performance
# print("Accuracy:", accuracy_score(y_test, y_pred))
# print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=iris.target_names))

# # Step 8: Visualize confusion matrix
# cm = confusion_matrix(y_test, y_pred)
# plt.figure(figsize=(6, 4))
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
#             xticklabels=iris.target_names,
#             yticklabels=iris.target_names)
# plt.title('Confusion Matrix')
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.show()

import pandas as pd                          # read CSV, DataFrame handling
import numpy as np                           # numeric operations
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load heart dataset (ensure path is correct)
df = pd.read_csv("C:\\Users\\pc\\OneDrive\\Desktop\\ML LAB\\DATASETS\\heart.csv")

# Basic cleaning: drop rows with missing values
df = df.dropna().reset_index(drop=True)

# If 'target' encodes severity (>0) convert to binary: 0 = no disease, 1 = disease
if df['target'].nunique() > 2:
    df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)

# Features and labels
X = df.drop('target', axis=1).values        # feature matrix (all columns except 'target')
y = df['target'].values                     # target vector

# Quick class distribution check
print("Class distribution:\n", pd.Series(y).value_counts())

# Split into train/test (stratify to keep class balance)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Scale features: fit on train, transform both
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define and train MLP
model = MLPClassifier(hidden_layer_sizes=(50,), activation='relu',
                      solver='adam', max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=['NoDisease','Disease']))

# Confusion matrix plot (integers, readable colors)
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['NoDisease','Disease'],
            yticklabels=['NoDisease','Disease'])
plt.title('Confusion Matrix (MLP on heart.csv)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()