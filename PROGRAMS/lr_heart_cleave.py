# # Import necessary libraries
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
# import matplotlib.pyplot as plt
# import seaborn as sns

# columns = [
#     "age","sex","cp","trestbps","chol","fbs","restecg",
#     "thalach","exang","oldpeak","slope","ca","thal","target"
# ]

# # Load the .data file directly
# df_orig = pd.read_csv("C:\\Users\\pc\\OneDrive\\Desktop\\ML LAB\\DATASETS\\processed.cleveland.data", names=columns)

# # Check if there are missing values marked as '?'
# print(df_orig.head())
# print(df_orig.isnull().sum())

# df_orig = df_orig.replace('?', pd.NA).dropna()  # Remove rows with missing values
# df_orig = df_orig.astype(float)  # Convert to numeric
# df_orig.to_csv("heart.csv", index=False)

# df = pd.read_csv("heart.csv")  # Make sure the file exists in the same folder

# # Load the dataset
# df = pd.read_csv("heart.csv")

# # Display first few rows
# print("Dataset Preview:")
# print(df.head())

# # Split features and target
# X = df.drop("target", axis=1)
# y = df["target"]

# # Split data into training and testing sets (80% train, 20% test)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Standardize the features
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

# # Create a Logistic Regression model
# log_reg = LogisticRegression(max_iter=1000, random_state=42)

# # Train the model
# log_reg.fit(X_train, y_train)

# # Make predictions
# y_pred = log_reg.predict(X_test)

# # Evaluate performance
# print("\nAccuracy:", accuracy_score(y_test, y_pred))
# print("\nClassification Report:\n", classification_report(y_test, y_pred))

# # Confusion Matrix
# cm = confusion_matrix(y_test, y_pred)

# # Plot confusion matrix
# plt.figure(figsize=(6,5))
# sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', cbar=False)
# plt.title('Confusion Matrix for Logistic Regression')
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.show()



# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

columns = [
    "age","sex","cp","trestbps","chol","fbs","restecg",
    "thalach","exang","oldpeak","slope","ca","thal","target"
]

# Load the .data file directly
df_orig = pd.read_csv("C:\\Users\\pc\\OneDrive\\Desktop\\ML LAB\\DATASETS\\processed.cleveland.data", names=columns)

# Check if there are missing values marked as '?'
print(df_orig.head())
print(df_orig.isnull().sum())

df_orig = df_orig.replace('?', pd.NA).dropna()  # Remove rows with missing values
df_orig = df_orig.astype(float)  # Convert to numeric
df_orig.to_csv("heart.csv", index=False)

# Load the dataset
df = pd.read_csv("heart.csv")  # Make sure the file exists in the same folder

# Display first few rows
print("Dataset Preview:")
print(df.head())

# Split features and target
X = df.drop("target", axis=1)
y = df["target"]

# Standardize the features (fit on all data for CV)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Create a Logistic Regression model
log_reg = LogisticRegression(max_iter=1000, random_state=42)

# Cross-validation with 5 folds
cv_scores = cross_val_score(log_reg, X_scaled, y, cv=5, scoring='accuracy')

print("Cross-Validation Results:")
print(f"CV Accuracy Scores: {cv_scores}")
print(f"Mean CV Accuracy: {cv_scores.mean():.4f}")
print(f"Standard Deviation: {cv_scores.std():.4f}")
print(f"95% Confidence Interval: {cv_scores.mean() - 2* cv_scores.std():.4f} - {cv_scores.mean() + 2* cv_scores.std():.4f}")

# Plot CV scores
plt.figure(figsize=(8,5))
plt.bar(range(1, 6), cv_scores, alpha=0.7, color='skyblue', edgecolor='black')
plt.axhline(y=cv_scores.mean(), color='red', linestyle='--', label=f'Mean: {cv_scores.mean():.4f}')
plt.xlabel('Fold')
plt.ylabel('Accuracy')
plt.title('Cross-Validation Accuracy Scores')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Traditional train/test split for confusion matrix
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
log_reg.fit(X_train, y_train)
y_pred = log_reg.predict(X_test)
print("Accuracy: ", accuracy_score(y_pred, y_test))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', cbar=False)
plt.title('Confusion Matrix for Logistic Regression')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()