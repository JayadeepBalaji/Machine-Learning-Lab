# Importing required libraries for data handling, model building, evaluation, and visualization
import pandas as pd  # pandas is used for data manipulation and analysis, especially with tabular data (DataFrames)
from sklearn.model_selection import train_test_split  # for splitting data into training and testing sets
from sklearn.tree import DecisionTreeClassifier, plot_tree  # DecisionTreeClassifier builds the tree, plot_tree visualizes it
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix  # for evaluating model performance
import matplotlib.pyplot as plt  # for plotting graphs
import seaborn as sns
# 1. Load dataset
# pd.read_csv reads a CSV file and loads it into a DataFrame (df). 'sep=;' means columns are separated by semicolons in the file.
df = pd.read_csv("C:\\Users\\pc\\OneDrive\\Desktop\\ML LAB\\DATASETS\\winequality-red.csv", sep=';')
# plt.pie(df['quality'].value_counts(), labels=df['quality'].value_counts().index) 
plt.show()
df.replace({3:0, 4:0, 5:0, 6:1, 7:1, 8:1}, inplace=True)

# 2. Features and Target
# X contains all columns except 'quality' (these are the input features for prediction)
# y contains only the 'quality' column (this is the target variable we want to predict)
X = df.drop("quality", axis=1)  # axis=1 means drop a column, not a row
y = df["quality"]


# 3. Split into training and testing sets
# train_test_split splits X and y into training and testing sets
# test_size=0.3 means 30% of the data is for testing, 70% for training
# random_state=42 sets the seed for the random number generator, so the split is reproducible (same every run). If set to None, the split will be different each time.
# stratify=y keeps the class distribution similar in both sets (important for classification) [to avoid class imbalance]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)


# 4. Create and train Decision Tree
# DecisionTreeClassifier is initialized with:
#   criterion='entropy' (splits are chosen based on information gain/entropy)
#   max_depth=4 (limits the tree to 4 levels to prevent overfitting)
#   random_state=42 (sets the seed for the random number generator, so results are reproducible. If None, results will vary each run)
clf = DecisionTreeClassifier(criterion="entropy", max_depth=4, random_state=42)
# .fit() trains the model using the training data (X_train, y_train)
clf.fit(X_train, y_train)


# 5. Evaluate model
# .predict() uses the trained model to predict the target for X_test
y_pred = clf.predict(X_test)
confusionMatrix = confusion_matrix(y_test, y_pred)
# accuracy_score compares y_test (true labels) and y_pred (predicted labels) and returns the fraction correct
print("Accuracy:", accuracy_score(y_test, y_pred))
# classification_report gives precision, recall, f1-score, and support for each class
print("\nClassification Report:\n", classification_report(y_test, y_pred))
# To display integers (like 202) without scientific notation, use fmt="d"
# You can change the color of the heatmap in seaborn using the cmap parameter.
sns.heatmap(confusionMatrix, annot = True, fmt = 'd', cmap='Blues')
print("Confusion Matrix:\n", confusionMatrix)
# 6. Visualize Decision Tree
# plt.figure sets the size of the plot (width=15, height=10 inches)
plt.figure(figsize=(15,10))
# plot_tree draws the trained decision tree:
#   feature_names=X.columns labels each split with the feature name
#   class_names=[str(c) for c in sorted(y.unique())] labels each class (wine quality levels)
#   filled=True colors the nodes by class
#   fontsize=8 sets the font size for readability
plot_tree(clf, feature_names=X.columns, class_names=[str(c) for c in sorted(y.unique())],
          filled=True, fontsize=8)
# plt.show() displays the plot
plt.show()

# ---
# Summary of logic and parameters:
# 1. Data is loaded and organized into features (X) and target (y).
# 2. Data is split into training and testing sets to evaluate generalization.
# 3. DecisionTreeClassifier is configured with parameters to control how the tree is built:
#    - criterion: 'entropy' uses information gain for splits (alternative: 'gini').
#    - max_depth: limits tree complexity to avoid overfitting.
#    - random_state: ensures reproducibility.
# 4. Model is trained (fit) on training data, then used to predict (predict) on test data.
# 5. Model performance is measured using accuracy and a detailed classification report.
# 6. The tree is visualized for interpretability, showing how decisions are made.

# DecisionTreeClassifier(
#     criterion='gini',         # Function to measure the quality of a split: 'gini' (default) or 'entropy'
#     splitter='best',          # Strategy to choose the split at each node: 'best' (default) or 'random'
#     max_depth=None,           # Maximum depth of the tree (how many splits). None means unlimited.
#     min_samples_split=2,      # Minimum number of samples required to split an internal node
#     min_samples_leaf=1,       # Minimum number of samples required to be at a leaf node
#     max_features=None,        # Number of features to consider when looking for the best split
#     random_state=None,        # Seed for random number generator (for reproducibility)
#     max_leaf_nodes=None,      # Maximum number of leaf nodes
#     min_impurity_decrease=0.0 # A node will be split if this split induces a decrease of the impurity greater than or equal to this value
# )