import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
# 1. Generate or load data
df = pd.read_csv("C:\\Users\\pc\\OneDrive\\Desktop\\ML LAB\\DATASETS\\kc_house_data.csv")
df_cleaned = df.dropna()
df_cleaned = df_cleaned.drop_duplicates()
df_cleaned.columns = [col.lower() for col in df_cleaned.columns]
x = df_cleaned[['sqft_living']].values # x : must be a 2D array even if it has only one feature
y = df_cleaned['price'] # y : must be a 1D array 
# 2. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# 3. Create a Linear Regression model
model = LinearRegression()

# 4. Train the model using the training data
model.fit(X_train, y_train)

# 5. Print the intercept and coefficient
print(f"Intercept: {model.intercept_}")
print(f"Coefficient (slope): {model.coef_[0]}")

# 6. Make predictions on the test set
y_pred = model.predict(X_test)

print("R^2 score:", round(r2_score(y_test, y_pred), 4))

# 7. Visualize the results (optional)
plt.scatter(X_train, y_train, color='violet', label='Training data')
plt.scatter(X_test, y_test, color='blue', label='Test data')
plt.plot(x, model.predict(x), color='red', label='Regression line')
plt.xlabel('Independent Variable (X)')
plt.ylabel('Dependent Variable (y)')
plt.title('Linear Regression')
plt.legend()
plt.show()

# 8. Predict for a new value
new_X = np.array([[2000]]) # Example new independent variable value
predicted_y = model.predict(new_X)
print(f"Predicted y for X={new_X[0][0]}: {predicted_y[0]}")