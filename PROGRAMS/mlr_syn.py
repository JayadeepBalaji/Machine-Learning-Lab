import pandas as pd
import statsmodels.api as sm

# --------------------------
# 1. Create a sample dataset
# --------------------------
data = {
    'Area': [1500, 1600, 1700, 1800, 1900, 2000],
    'Bedrooms': [3, 3, 4, 4, 4, 5],
    'Age': [10, 8, 5, 7, 3, 2],
    'Price': [400000, 430000, 450000, 500000, 520000, 580000]
}

df = pd.DataFrame(data)

# --------------------------
# 2. Define independent and dependent variables
# --------------------------
X = df[['Area', 'Bedrooms', 'Age']]   # Independent variables
y = df['Price']                       # Dependent variable

# Add a constant term (for intercept)
X = sm.add_constant(X)

# --------------------------
# 3. Fit the OLS model
# --------------------------
model = sm.OLS(y, X).fit()

# --------------------------
# 4. Print the summary
# --------------------------
print(model.summary())

# --------------------------
# 5. Make predictions
# --------------------------
new_data = pd.DataFrame({
    'const': 1,
    'Area': [2100, 2200],
    'Bedrooms': [4, 5],
    'Age': [1, 2]
})
predictions = model.predict(new_data)
print("\nPredicted Prices:\n", predictions)
import matplotlib.pyplot as plt

# Training data predictions
df['Predicted'] = model.predict(X)

plt.scatter(df['Price'], df['Predicted'], color="blue", label="Training Data")
plt.scatter(predictions, predictions, color="red", marker="x", s=100, label="New Predictions")

# Perfect fit line
plt.plot([df['Price'].min(), df['Price'].max()],
         [df['Price'].min(), df['Price'].max()],
         'k--')

plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Prices")
plt.legend()
plt.show()

