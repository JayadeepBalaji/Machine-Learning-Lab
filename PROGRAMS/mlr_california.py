import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing

# --------------------------
# 1. Load real dataset
# --------------------------
housing = fetch_california_housing(as_frame=True)
df = pd.DataFrame(housing)   # pandas DataFrame

# --------------------------
# 2. Select predictors (independent variables) and target
# --------------------------
X = df[['MedInc', 'AveRooms', 'AveOccup']]  # predictors
y = df['MedHouseVal']                        # target: median house value

# --------------------------
# 3. Add constant (intercept term)
# --------------------------
X = sm.add_constant(X)

# --------------------------
# 4. Fit OLS model
# --------------------------
model = sm.OLS(y, X).fit()

# --------------------------
# 5. Show summary
# --------------------------
print(model.summary())

# --------------------------
# 6. Make predictions
# --------------------------
new_data = pd.DataFrame({
    'const': 1,
    'MedInc': [3.5, 7.0],      # median income in block group
    'AveRooms': [5.0, 6.5],    # average number of rooms
    'AveOccup': [2.5, 3.0]     # average occupants per household
})

predictions = model.predict(new_data)
print("\nPredicted House Values:\n", predictions)

