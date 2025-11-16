import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv("C:\\Users\\pc\\OneDrive\\Desktop\\ML LAB\\DATASETS\\kc_house_data.csv")
df_cleaned = df.dropna()
df_cleaned = df_cleaned.drop_duplicates()
df_cleaned.columns = [col.lower() for col in df_cleaned.columns]
df_cleaned.to_csv("C:\\Users\\pc\\OneDrive\\Desktop\\ML LAB\\DATASETS\\house_data_cleaned.csv", index = False)
y = df_cleaned['price']
df_cleaned = df_cleaned.drop('price', axis = 1)
X_train, X_test, Y_train, Y_test = train_test_split(df_cleaned, y, test_size=0.20)

print(len(X_train), len(X_test))

# Combine X_train and Y_train for statsmodels 
train_data = X_train.copy() # copy of df_cleaned
train_data['price'] = Y_train # adding y_train, which is price, to train_data

model = smf.ols(formula='price ~ sqft_living', data=train_data) # needs the dataframe with the target as well

res = model.fit()

plt.scatter(X_train['sqft_living'], Y_train, color='violet', label='Training data')
plt.scatter(X_test['sqft_living'], Y_test, color='blue', label='Test data')
plt.plot(X_train['sqft_living'], res.predict(train_data), color='red', label='Regression line')
# res.predict() : requires the same format as the training data
plt.xlabel('Sq. ft. living')
plt.ylabel('Price')
plt.title('Linear Regression')
plt.legend()
plt.show()

print(res.summary())
