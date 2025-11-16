import pandas as pd
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("C:\\Users\\pc\\OneDrive\\Desktop\\ML LAB\\DATASETS\\kc_house_data.csv")
print(df.head())

print("\nCorrelation between important features:")
corr = df[['price', 'sqft_living', 'bedrooms', 'bathrooms', 'floors']].corr()
print(corr)

sns.heatmap(corr, annot=True)
plt.title("Correlation Heatmap")
plt.show()


train, test = train_test_split(df, test_size=0.2, random_state=42)


model1 = smf.ols(formula='price ~ sqft_living', data=train)
res1=model1.fit()
print("\nModel 1 Summary:")
print(res1.summary())

model2 = smf.ols(formula='price ~ sqft_living + bedrooms + bathrooms+ grade', data=train)
res2=model2.fit()
print("\nModel 2 Summary:")
print(res2.summary())

'''
model3 = smf.ols(formula='price ~ sqft_living + bedrooms + bathrooms + floors+id+date+sqft_lot+waterfront+view+condition+grade+sqft_above+sqft_basement+yr_built+yr_renovated+zipcode+lat+long+sqft_living15+sqft_lot15', data=train)
res3=model3.fit()
print("\nModel 3 Summary:")
print(res3.summary())
'''