# import dataset
import pandas as pd

dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:,1:-1].values
Y = dataset.iloc[:,-1].values
# import regression model
from sklearn.linear_model import LinearRegression as lr
from sklearn.preprocessing import PolynomialFeatures

regressor = lr()
regressor.fit(X,Y)

poly_reg = PolynomialFeatures(degree=5)
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = lr()
lin_reg_2.fit(X_poly,Y)

#  Test
y_pred = regressor.predict(X)

# Visualsing
import  matplotlib.pyplot as plt

plt.scatter(X,Y, color="green")
plt.plot(X,regressor.predict(X),color="blue")
plt.plot(X,lin_reg_2.predict(X_poly),color="red")
plt.title("salary vs level")
plt.xlabel("level")
plt.xlabel("salary")
plt.show()
