# import dataset
import pandas as pd
import numpy as np

dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:,1].values
Y = dataset.iloc[:,-1].values
X = X.reshape(len(X),1)
Y = Y.reshape(len(Y),1)

# feature scaling
from sklearn.preprocessing import StandardScaler

x_scaler = StandardScaler()
y_scaler = StandardScaler()
X = x_scaler.fit_transform(X)
Y = y_scaler.fit_transform(Y)

# import regression model
from sklearn.svm import SVR

regressor = SVR(kernel="rbf",)
regressor.fit(X,Y)

#  Test
y_pred = y_scaler.inverse_transform(regressor.predict(x_scaler.transform([[6.5]]))) 

# Visualsing
import  matplotlib.pyplot as plt

plt.scatter(x_scaler.inverse_transform(X),y_scaler.inverse_transform(Y), color="green")
plt.plot(x_scaler.inverse_transform(X),y_scaler.inverse_transform(regressor.predict(X)),color="blue")
plt.title("salary vs level")
plt.xlabel("level")
plt.xlabel("salary")
plt.show()

from  sklearn.preprocessing import PolynomialFeatures

X_grid = np.arange(min(x_scaler.inverse_transform(X)),max(x_scaler.inverse_transform(X)),0.1)
X_grid = X_grid.reshape(len(X_grid),1)
poly_scaler = PolynomialFeatures(degree=5)
plt.scatter(x_scaler.inverse_transform(X),y_scaler.inverse_transform(Y), color="green")
plt.plot(X_grid,y_scaler.inverse_transform(regressor.predict(x_scaler.transform(X_grid))),color="blue")
plt.title("salary vs level 2")
plt.xlabel("level")
plt.xlabel("salary")
plt.show()    
