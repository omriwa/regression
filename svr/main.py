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


# plt.title("salary vs level")
# plt.xlabel("level")
# plt.xlabel("salary")
# plt.show()
