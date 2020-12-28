# import dataset
import pandas as pd

dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,-1].values

# hot encoding
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import numpy as np

ct = ColumnTransformer(transformers=[("encoder",OneHotEncoder(),[0])],remainder="passthrough")
X = np.array(ct.transform(X))
# splitting the dataset
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=0, test_size=0.2)

# import regression model
from sklearn.linear_model import LinearRegression as lr

regressor = lr()
regressor.fit(X_train,Y_train)

#  Test
y_pred = regressor.predict(X_test)

# Visualsing
import  matplotlib.pyplot as plt

plt.scatter(X_train,Y_train, color='red')
plt.scatter(X_test,y_pred, color="green")
plt.plot(X_train,regressor.predict(X_train),color="blue")
plt.title("salary vs experience")
plt.xlabel("years of experience")
plt.xlabel("salary")
plt.show()
