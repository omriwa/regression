# import dataset
import pandas as pd

dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,-1].values

# splitting the dataset
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=0, test_size=0.2)

# import regression model
from sklearn.linear_model import LinearRegression as lr
regressor = lr()
regressor.fit(X_train,Y_train)

