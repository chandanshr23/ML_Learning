import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression 
dataset=pd.read_csv('C:/Users/2026/Desktop/ML/DataSet/Salary_Data.csv')

X=dataset.iloc[:, :-1].values
y= dataset.iloc[:, -1].values

X_train, X_Test,y_train, y_test=train_test_split(X,y,test_size=0.2,random_state=0)

#Linear Regression

regressor=LinearRegression()
regressor.fit(X_train,y_train)

#Predict 

y_pred=regressor.predict(X_Test)

#Plot for Train Data
plt.scatter(X_train,y_train,color='red')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title("Salary vs Expectations")
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

#Plot for Test Data
plt.scatter(X_Test,y_test,color='red')
plt.plot(X_Test,regressor.predict(X_train),color='blue')
plt.title("Salary vs Experience Test Set")
plt.xlabel("Year of Experience")
plt.ylabel("Salary")
plt.show()