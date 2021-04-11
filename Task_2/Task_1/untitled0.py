# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 14:28:29 2021

@author: User
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv('marks_data.csv')
data.head()

data.info()
data.describe()
data.shape
plt.scatter(data['Hours'], data['Scores'])
plt.xlabel("Number of Hours")
plt.ylabel("Scores")
plt.title("Hours vs Scores")
plt.show()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data['Hours'].values.reshape(-1,1), data['Scores'], test_size = 0.2, random_state = 42)
X_train.shape, y_train.shape, X_test.shape, y_test.shape
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)
coefficient = model.coef_
intercept = model.intercept_

# Since, y = m*x + c
line = (data['Hours'].values * coefficient) + intercept
plt.scatter(data.Hours, data.Scores)
plt.plot(data.Hours, line)
plt.show()
pred = model.predict(X_test)
pred
pred_compare = pd.DataFrame({'Actual Values': y_test, 'Predicted Values':pred})
pred_compare
from sklearn import metrics
print("Mean Absolute Error: ", metrics.mean_absolute_error(y_test, pred))
print("Mean Squared Error: ", metrics.mean_squared_error(y_test, pred))
print("Root Mean Squared Error: ", metrics.mean_squared_error(y_test, pred)**0.5))
print("R2 Score: ", metrics.r2_score(y_test, pred)
