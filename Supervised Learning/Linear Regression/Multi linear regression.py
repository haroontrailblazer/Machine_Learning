# Importing necessary libraries
import numpy as np 
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as mp
from mpl_toolkits.mplot3d import Axes3D

#creating a data frame
d={'r':[10,20,30,40,50],'t':[50,60,70,80,90],'n':[20,25,30,35,40],'s':[200,240,300,350,400]}
df=pd.DataFrame(d)

# assign values to x and y
x=df[['r','t','n']]
y=df['s']

# Splitting the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Creating a LinearRegression object and fitting the model
model = LinearRegression()
model.fit(x_train, y_train)

# Making predictions on the test set
y_pred = model.predict(x_test)

# Evaluating the model
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R^2 Score:", r2_score(y_test, y_pred))

# Plotting the results
mp.figure(figsize=(10, 6))
ax = mp.axes(projection='3d')
ax.scatter(x_test['r'], x_test['t'], y_test, color='red', label='Actual')
ax.scatter(x_test['r'], x_test['t'], y_pred, color='blue', label='Predicted')
ax.set_xlabel('r')
ax.set_ylabel('t')
ax.set_zlabel('s')
mp.title('3D Scatter Plot of Actual vs Predicted')
mp.legend()
mp.show()

# This code implements a multi-linear regression model using a custom LinearRegression class.