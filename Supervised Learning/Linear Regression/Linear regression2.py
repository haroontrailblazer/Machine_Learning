from sklearn.metrics import mean_squared_error,r2_score
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as mp

#Generating random data for linear regression
np.random.seed(0) 
x = np.random.rand(100, 1) * 10 
y = 2.5 * x.squeeze() + np.random.randn(100) * 2

#Splitting the data into training and testing data
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

#Adding constant to the data
x_train_sm=sm.add_constant(x_train)
x_test_sm=sm.add_constant(x_test)

#Fitting the model
model=sm.OLS(y_train,x_train_sm).fit()

#Predicting the values
y_pred=model.predict(x_test_sm)

#Printing the model summary
print(model.summary())
#Calculating the mean squared error and r2 score
mse = mean_squared_error(y_test,y_pred)
r2score = r2_score(y_test,y_pred)
print(f'Mean_squared_error: {mse}')
print(f'r2_score: {r2score}')

#Plotting the model
mp.style.use('dark_background')
mp.scatter(x_train,y_train,color='pink',alpha=0.3,label='Dataset Values')
mp.plot(x_test,y_pred,linestyle='--',color='blue',label='Support line')
mp.legend()
mp.title('OLS regression model')
mp.show()