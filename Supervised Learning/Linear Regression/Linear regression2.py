from sklearn.metrics import mean_squared_error,r2_score
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as mp
np.random.seed(0) 
x = np.random.rand(100, 1) * 10 
y = 2.5 * X.squeeze() + np.random.randn(100) * 2

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

x_train_sm=sm.add_constant(x_train)
x_test_sm=sm.add_constant(x_test)

model=sm.OLS(x_test_sm,y_train.fit())
y_pred=model.predict(x_test_sm)

mse = mean_squared_error(y_test,y_pred)
r2score = r2_score(y_test,y_pred)

print(f'Mean_squared_error: {mse}')
print(f'r2_score: {r2score}')

mp.style.use('dark_background')
mp.scatter(x_train,y_train,color='pink',alpha=0.3,label='Dataset Values')
mp.plot(x_test,y_pred,linestyle='--',color='blue',label='Support line')
mp.legend()
mp.title('OLS regression model')
mp.show()