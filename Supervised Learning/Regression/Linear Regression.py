import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Sample data
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([3, 4, 2, 5, 6])

# Model
model = LinearRegression()
model.fit(X, y)

# Prediction
pred = model.predict([[6]])

# Visualization
plt.style.use('dark_background')
plt.scatter(X, y, color='blue', label='Data')
plt.plot(X, model.predict(X), color='red', label='Regression Line')
plt.scatter(6, pred, color='green', label='Prediction')
plt.legend()
plt.title("Linear Regression")
plt.xlabel("X")
plt.ylabel("y")
plt.show()

print(f"Prediction for x=6: {pred[0]:.2f}")