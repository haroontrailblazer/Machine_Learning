from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np

# Quadratic data
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 5, 10, 17, 26])  # roughly y = x² + 1

# Model pipeline
model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
model.fit(X, y)

# Prediction
pred = model.predict([[6]])

# Visualization
X_range = np.linspace(1, 6, 100).reshape(-1, 1)
y_range = model.predict(X_range)
plt.scatter(X, y, color='blue', label='Data')
plt.plot(X_range, y_range, color='red', label='Polynomial Fit')
plt.scatter(6, pred, color='green', label='Prediction')
plt.title("Polynomial Regression (degree 2)")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.show()

print(f"Prediction for x=6: {pred[0]:.2f}")