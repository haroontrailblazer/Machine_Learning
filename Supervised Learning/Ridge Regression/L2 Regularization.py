import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Generate a simple regression dataset
np.random.seed(42)
X = np.linspace(0, 10, 100).reshape(-1, 1)
y = 3 * X.squeeze() + np.random.randn(100) * 2

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# RidgeCV will automatically find the best alpha using cross-validation
ridge_cv = RidgeCV(alphas=[0.01, 0.1, 1, 10], cv=5)
ridge_cv.fit(X_train, y_train)
print("Best alpha:", ridge_cv.alpha_)

# Create and train the Ridge regression model
ridge = Ridge(alpha=ridge_cv.alpha_)
ridge.fit(X_train, y_train)

# Predict
y_pred = ridge.predict(X_test)

# Evaluate
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")

# Plot
plt.scatter(X, y, color='blue', label='Data')
plt.plot(X_test, y_pred, color='red', label='Ridge Prediction')
plt.xlabel("X")
plt.ylabel("y")
plt.title("Ridge Regression")
plt.legend()
plt.show()