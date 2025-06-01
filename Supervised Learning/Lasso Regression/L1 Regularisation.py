import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso, LassoCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Generate a simple regression dataset
np.random.seed(42)
X = np.linspace(0, 10, 100).reshape(-1, 1)
y = 3 * X.squeeze() + np.random.randn(100) * 2

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# LassoCV will automatically find the best alpha using cross-validation
lasso_cv = LassoCV(alphas=[0.01, 0.1, 1, 10], cv=5)
lasso_cv.fit(X_train, y_train)
print("Best alpha:", lasso_cv.alpha_)

# Create and train the Lasso regression model
lasso = Lasso(alpha=lasso_cv.alpha_)
lasso.fit(X_train, y_train)

# Predict
y_pred = lasso.predict(X_test)

# Evaluate
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")

# Plot
plt.scatter(X, y, color='blue', label='Data')
plt.plot(X_test, y_pred, color='red', label='Lasso Prediction')
plt.xlabel("X")
plt.ylabel("y")
plt.title("Lasso Regression")
plt.legend()
plt.show()