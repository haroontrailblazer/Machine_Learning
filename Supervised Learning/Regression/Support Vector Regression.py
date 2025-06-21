import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR

# 1. Linear Kernal
# Step 1: Create manual feature and target data
X = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])
y = np.array([3, 4, 5, 6, 7.5, 8, 9, 9.5, 10, 10.2])  # Mildly nonlinear

# Step 2: Initialize SVR model with linear kernel
model = SVR(kernel='linear', C=100.0, epsilon=0.2)
model.fit(X, y)

# Step 3: Predict for a new data point
sample = np.array([[11]])
pred = model.predict(sample)

# Step 4: Visualize model prediction
X_range = np.linspace(1, 11, 100).reshape(-1, 1)
y_range = model.predict(X_range)

# Plotting the results
plt.style.use('dark_background')
plt.scatter(X, y, color='blue', label='Original data')
plt.plot(X_range, y_range, color='red', label='SVR prediction line')
plt.scatter(sample[0, 0], pred, color='green', label='Prediction')
plt.title("Support Vector Regression (SVR - Linear Kernel)")
plt.xlabel("Feature")
plt.ylabel("Target")
plt.legend()
plt.grid(True,alpha=0.3,linewidth=0.4)
plt.show()

# Step 5: Output results
print(f"Prediction for x = {sample.flatten()[0]}: {pred[0]:.2f}")
print(f"Model coefficients: {model.coef_}")

# 2. RBF kernel SVR
X = np.array([[i] for i in range(1, 11)])
y = np.sin(X).ravel() + 0.1 * np.random.randn(10)  # noisy sine pattern

# Fit SVR with RBF kernel
model = SVR(kernel='rbf', C=100, gamma=0.5, epsilon=0.1)
model.fit(X, y)

# Predict a new value
sample = np.array([[11]])
pred = model.predict(sample)

# Plot the model
X_range = np.linspace(1, 11, 100).reshape(-1, 1)
y_range = model.predict(X_range)

# Plotting the results
plt.style.use('dark_background')
plt.scatter(X, y, color='blue', label='Training data')
plt.plot(X_range, y_range, color='red', label='SVR (RBF kernel)')
plt.scatter(sample, pred, color='green', label='Prediction')
plt.title("SVR with RBF Kernel")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.show()

print(f"Prediction at x = {sample.flatten()[0]}: {pred[0]:.2f}")


# 3. Polynomial kernel SVR
# Fit SVR with Polynomial kernel
model = SVR(kernel='poly', degree=3, C=100, epsilon=0.1, coef0=1)
model.fit(X, y)

# Predict a new value
pred_poly = model.predict(sample)

# Plot the model
y_range_poly = model.predict(X_range)

plt.scatter(X, y, color='blue', label='Training data')
plt.plot(X_range, y_range_poly, color='purple', label='SVR (Polynomial kernel)')
plt.scatter(sample, pred_poly, color='orange', label='Prediction')
plt.title("SVR with Polynomial Kernel (degree 3)")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.show()

print(f"Prediction at x = {sample.flatten()[0]}: {pred_poly[0]:.2f}")