import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV

# Custom features
feature1 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape(-1, 1)
feature2 = np.array([10, 9, 8, 7, 6, 5, 4, 3, 2, 1]).reshape(-1, 1)

# Combine features horizontally
X = np.hstack((feature1, feature2))

# Define target with both features contributing
# y = 3*feature1 + 2*feature2
y = 3 * feature1.flatten() + 2 * feature2.flatten()

# Ridge regression with built-in CV over alphas
alphas = [0.01, 0.1, 1.0, 10.0, 100.0]
model = RidgeCV(alphas=alphas, store_cv_values=True)
model.fit(X, y)

# Make prediction
sample = np.array([[11, 0]])
pred = model.predict(sample)

# Plotting (only feature1 vs target for visualization)
plt.style.use('dark_background')
plt.scatter(X[:, 0], y, color='blue', label='Data')
plt.scatter(sample[0, 0], pred, color='purple', label='Prediction')
plt.title("Ridge Regression with Cross-Validation")
plt.xlabel("Feature 1")
plt.ylabel("Target")
plt.legend()
plt.grid(True,linewidth=0.5,alpha=0.2)
plt.show()

print(f"Best alpha selected by RidgeCV: {model.alpha_}")
print(f"Prediction for sample {sample.tolist()[0]}: {pred[0]:.2f}")
print(f"Model coefficients: {model.coef_}")