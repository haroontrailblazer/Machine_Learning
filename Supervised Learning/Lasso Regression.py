import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LassoCV

# Create custom features
feature1 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape(-1, 1)
feature2 = np.array([10, 9, 8, 7, 6, 5, 4, 3, 2, 1]).reshape(-1, 1)

# Stack features horizontally
X = np.hstack((feature1, feature2))

# Define target with a sparse linear relationship
# Let's simulate y = 3 * feature1 (feature2 is noise)
y = 3 * feature1.flatten()

# Use LassoCV to automatically choose best alpha
model = LassoCV(cv=5)
model.fit(X, y)

# Prediction for a new data point
sample = np.array([[11, 0]])
pred = model.predict(sample)

# Visualization
plt.style.use('dark_background')
plt.scatter(X[:, 0], y, color='blue', label='Data')
plt.scatter(sample[0, 0], pred, color='green', label='Prediction')
plt.title("Lasso Regression with Cross-Validation")
plt.xlabel("Feature 1")
plt.ylabel("Target")
plt.legend()
plt.grid(True,alpha=0.2,linewidth=0.5)
plt.show()

print(f"Best alpha selected by LassoCV: {model.alpha_:.4f}")
print(f"Prediction for sample {sample.tolist()[0]}: {pred[0]:.2f}")
print(f"Model coefficients: {model.coef_}")