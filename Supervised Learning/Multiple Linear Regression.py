import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Custom data (2 features, 10 samples)
feature1 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape(-1, 1)
feature2 = np.array([10, 9, 8, 7, 6, 5, 4, 3, 2, 1]).reshape(-1, 1)

# Combine the features
X = np.hstack((feature1, feature2))

# Target variable (some custom linear relationship)
# For example: y = 2*feature1 + 3*feature2
y = 2 * feature1.flatten() + 3 * feature2.flatten()

# Train model
model = LinearRegression()
model.fit(X, y)

# Predict for a new sample
sample = np.array([[11, 0]])  # e.g., feature1=11, feature2=0
pred = model.predict(sample)

# Visualization: Just show relation of feature1 vs target
plt.style.use('dark_background')
plt.scatter(X[:, 0], y, color='blue', label='Feature 1 vs target')
plt.scatter(sample[0, 0], pred, color='green', label='Prediction')
plt.title("Multiple Linear Regression (Manual Data)")
plt.xlabel("Feature 1")
plt.ylabel("Target")
plt.legend()
plt.grid(True)
plt.show()

print(f"Prediction for sample {sample.tolist()[0]}: {pred[0]:.2f}")