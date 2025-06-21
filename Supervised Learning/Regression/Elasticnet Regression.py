import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import ElasticNetCV

# Step 1: Create manual feature data
feature1 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape(-1, 1)
feature2 = np.array([10, 9, 8, 7, 6, 5, 4, 3, 2, 1]).reshape(-1, 1)

# Step 2: Stack features side by side
X = np.hstack((feature1, feature2))

# Step 3: Create target with both features contributing
# For example: y = 2.5 * feature1 + 1.5 * feature2
y = 2.5 * feature1.flatten() + 1.5 * feature2.flatten()

# Step 4: Initialize ElasticNetCV (auto alpha + l1_ratio tuning)
model = ElasticNetCV(cv=5, l1_ratio=[.1, .5, .7, .9, .95, 1.0])
model.fit(X, y)

# Step 5: Prediction for a new sample
sample = np.array([[11, 0]])  # feature1 = 11, feature2 = 0
pred = model.predict(sample)

# Step 6: Visualize only Feature 1 vs Target for clarity
plt.style.use('dark_background')
plt.scatter(X[:, 0], y, color='blue', label='Feature 1 vs target')
plt.scatter(sample[0, 0], pred, color='orange', label='Prediction')
plt.title("Elastic Net Regression (Cross-Validated)")
plt.xlabel("Feature 1")
plt.ylabel("Target")
plt.legend()
plt.grid(True,alpha=0.2,linewidth=0.5)
plt.show()

# Step 7: Print results
print(f"Best alpha selected by ElasticNetCV: {model.alpha_:.4f}")
print(f"Best l1_ratio selected: {model.l1_ratio_}")
print(f"Prediction for sample {sample.tolist()[0]}: {pred[0]:.2f}")
print(f"Model coefficients: {model.coef_}")