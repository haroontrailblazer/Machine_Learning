import numpy as np
from sklearn.tree import DecisionTreeRegressor

# Sample data
X = np.array([[i] for i in range(1, 11)])
y = np.array([3, 4, 5, 6, 6.5, 7, 7.5, 7.5, 8, 8])

# Initialize model with controlled depth and leaf size
model = DecisionTreeRegressor(max_depth=3, min_samples_leaf=2)
model.fit(X, y)

# Predict for a new value
sample = np.array([[11]])
pred = model.predict(sample)

print(f"Prediction for x = {sample[0][0]}: {pred[0]:.2f}")
print(f"Model depth: {model.get_depth()}, Number of leaves: {model.get_n_leaves()}")