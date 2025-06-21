import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import BayesianRidge

# Step 1: Create sample data
X = np.array([[i] for i in range(1, 11)])
y = np.array([3.1, 4.2, 5.0, 5.6, 6.1, 6.3, 7.0, 7.2, 7.5, 7.6])  # Slightly noisy

# Step 2: Initialize and train Bayesian Ridge model
model = BayesianRidge()
model.fit(X, y)

# Step 3: Predict for a new sample
sample = np.array([[11]])
pred = model.predict(sample)

# Step 4: Predict across a smooth range for plotting
X_range = np.linspace(1, 11, 200).reshape(-1, 1)
y_range, std_range = model.predict(X_range, return_std=True)

# Step 5: Plot
plt.style.use('dark_background')
plt.scatter(X, y, color='blue', label='Training Data')
plt.plot(X_range, y_range, color='darkred', label='Bayesian Prediction')
plt.fill_between(X_range.flatten(), y_range - std_range, y_range + std_range,
                 color='pink', alpha=0.4, label='Uncertainty (±1 std)')
plt.scatter(sample, pred, color='green', label='Prediction for x=11')
plt.title("Bayesian Linear Regression")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.show()

# Step 6: Output prediction
print(f"Prediction for x = {sample[0][0]}: {pred[0]:.2f}")