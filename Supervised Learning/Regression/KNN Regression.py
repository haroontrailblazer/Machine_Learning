import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor

# Step 1: Create the dataset
X = np.array([[i] for i in range(1, 11)])  # Features from 1 to 10
y = np.array([3, 4, 5, 5.5, 6, 6.2, 7, 7.1, 7.4, 7.5])  # Target with gentle curve

# Step 2: Initialize and train the KNN model
model = KNeighborsRegressor(n_neighbors=3)
model.fit(X, y)

# Step 3: Predict for a new input
sample = np.array([[11]])
pred = model.predict(sample)

# Step 4: Predict across a smooth range for plotting
X_range = np.linspace(1, 11, 200).reshape(-1, 1)
y_range = model.predict(X_range)

plt.style.use('dark_background')
plt.scatter(X, y, color='blue', label='Training Data')
plt.plot(X_range, y_range, color='magenta', label='KNN Prediction')
plt.scatter(sample, pred, color='green', label='Prediction for x=11')
plt.title("K-Nearest Neighbors Regression (k=3)")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.show()

# Step 5: Output prediction
print(f"Prediction for x = {sample[0][0]}: {pred[0]:.2f}")