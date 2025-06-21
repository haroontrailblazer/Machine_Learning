import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor


# Random Forest Regressor Example
# Step 1: Define input feature and target manually
X = np.array([[i] for i in range(1, 11)])  # Feature values 1 to 10
y = np.array([3, 4, 5, 5.5, 6, 6.2, 7, 7.1, 7.4, 7.5])  # Target values

# Step 2: Initialize and train the model
model = RandomForestRegressor(n_estimators=100, max_depth=4, random_state=42)
model.fit(X, y)

# Step 3: Predict for a new value
sample = np.array([[11]])
pred = model.predict(sample)

# Step 4: Plot predictions across a smooth input range
X_range = np.linspace(1, 11, 200).reshape(-1, 1)
y_range = model.predict(X_range)

# Plotting the results
plt.style.use('dark_background')
plt.scatter(X, y, color='blue', label='Training Data')
plt.plot(X_range, y_range, color='red', label='Random Forest Prediction')
plt.scatter(sample, pred, color='green', label='Prediction for x=11')
plt.title("Random Forest Regression")
plt.xlabel("Feature")
plt.ylabel("Target")
plt.legend()
plt.grid(True)
plt.show()

# Step 5: Output prediction
print(f"Prediction for x = {sample[0][0]}: {pred[0]:.2f}")



# histGradiantBoostingRegressor Example
from sklearn.ensemble import HistGradientBoostingRegressor

# Training data
X = np.array([[i] for i in range(1, 11)])
y = np.array([3, 4, 5, 5.5, 6, 6.2, 7, 7.1, 7.4, 7.5])

# Model
model = HistGradientBoostingRegressor(max_iter=100, learning_rate=0.1)
model.fit(X, y)

# Predict for a new sample
sample = np.array([[11]])
pred = model.predict(sample)

# Plot
X_range = np.linspace(1, 11, 200).reshape(-1, 1)
y_range = model.predict(X_range)

plt.style.use('dark_background')
plt.scatter(X, y, color='blue', label='Training Data')
plt.plot(X_range, y_range, color='red', label='HistGradientBoosting')
plt.scatter(sample, pred, color='green', label='Prediction')
plt.title("HistGradientBoosting Regressor")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.show()

print(f"Prediction for x = 11: {pred[0]:.2f}")


# Xg boost Regressor Example
from xgboost import XGBRegressor

# Training data
X = np.array([[i] for i in range(1, 11)])
y = np.array([3, 4, 5, 5.5, 6, 6.2, 7, 7.1, 7.4, 7.5])

# Model
model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3)
model.fit(X, y)

# Predict for a new sample
sample = np.array([[11]])
pred = model.predict(sample)

# Plot
X_range = np.linspace(1, 11, 200).reshape(-1, 1)
y_range = model.predict(X_range)

plt.style.use('dark_background')
plt.scatter(X, y, color='blue', label='Training Data')
plt.plot(X_range, y_range, color='green', label='XGBoost')
plt.scatter(sample, pred, color='orange', label='Prediction')
plt.title("XGBoost Regressor")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.show()

print(f"Prediction for x = 11: {pred[0]:.2f}")


# CatBoost Regressor Example
from catboost import CatBoostRegressor

# Training data
X = np.array([[i] for i in range(1, 11)])
y = np.array([3, 4, 5, 5.5, 6, 6.2, 7, 7.1, 7.4, 7.5])

# Model
model = CatBoostRegressor(iterations=100, learning_rate=0.1, depth=3, verbose=0)
model.fit(X, y)

# Predict for a new sample
sample = np.array([[11]])
pred = model.predict(sample)

# Plot
X_range = np.linspace(1, 11, 200).reshape(-1, 1)
y_range = model.predict(X_range)

plt.style.use('dark_background')
plt.scatter(X, y, color='blue', label='Training Data')
plt.plot(X_range, y_range, color='purple', label='CatBoost')
plt.scatter(sample, pred, color='red', label='Prediction')
plt.title("CatBoost Regressor")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.show()

print(f"Prediction for x = 11: {pred[0]:.2f}")


# LightGBM Regressor Example
import lightgbm as lgb

# Step 1: Create sample data
X = np.array([[i] for i in range(1, 11)])  # Features from 1 to 10
y = np.array([3, 4, 5, 5.5, 6, 6.2, 7, 7.1, 7.4, 7.5])  # Slightly nonlinear target

# Step 2: Initialize and train the LightGBM model
model = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.1, max_depth=3)
model.fit(X, y)

# Step 3: Predict for a new sample
sample = np.array([[11]])
pred = model.predict(sample)

# Step 4: Plot predictions across a smooth input range
X_range = np.linspace(1, 11, 200).reshape(-1, 1)
y_range = model.predict(X_range)

plt.style.use('dark_background')
plt.scatter(X, y, color='blue', label='Training Data')
plt.plot(X_range, y_range, color='darkorange', label='LightGBM Prediction')
plt.scatter(sample, pred, color='green', label='Prediction for x=11')
plt.title("LightGBM Regression")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.show()

# Step 5: Output prediction
print(f"Prediction for x = {sample[0][0]}: {pred[0]:.2f}")