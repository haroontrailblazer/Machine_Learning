# Polynomial regression with multiple features
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Sample data: Study hours and sleep schedule (hours per night)
study_hours = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape(-1, 1)
sleep_hours = np.array([4, 5, 6, 7, 8, 8, 7, 6, 5, 4]).reshape(-1, 1)
performance = np.array([50, 55, 60, 65, 70, 75, 80, 85, 90, 95])  # Example scores

# Combine features
X = np.hstack((study_hours, sleep_hours))

# Apply polynomial transformation
poly = PolynomialFeatures(degree=2)  # Quadratic polynomial regression
X_poly = poly.fit_transform(X)

# Train regression model
model = LinearRegression()
model.fit(X_poly, performance)

# Predict scores
input_features = np.array([[8, 6]])  # 8 study hours and 6 sleep hours
input_features_poly = poly.transform(input_features)
predicted_performance = model.predict(input_features_poly)  # Predict for 8 study hours and 6 sleep hours

# Plot results
plt.scatter(study_hours, performance, color='blue', marker='x', label="Actual Performance")
plt.scatter([8], predicted_performance, color='red', alpha=0.6, label="Predicted Performance (8,6)")
plt.xlabel("Study Hours")
plt.ylabel("Performance Score")
plt.legend()
plt.show()

# Display model coefficients
print("Polynomial Regression Coefficients:", model.coef_)
print("Intercept:", model.intercept_)
print("Polynomial Features:", poly.get_feature_names_out())
print("Predicted Performance:", predicted_performance)