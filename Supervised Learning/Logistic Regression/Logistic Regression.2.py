from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt

# Example dataset (X: features, y: labels)
X = np.array([[1], [2], [3], [4], [5]])  # Features
y = np.array([0, 0, 1, 1, 1])            # Labels

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Logistic Regression model
model = LogisticRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Plot the data points
plt.scatter(X, y, color='blue', label='Data Points')

# Generate a range of values for the decision boundary
X_range = np.linspace(X.min() - 1, X.max() + 1, 100).reshape(-1, 1)
y_prob = model.predict_proba(X_range)[:, 1]  # Probability of class 1

# Plot the decision boundary
plt.plot(X_range, y_prob, color='red', label='Decision Boundary')
plt.axhline(0.5, color='green', linestyle='--', label='Threshold (0.5)')
plt.xlabel("Feature")
plt.ylabel("Probability")
plt.title("Logistic Regression Decision Boundary")
plt.legend()
plt.show()