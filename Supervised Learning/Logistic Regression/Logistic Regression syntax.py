import numpy as np
from sklearn.linear_model import LogisticRegression

# Example data: x=sale,demand, y=sucess/failure
X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
y = np.array([0, 0, 1, 1])

# Create a logistic regression model
model = LogisticRegression()

# Fit the model to the data
model.fit(X, y)

# Make predictions
predictions = model.predict([[1.5, 2.5], [3.5, 4.5]])

# Print the predictions
print("Predictions for [1.5, 2.5] is: ", predictions[0])
print("Predictions for [3.5, 4.5] is: ", predictions[1])