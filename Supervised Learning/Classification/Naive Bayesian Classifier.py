import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB

# Step 1: Dataset
data = {
    'Hours_Studied': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Passed':        [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
}
df = pd.DataFrame(data)
X = df[['Hours_Studied']]
y = df['Passed']

# Step 2: Train the model
model = GaussianNB()
model.fit(X, y)

# Step 3: Predict on new values (without loop)
X_vals = np.linspace(0, 11, 100).reshape(-1, 1)
y_preds = model.predict(X_vals)

# Step 4: Print all predictions at once
print("Hours Studied Range:\n", X_vals.flatten())
print("Predicted Labels:\n", y_preds)

# Step 5: Plot
plt.style.use('dark_background')
plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X_vals, y_preds, color='crimson', label='Naive Bayes Prediction')
plt.xlabel("Hours Studied")
plt.ylabel("Predicted Class")
plt.title("Naive Bayes Classification")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.2)
plt.show()