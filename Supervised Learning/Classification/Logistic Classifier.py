import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

# Step 1: Sample Data
data = {
    'Hours_Studied': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Passed':        [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
}
df = pd.DataFrame(data)

# Step 2: Prepare Features and Labels
X = df[['Hours_Studied']]  # Feature must be 2D
y = df['Passed']

# Step 3: Train Model Directly
model = LogisticRegression()
model.fit(X, y)

# Step 4: Visualize Predictions
X_vals = np.linspace(0, 11, 100).reshape(-1, 1)
y_probs = model.predict_proba(X_vals)[:, 1]

# Step 5: Plotting
plt.style.use('dark_background')
plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X_vals, y_probs, color='red', label='Logistic Curve')
plt.xlabel("Hours Studied")
plt.ylabel("Probability of Passing")
plt.title("Logistic Regression: Hours Studied vs Passing Probability")
plt.legend()
plt.grid(True,alpha=0.3,linewidth=0.5)
plt.show()