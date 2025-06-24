import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

# Step 1: Dataset
data = {
    'Hours_Studied': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Passed':        [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
}
df = pd.DataFrame(data)

# Step 2: Prepare Input and Output
X = df[['Hours_Studied']]
y = df['Passed']

# Step 3: Train Random Forest
rf = RandomForestClassifier(n_estimators=10, random_state=42)
rf.fit(X, y)

# Step 4: Visualize Predictions
X_vals = np.linspace(0, 11, 100).reshape(-1, 1)
y_preds = rf.predict(X_vals)

plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X_vals, y_preds, color='purple', linewidth=2, label='Random Forest Prediction')
plt.xlabel("Hours Studied")
plt.ylabel("Predicted Outcome")
plt.title("Random Forest: Hours Studied vs. Pass Prediction")
plt.legend()
plt.grid(True, alpha=0.3, linestyle='--')
plt.show()