import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

# Step 1: Create classification dataset
data = {
    'Hours_Studied': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Passed':        [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]  # 0 = Fail, 1 = Pass
}
df = pd.DataFrame(data)

# Step 2: Feature and target
X = df[['Hours_Studied']]
y = df['Passed']

# Step 3: Train KNN Classifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X, y)

# Step 4: Predict
X_vals = np.linspace(0, 11, 100).reshape(-1, 1)
y_preds = knn.predict(X_vals)

# Step 5: Print predictions
print("Hours Studied:\n", X_vals.flatten())
print("Predicted Class Labels:\n", y_preds)

# Step 6: Plot
plt.style.use('dark_background')
plt.scatter(X, y, color='lightblue', label='Actual Data')
plt.plot(X_vals, y_preds, color='yellow', label='KNN Classification')
plt.xlabel("Hours Studied")
plt.ylabel("Predicted Class (0 = Fail, 1 = Pass)")
plt.title("KNN Classification: Hours Studied vs. Pass/Fail")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()