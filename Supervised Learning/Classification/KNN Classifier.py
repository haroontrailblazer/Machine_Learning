import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor

# Step 1: Create the dataset
data = {
    'Hours_Studied': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Exam_Marks':    [33, 35, 45, 50, 60, 65, 70, 75, 85, 90]
}
df = pd.DataFrame(data)

# Step 2: Feature and target
X = df[['Hours_Studied']]
y = df['Exam_Marks']

# Step 3: Create and train KNN regressor
knn = KNeighborsRegressor(n_neighbors=3)
knn.fit(X, y)

# Step 4: Predict
X_vals = np.linspace(0, 11, 100).reshape(-1, 1)
y_preds = knn.predict(X_vals)

# Step 5: Print results as arrays
print("Hours Studied:\n", X_vals.flatten())
print("Predicted Exam Marks:\n", y_preds)

# Step 6: Plot
plt.style.use('dark_background')
plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X_vals, y_preds, color='orange', label='KNN Prediction')
plt.xlabel("Hours Studied")
plt.ylabel("Predicted Exam Marks")
plt.title("KNN Regression: Hours vs. Marks")
plt.legend()
plt.grid(True,alpha=0.3)
plt.show()