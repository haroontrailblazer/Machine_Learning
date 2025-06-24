import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC

# Step 1: Sample Data
data = {
    'Hours_Studied': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Passed':        [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
}
df = pd.DataFrame(data)

# Step 2: Feature and Target
X = df[['Hours_Studied']]
y = df['Passed']

# Step 3: Train SVM Classifier (with linear kernel)
svm = SVC(kernel='linear', probability=True)
svm.fit(X, y)

# Step 4: Predict on new values and print results
X_vals = np.linspace(0, 11, 100).reshape(-1, 1)
y_preds = svm.predict(X_vals)

print("Predicted values for Hours_Studied from 0 to 11:")
for h, pred in zip(X_vals.flatten(), y_preds):
    print(f"Hours: {h:.2f} => Predicted: {pred}")

# Step 5: Visualization
plt.style.use('dark_background')
plt.scatter(X, y, color='blue', label='Original Data')
plt.plot(X_vals, y_preds, color='orange', label='SVM Prediction')
plt.xlabel("Hours Studied")
plt.ylabel("Predicted Class")
plt.title("SVM Classification: Hours Studied vs. Pass Prediction")
plt.legend()
plt.grid(True,alpha=0.5)
plt.show()

# Train SVM with RBF kernel
svm_rbf = SVC(kernel='rbf', probability=True)
svm_rbf.fit(X, y)

# Predictions
X_vals = np.linspace(0, 11, 100).reshape(-1, 1)
y_preds_rbf = svm_rbf.predict(X_vals)

print("RBF Kernel Predictions:")
for h, pred in zip(X_vals.flatten(), y_preds_rbf):
    print(f"Hours: {h:.2f} => Predicted: {pred}")

# Plot
plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X_vals, y_preds_rbf, color='darkgreen', label='RBF SVM')
plt.title("SVM with RBF Kernel")
plt.xlabel("Hours Studied")
plt.ylabel("Predicted Class")
plt.legend()
plt.grid(True,alpha=0.5)
plt.show()

# Train SVM with Polynomial kernel (degree 3 by default)
svm_poly = SVC(kernel='poly', degree=3, probability=True)
svm_poly.fit(X, y)

# Predictions
y_preds_poly = svm_poly.predict(X_vals)

print("\nPolynomial Kernel Predictions:")
for h, pred in zip(X_vals.flatten(), y_preds_poly):
    print(f"Hours: {h:.2f} => Predicted: {pred}")

# Plot
plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X_vals, y_preds_poly, color='maroon', label='Polynomial SVM')
plt.title("SVM with Polynomial Kernel")
plt.xlabel("Hours Studied")
plt.ylabel("Predicted Class")
plt.legend()
plt.grid(True, alpha=0.5)
plt.show()