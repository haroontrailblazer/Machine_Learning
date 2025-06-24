import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree

# Step 1: Create Dataset
data = {
    'Hours_Studied': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Passed':        [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
}
df = pd.DataFrame(data)

# Step 2: Prepare Features and Target
X = df[['Hours_Studied']]
y = df['Passed']

# Step 3: Train Decision Tree Classifier
clf = DecisionTreeClassifier()
clf.fit(X, y)

# Step 4: Visualize Prediction Output
X_vals = np.linspace(0, 11, 100).reshape(-1, 1)
y_preds = clf.predict(X_vals)


# Step 5: Plot the Results
plt.style.use('dark_background')
plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X_vals, y_preds, color='red', linewidth=2, label='Decision Boundary')
plt.xlabel("Hours Studied")
plt.ylabel("Predicted Outcome")
plt.title("Decision Tree: Hours Studied vs. Passing Prediction")
plt.legend()
plt.grid(True,alpha=0.3)
plt.show()

# Optional: Visualize the Decision Tree Structure
plt.figure(figsize=(6,4))
plot_tree(clf, feature_names=['Hours_Studied'], class_names=['Fail', 'Pass'], filled=False)
plt.title("Decision Tree Structure")
plt.show()