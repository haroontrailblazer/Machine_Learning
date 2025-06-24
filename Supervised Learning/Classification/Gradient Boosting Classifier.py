import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Dataset
data = {
    'Hours_Studied': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Passed':        [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
}
df = pd.DataFrame(data)
X = df[['Hours_Studied']]
y = df['Passed']
X_vals = np.linspace(0, 11, 100).reshape(-1, 1)

# AdaBoost Classifier
from sklearn.ensemble import AdaBoostClassifier
ada = AdaBoostClassifier(n_estimators=50, random_state=0)
ada.fit(X, y)
y_preds_ada = ada.predict(X_vals)
plt.plot(X_vals, y_preds_ada, label='AdaBoost', color='orange')

# XGBoost Classifier
from xgboost import XGBClassifier
xgb = XGBClassifier(n_estimators=100, learning_rate=0.1, use_label_encoder=False, eval_metric='logloss')
xgb.fit(X, y)
y_preds_xgb = xgb.predict(X_vals)
plt.plot(X_vals, y_preds_xgb, label='XGBoost', color='green')

# LightGBM Classifier
from lightgbm import LGBMClassifier
lgb = LGBMClassifier(n_estimators=100, learning_rate=0.1, random_state=0)
lgb.fit(X, y)
y_preds_lgb = lgb.predict(X_vals)
plt.plot(X_vals, y_preds_lgb, label='LightGBM', color='blue')

# CatBoost Classifier
from catboost import CatBoostClassifier
cat = CatBoostClassifier(iterations=100, learning_rate=0.1, verbose=0)
cat.fit(X, y)
y_preds_cat = cat.predict(X_vals)
plt.plot(X_vals, y_preds_cat, label='CatBoost', color='purple')

# Plot
plt.style.use('dark_background')
plt.scatter(X, y, color='white', label='Actual Data')
plt.xlabel("Hours Studied")
plt.ylabel("Predicted Class (0 = Fail, 1 = Pass)")
plt.title("Gradient Boosting Classifiers")
plt.legend()
plt.grid(True, alpha=0.2)
plt.show()