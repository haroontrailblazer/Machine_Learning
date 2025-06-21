# 📊 Regression Models Summary

This repository contains a comprehensive walkthrough of various regression models implemented in Python using libraries like `scikit-learn`, `XGBoost`, `LightGBM`, `CatBoost`, and more. Each model is demonstrated with custom data, visualizations, and key parameter explanations.

---

## ✅ Models Covered

### 1. Linear Regression
- Assumes a straight-line relationship between features and target.
- Fast and interpretable, but limited to linear patterns.

### 2. Ridge Regression
- Adds **L2 regularization** via `alpha`.
- Shrinks coefficients to reduce overfitting.

### 3. Lasso Regression
- Adds **L1 regularization** via `alpha`.
- Performs feature selection by zeroing out less important coefficients.

### 4. ElasticNet Regression
- Combines L1 and L2 regularization.
- Key parameters: `alpha`, `l1_ratio`.

### 5. Support Vector Regression (SVR)
- Uses margins and kernels to fit nonlinear data.
- Key parameters: `C`, `epsilon`, `kernel`, `gamma`.

### 6. Decision Tree Regression
- Splits data into regions using if-else rules.
- Key parameters: `max_depth`, `min_samples_leaf`.

### 7. Random Forest Regression
- Ensemble of decision trees.
- Key parameters: `n_estimators`, `max_depth`.

### 8. HistGradientBoostingRegressor
- Fast, histogram-based boosting.
- Key parameters: `max_iter`, `learning_rate`.

### 9. XGBoost Regressor
- Gradient boosting with regularization.
- Key parameters: `n_estimators`, `learning_rate`, `max_depth`.

### 10. CatBoost Regressor
- Boosting with native categorical support.
- Key parameters: `iterations`, `learning_rate`, `depth`.

### 11. LightGBM Regressor
- Leaf-wise boosting for speed and accuracy.
- Key parameters: `n_estimators`, `learning_rate`, `max_depth`.

### 12. K-Nearest Neighbors Regression
- Predicts by averaging nearby points.
- Key parameter: `n_neighbors`.

### 13. Bayesian Ridge Regression
- Linear regression with priors.
- Returns prediction with uncertainty (mean ± std).

### 14. Gaussian Process Regression
- Non-parametric Bayesian model.
- Predicts a distribution over functions with smooth uncertainty bands.

---

## 🧠 Key Concepts & Parameters

- `alpha`: Regularization strength (Ridge, Lasso, ElasticNet).
- `l1_ratio`: Balance between L1 and L2 (ElasticNet).
- `max_depth`: Tree complexity control.
- `n_estimators`: Number of trees in ensemble models.
- `learning_rate`: Step size in boosting.
- `C`, `epsilon`, `gamma`: SVR tuning knobs.
- `n_neighbors`: Number of neighbors in KNN.
- `return_std=True`: Enables uncertainty in Bayesian models.

---

## 🧾 Final Thoughts

Each model has its own **reliability and strengths**. There are no universal disadvantages — only **different fits for different data types, sizes, and goals**. Whether you're working with small, noisy datasets or large, structured ones, there's a regression model that fits the moment.

---

## 📂 Files

You can find Git documentation files on your system here: [git-cat-file.html](file:///C:/Users/haroo/Git/mingw64/share/doc/git-doc/git-cat-file.html), [git-diff-files.html](file:///C:/Users/haroo/Git/mingw64/share/doc/git-doc/git-diff-files.html), [git-ls-files.html](file:///C:/Users/haroo/Git/mingw64/share/doc/git-doc/git-ls-files.html), [git-merge-file.html](file:///C:/Users/haroo/Git/mingw64/share/doc/git-doc/git-merge-file.html) and [git-merge-one-file.html](file:///C:/Users/haroo/Git/mingw64/share/doc/git-doc/git-merge-one-file.html).

---
