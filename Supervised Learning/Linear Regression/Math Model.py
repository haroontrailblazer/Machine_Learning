import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Create Sample Dataset
x = np.array([1500,750,1000,1250,1500,1750,2000,2250,2500,2750])
y = np.array([100,150,200,230,270,310,340,38,420,450])

mean_x = np.mean(x)
mean_y = np.mean(y)

# Calculate the slope (m) and intercept (b) of the line
m = np.sum(x-mean_x * y-mean_y) / np.sum((x-mean_x)**2)
b = mean_y - m * mean_x

# Create the regression line
regression_line = m * x + b

# Prediction for new data points
new_x = np.array([1600, 1800, 2200])

# Plot the data points and the regression line
plt.style.use('dark_background')
plt.scatter(x, y, color='blue', label='Data Points')
plt.plot(x, regression_line, color='red', label='Regression Line')
plt.scatter(new_x, m * new_x + b, color='green', label='Predictions', marker='x')
plt.scatter(mean_x, mean_y, color='orange', label='Mean Point', marker='o')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Simple Linear Regression')
plt.legend()
plt.show()

# Print the slope and intercept
print(f"Slope (m): {m}")
print(f"Intercept (b): {b}\n")

# Print the mean of x and y
print(f"Mean of x: {mean_x}")
print(f"Mean of y: {mean_y}\n")

# Print the regression line values
print(f"Regression Line: {regression_line}\n")

# Print the correlation coefficient
correlation_coefficient = np.corrcoef(x, y)[0, 1]
print(f"Correlation Coefficient: {correlation_coefficient}")

# Print the covariance
covariance = np.cov(x, y)[0, 1]
print(f"Covariance: {covariance}")

# Print the R-squared value
r_squared = correlation_coefficient ** 2
print(f"R-squared: {r_squared}")