# Import libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Input dataset
X = np.array([[1],[2],[3],[4],[5]])
y = np.array([1,4,9,16,25])

# Apply non-linear transformation (Polynomial)
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

# Create model
model = LinearRegression()

# Train model
model.fit(X_poly, y)

# Prediction
y_pred = model.predict(X_poly)

# Print predicted values
print("Predicted values:", y_pred)

# Plot original data
plt.scatter(X, y)

# Plot non-linear curve
plt.plot(X, y_pred)

# Labels
plt.title("Non Linear Transformation (Polynomial Regression)")
plt.xlabel("Input")
plt.ylabel("Output")

# Show graph
plt.show()
