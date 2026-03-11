import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Dataset
X = np.array([[1],[2],[3],[4],[5]])
y = np.array([2,4,5,4,5])

# Model create
model = LinearRegression()

# Model train
model.fit(X, y)

# Prediction
y_pred = model.predict(X)

# Print result
print("Predicted values:", y_pred)

# Graph
plt.scatter(X, y)
plt.plot(X, y_pred)
plt.title("Simple Linear Regression")
plt.xlabel("Input")
plt.ylabel("Output")
plt.show()
