import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Dataset
X = np.array([[1],[2],[3],[4],[5]])   # Input (Study hours)
y = np.array([2,4,6,8,10])            # Output (Marks)

# Train and Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Model
model = LinearRegression()

# Training
model.fit(X_train, y_train)

# Prediction
prediction = model.predict(X_test)

print("Predicted values:", prediction)
