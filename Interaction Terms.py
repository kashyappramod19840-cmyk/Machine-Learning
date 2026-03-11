# Import libraries
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Dataset
data = {
    "study_hours": [1,2,3,4,5,6,7,8],
    "attendance": [60,65,70,75,80,85,90,95],
    "marks": [50,55,60,65,70,75,80,85]
}

df = pd.DataFrame(data)

# Create interaction term
df["interaction"] = df["study_hours"] * df["attendance"]

# Independent variables
X = df[["study_hours", "attendance", "interaction"]]

# Dependent variable
y = df["marks"]

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Model
model = LinearRegression()

# Train model
model.fit(X_train, y_train)

# Prediction
pred = model.predict(X_test)

# Output
print("Predicted values:", pred)
