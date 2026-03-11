# Import libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Dataset (multiple independent variables)
data = {
    "study_hours": [1,2,3,4,5,6,7,8],
    "attendance": [60,65,70,75,80,85,90,95],
    "marks": [50,55,60,65,70,75,80,85]
}

df = pd.DataFrame(data)

# Independent variables
X = df[["study_hours","attendance"]]

# Dependent variable
y = df["marks"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

# Model
model = LinearRegression()

# Train model
model.fit(X_train,y_train)

# Prediction
pred = model.predict(X_test)

# Output
print("Predicted values:", pred)
