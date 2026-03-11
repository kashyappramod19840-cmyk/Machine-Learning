# Import libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Dataset
data = {
    "study_hours": [1,2,3,4,5,6,7,8],
    "attendance": [50,55,60,65,70,75,80,85],
    "pass": [0,0,0,0,1,1,1,1]
}

df = pd.DataFrame(data)

# Independent variables
X = df[["study_hours","attendance"]]

# Dependent variable
y = df["pass"]

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Model
model = LogisticRegression()

# Train model
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))
