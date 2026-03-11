# Import libraries
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Dataset
X = np.array([[1],[2],[3],[4],[5],[6],[7],[8],[9],[10]])
y = np.array([2,4,6,8,10,12,14,16,18,20])

# Model
model = LinearRegression()

# KFold
kf = KFold(n_splits=5)

scores = []

for train_index, test_index in kf.split(X):
    
    # Split data
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Train model
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Score
    score = r2_score(y_test, y_pred)
    scores.append(score)

# Print scores
print("R2 Scores for each fold:", scores)
print("Average Score:", np.mean(scores))
