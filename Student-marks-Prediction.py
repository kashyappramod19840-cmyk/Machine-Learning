import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

np.random.seed(42)
n = 200

study_hours = np.random.uniform(1, 10, n)
attendance = np.random.uniform(50, 100, n)
previous_marks = np.random.uniform(40, 95, n)

final_marks = (
    5 * study_hours +
    0.3 * attendance +
    0.5 * previous_marks +
    np.random.normal(0, 5, n)
)

df = pd.DataFrame({
    "Study_Hours": study_hours,
    "Attendance": attendance,
    "Previous_Marks": previous_marks,
    "Final_Marks": final_marks
})


X = df[["Study_Hours", "Attendance", "Previous_Marks"]]
y = df["Final_Marks"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("R2 Score:", r2_score(y_test, y_pred))

plt.figure(figsize=(6,4))
plt.scatter(df["Study_Hours"], df["Final_Marks"], color="blue", alpha=0.6)
plt.xlabel("Study Hours")
plt.ylabel("Final Marks")
plt.title("Study Hours vs Final Marks")
plt.show()


plt.figure(figsize=(6,4))
plt.scatter(X_test["Study_Hours"], y_test, color="blue", label="Actual")
plt.scatter(X_test["Study_Hours"], y_pred, color="red", label="Predicted")
plt.xlabel("Study Hours")
plt.ylabel("Final Marks")
plt.title("Actual vs Predicted (Study Hours)")
plt.legend()
plt.show()

plt.figure(figsize=(6,4))
plt.scatter(y_test, y_pred, color="green")
plt.xlabel("Actual Marks")
plt.ylabel("Predicted Marks")
plt.title("Actual vs Predicted Marks")
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         color='red')   
plt.show()


residuals = y_test - y_pred
plt.figure(figsize=(6,4))
plt.scatter(y_pred, residuals, color="purple")
plt.axhline(y=0, color='red')
plt.xlabel("Predicted Marks")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.show()
