import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

np.random.seed(42)
n = 200

area = np.random.uniform(500, 3500, n)
bedrooms = np.random.randint(1, 6, n)
age = np.random.uniform(0, 30, n)

price = (
    150 * area +
    10000 * bedrooms -
    2000 * age +
    np.random.normal(0, 20000, n)
)

df = pd.DataFrame({
    "Area": area,
    "Bedrooms": bedrooms,
    "Age": age,
    "Price": price
})

X = df[["Area", "Bedrooms", "Age"]]
y = df["Price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("R2 Score:", r2_score(y_test, y_pred))

plt.figure(figsize=(6,4))
plt.scatter(df["Area"], df["Price"], color="blue", alpha=0.6)
plt.xlabel("Area (sq ft)")
plt.ylabel("House Price")
plt.title("Area vs House Price")
plt.show()

plt.figure(figsize=(6,4))
plt.scatter(X_test["Area"], y_test, color="blue", label="Actual")
plt.scatter(X_test["Area"], y_pred, color="red", label="Predicted")
plt.xlabel("Area (sq ft)")
plt.ylabel("House Price")
plt.title("Actual vs Predicted (Area)")
plt.legend()
plt.show()

plt.figure(figsize=(6,4))
plt.scatter(y_test, y_pred, color="green")
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted House Price")
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    color="red"
)
plt.show()

residuals = y_test - y_pred
plt.figure(figsize=(6,4))
plt.scatter(y_pred, residuals, color="purple")
plt.axhline(y=0, color="red")
plt.xlabel("Predicted Price")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.show()
