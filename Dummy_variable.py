import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

np.random.seed(42)
n = 200

area = np.random.uniform(500, 3500, n)
bedrooms = np.random.randint(1, 6, n)
city = np.random.choice(["Delhi", "Mumbai", "Bangalore"], n)

price = (
    120 * area +
    15000 * bedrooms +
    np.where(city == "Delhi", 50000, 0) +
    np.where(city == "Mumbai", 80000, 0) +
    np.random.normal(0, 20000, n)
)

df = pd.DataFrame({
    "Area": area,
    "Bedrooms": bedrooms,
    "City": city,
    "Price": price
})

# Create dummy variables
df = pd.get_dummies(df, columns=["City"], drop_first=True)

X = df.drop("Price", axis=1)
y = df["Price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("R2 Score:", r2_score(y_test, y_pred))
