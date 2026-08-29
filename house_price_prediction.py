import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# =========================================================
# 1. LOAD DATASET
# =========================================================

df = pd.read_csv("house_prices.csv")

print("\n========== DATASET ==========")
print(df)

print("\n========== DATASET INFORMATION ==========")
print(df.info())

print("\n========== STATISTICAL SUMMARY ==========")
print(df.describe())


# =========================================================
# 2. CHECK MISSING VALUES
# =========================================================

print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())


# =========================================================
# 3. SELECT FEATURES AND TARGET
# =========================================================

X = df.drop("Price", axis=1)
y = df["Price"]


# =========================================================
# 4. IDENTIFY CATEGORICAL AND NUMERICAL FEATURES
# =========================================================

categorical_features = ["Location"]

numerical_features = [
    "Area",
    "Bedrooms",
    "Bathrooms",
    "Age"
]


# =========================================================
# 5. PREPROCESSING
# =========================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "location",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ],
    remainder="passthrough"
)


# =========================================================
# 6. CREATE LINEAR REGRESSION MODEL
# =========================================================

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression())
    ]
)


# =========================================================
# 7. SPLIT DATA INTO TRAINING AND TESTING SETS
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\n========== DATA SPLIT ==========")
print("Training records:", len(X_train))
print("Testing records:", len(X_test))


# =========================================================
# 8. TRAIN MODEL
# =========================================================

model.fit(X_train, y_train)

print("\nModel training completed successfully.")


# =========================================================
# 9. MAKE PREDICTIONS
# =========================================================

y_pred = model.predict(X_test)


# =========================================================
# 10. MODEL EVALUATION
# =========================================================

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n========== MODEL PERFORMANCE ==========")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R² Score: {r2:.4f}")


# =========================================================
# 11. ACTUAL VS PREDICTED PRICES
# =========================================================

results = pd.DataFrame({
    "Actual Price": y_test.values,
    "Predicted Price": y_pred
})

print("\n========== ACTUAL VS PREDICTED ==========")
print(results)


# =========================================================
# 12. VISUALIZATION
# =========================================================

plt.figure(figsize=(8, 6))

plt.scatter(y_test, y_pred)

plt.xlabel("Actual House Price")
plt.ylabel("Predicted House Price")
plt.title("Actual vs Predicted House Prices")

plt.tight_layout()

plt.savefig("actual_vs_predicted.png")

plt.show()


# =========================================================
# 13. PREDICT PRICE FOR A NEW HOUSE
# =========================================================

new_house = pd.DataFrame({
    "Area": [1500],
    "Bedrooms": [3],
    "Bathrooms": [2],
    "Age": [5],
    "Location": ["Mumbai"]
})

predicted_price = model.predict(new_house)[0]

print("\n========== NEW HOUSE PREDICTION ==========")
print("House Details:")
print(new_house)

print(f"\nPredicted House Price: ₹{predicted_price:,.2f}")