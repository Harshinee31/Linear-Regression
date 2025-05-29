# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
df = pd.read_csv('Housing.csv')

# Clean column names
df.columns = df.columns.str.strip()

# Check columns
print("Columns in dataset:", df.columns.tolist())

# Drop missing values (if any)
df = df.dropna()

# Define target
y = df['price']

# --- Simple Linear Regression (using 'area') ---
X_simple = df[['area']]

# Split data
X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X_simple, y, test_size=0.2, random_state=42)

# Train model
model_simple = LinearRegression()
model_simple.fit(X_train_s, y_train_s)

# Predict
y_pred_s = model_simple.predict(X_test_s)

# Evaluate
print("\n=== Simple Linear Regression (area vs price) ===")
print("MAE:", mean_absolute_error(y_test_s, y_pred_s))
print("MSE:", mean_squared_error(y_test_s, y_pred_s))
print("R² Score:", r2_score(y_test_s, y_pred_s))

# Plot
plt.scatter(X_test_s, y_test_s, color='blue', label='Actual')
plt.plot(X_test_s, y_pred_s, color='red', label='Predicted Line')
plt.xlabel('Area')
plt.ylabel('Price')
plt.title('Simple Linear Regression')
plt.legend()
plt.show()

# --- Multiple Linear Regression ---
features = ['area', 'bedrooms', 'bathrooms', 'stories']
X_multiple = df[features]

# Split data
X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(X_multiple, y, test_size=0.2, random_state=42)

# Train model
model_multiple = LinearRegression()
model_multiple.fit(X_train_m, y_train_m)

# Predict
y_pred_m = model_multiple.predict(X_test_m)

# Evaluate
print("\n=== Multiple Linear Regression ===")
print("MAE:", mean_absolute_error(y_test_m, y_pred_m))
print("MSE:", mean_squared_error(y_test_m, y_pred_m))
print("R² Score:", r2_score(y_test_m, y_pred_m))

# Show coefficients
print("\nCoefficients:")
for feature, coef in zip(features, model_multiple.coef_):
    print(f"{feature}: {coef:.2f}")
print(f"Intercept: {model_multiple.intercept_:.2f}")
