import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor

# Load dataset
df = pd.read_csv('cla/datasets/train.csv')

# Extract relevant features and target variable
X = df[['LotArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'TotRmsAbvGrd']]
y = df['SalePrice']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the BaggingRegressor model (no base_estimator needed)
bagging_model = BaggingRegressor(n_estimators=100, random_state=42)
bagging_model.fit(X_train, y_train)

# Make predictions
y_pred_bagging = bagging_model.predict(X_test)

# Evaluate the model with multiple metrics
mse_bagging = mean_squared_error(y_test, y_pred_bagging)
mae_bagging = mean_absolute_error(y_test, y_pred_bagging)
r2_bagging = r2_score(y_test, y_pred_bagging)

# Print the performance metrics
print(f'Bagging Mean Squared Error (MSE): {mse_bagging}')
print(f'Bagging Mean Absolute Error (MAE): {mae_bagging}')
print(f'Bagging R-squared (R²): {r2_bagging}')

# Visualize the actual vs predicted sale prices
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_bagging, alpha=0.5)
plt.xlabel('Actual Sale Price')
plt.ylabel('Predicted Sale Price')
plt.title('Actual vs Predicted Sale Prices (Bagging)')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)  # Line for perfect prediction
plt.show()

# Residual analysis (difference between actual and predicted values)
residuals_bagging = y_test - y_pred_bagging

# Plot residuals
plt.figure(figsize=(10, 6))
plt.scatter(y_pred_bagging, residuals_bagging, alpha=0.5)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Predicted Sale Price')
plt.ylabel('Residuals')
plt.title('Residuals vs Predicted Sale Price (Bagging)')
plt.show()

# Plot the distribution of residuals
plt.figure(figsize=(10, 6))
plt.hist(residuals_bagging, bins=30, alpha=0.7, color='blue', edgecolor='black')
plt.title('Residual Distribution (Bagging)')
plt.xlabel('Residual')
plt.ylabel('Frequency')
plt.show()

#集成Bagging
# Bagging Mean Squared Error (MSE): 2775462446.441244
# Bagging Mean Absolute Error (MAE): 34763.7530758317
# Bagging R-squared (R²): 0.6381558787107875