import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor

# Load dataset
df = pd.read_csv('cla/datasets/train.csv')

# Extract relevant features and target variable
X = df[['LotArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'TotRmsAbvGrd']]
y = df['SalePrice']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the AdaBoostRegressor model using DecisionTreeRegressor as base estimator
adaboost_model = AdaBoostRegressor(estimator=DecisionTreeRegressor(max_depth=3), n_estimators=100, random_state=42)
adaboost_model.fit(X_train, y_train)

# Make predictions
y_pred_adaboost = adaboost_model.predict(X_test)

# Evaluate the model with multiple metrics
mse_adaboost = mean_squared_error(y_test, y_pred_adaboost)
mae_adaboost = mean_absolute_error(y_test, y_pred_adaboost)
r2_adaboost = r2_score(y_test, y_pred_adaboost)

# Print the performance metrics
print(f'AdaBoost Mean Squared Error (MSE): {mse_adaboost}')
print(f'AdaBoost Mean Absolute Error (MAE): {mae_adaboost}')
print(f'AdaBoost R-squared (R²): {r2_adaboost}')

# Visualize the actual vs predicted sale prices
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_adaboost, alpha=0.5)
plt.xlabel('Actual Sale Price')
plt.ylabel('Predicted Sale Price')
plt.title('Actual vs Predicted Sale Prices (AdaBoost)')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)  # Line for perfect prediction
plt.show()

# Residual analysis (difference between actual and predicted values)
residuals_adaboost = y_test - y_pred_adaboost

# Plot residuals
plt.figure(figsize=(10, 6))
plt.scatter(y_pred_adaboost, residuals_adaboost, alpha=0.5)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Predicted Sale Price')
plt.ylabel('Residuals')
plt.title('Residuals vs Predicted Sale Price (AdaBoost)')
plt.show()

# Plot the distribution of residuals
plt.figure(figsize=(10, 6))
plt.hist(residuals_adaboost, bins=30, alpha=0.7, color='blue', edgecolor='black')
plt.title('Residual Distribution (AdaBoost)')
plt.xlabel('Residual')
plt.ylabel('Frequency')
plt.show()

#集成AdaBoost
# AdaBoost Mean Squared Error (MSE): 3425382744.2567673
# AdaBoost Mean Absolute Error (MAE): 43415.63950922032
# AdaBoost R-squared (R²): 0.5534241110831544