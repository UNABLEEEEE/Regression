import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb

# Load dataset
df = pd.read_csv('cla/datasets/train.csv')

# Extract relevant features and target variable
X = df[['LotArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'TotRmsAbvGrd']]
y = df['SalePrice']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the XGBoost regressor model
xgb_model = xgb.XGBRegressor(random_state=42)

# Define parameter grid for GridSearchCV
param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7],
    'min_child_weight': [1, 5, 10],
    'subsample': [0.7, 0.8, 1.0],
    'colsample_bytree': [0.7, 0.8, 1.0]
}

# Perform GridSearchCV
grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, scoring='neg_mean_squared_error', cv=5, verbose=2, n_jobs=-1)
grid_search.fit(X_train, y_train)

# Get the best parameters from GridSearchCV
best_params = grid_search.best_params_

# Train the model with the best parameters
best_xgb_model = grid_search.best_estimator_

# Make predictions
y_pred_xgb_optimized = best_xgb_model.predict(X_test)

# Evaluate the optimized model with multiple metrics
mse_xgb_optimized = mean_squared_error(y_test, y_pred_xgb_optimized)
mae_xgb_optimized = mean_absolute_error(y_test, y_pred_xgb_optimized)
r2_xgb_optimized = r2_score(y_test, y_pred_xgb_optimized)

# Print the optimized model performance
print(f'Optimized XGBoost Mean Squared Error (MSE): {mse_xgb_optimized}')
print(f'Optimized XGBoost Mean Absolute Error (MAE): {mae_xgb_optimized}')
print(f'Optimized XGBoost R-squared (R²): {r2_xgb_optimized}')

# Visualize the actual vs predicted sale prices for optimized model
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_xgb_optimized, alpha=0.5)
plt.xlabel('Actual Sale Price')
plt.ylabel('Predicted Sale Price')
plt.title('Actual vs Predicted Sale Prices (Optimized XGBoost)')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)  # Line for perfect prediction
plt.show()

# Residual analysis (difference between actual and predicted values)
residuals_xgb_optimized = y_test - y_pred_xgb_optimized

# Plot residuals for optimized model
plt.figure(figsize=(10, 6))
plt.scatter(y_pred_xgb_optimized, residuals_xgb_optimized, alpha=0.5)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Predicted Sale Price')
plt.ylabel('Residuals')
plt.title('Residuals vs Predicted Sale Price (Optimized XGBoost)')
plt.show()

# Plot the distribution of residuals for optimized model
plt.figure(figsize=(10, 6))
plt.hist(residuals_xgb_optimized, bins=30, alpha=0.7, color='blue', edgecolor='black')
plt.title('Residual Distribution (Optimized XGBoost)')
plt.xlabel('Residual')
plt.ylabel('Frequency')
plt.show()

# XGBoost回归GridSearchCV调优
# Optimized XGBoost Mean Squared Error (MSE): 2574644480.666287
# Optimized XGBoost Mean Absolute Error (MAE): 31872.04553724315
# Optimized XGBoost R-squared (R²): 0.6643370389938354