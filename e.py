import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv('cla/datasets/train.csv')

# Extract relevant features and target variable
X = df[['LotArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'TotRmsAbvGrd']]
y = df['SalePrice']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling for SVR
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize the SVR model
svr_model = SVR()

# Define parameter grid for GridSearchCV
param_grid = {
    'kernel': ['linear', 'poly', 'rbf'],
    'C': [0.1, 1, 10, 100],
    'epsilon': [0.01, 0.1, 0.5, 1]
}

# Perform grid search with 5-fold cross-validation
grid_search = GridSearchCV(svr_model, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train_scaled, y_train)

# Best parameters from grid search
print(f"Best parameters from grid search: {grid_search.best_params_}")

# Use the best model from grid search
best_svr_model = grid_search.best_estimator_

# Make predictions using the best model
y_pred_svr = best_svr_model.predict(X_test_scaled)

# Evaluate the model with multiple metrics
mse_svr = mean_squared_error(y_test, y_pred_svr)
mae_svr = mean_absolute_error(y_test, y_pred_svr)
r2_svr = r2_score(y_test, y_pred_svr)

# Print the performance metrics
print(f'SVR Mean Squared Error (MSE): {mse_svr}')
print(f'SVR Mean Absolute Error (MAE): {mae_svr}')
print(f'SVR R-squared (R²): {r2_svr}')

# Visualize the actual vs predicted sale prices
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_svr, alpha=0.5)
plt.xlabel('Actual Sale Price')
plt.ylabel('Predicted Sale Price')
plt.title('Actual vs Predicted Sale Prices (Optimized SVR)')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)  # Line for perfect prediction
plt.show()

# Residual analysis (difference between actual and predicted values)
residuals_svr = y_test - y_pred_svr

# Plot residuals
plt.figure(figsize=(10, 6))
plt.scatter(y_pred_svr, residuals_svr, alpha=0.5)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Predicted Sale Price')
plt.ylabel('Residuals')
plt.title('Residuals vs Predicted Sale Price (Optimized SVR)')
plt.show()

# Plot the distribution of residuals
plt.figure(figsize=(10, 6))
plt.hist(residuals_svr, bins=30, alpha=0.7, color='blue', edgecolor='black')
plt.title('Residual Distribution (Optimized SVR)')
plt.xlabel('Residual')
plt.ylabel('Frequency')
plt.show()


#SVR模型GridSearchCV调优
# Best parameters from grid search: {'C': 100, 'epsilon': 0.01, 'kernel': 'linear'}
# SVR Mean Squared Error (MSE): 4527053464.890937
# SVR Mean Absolute Error (MAE): 40735.09399144489
# SVR R-squared (R²): 0.40979648810124003