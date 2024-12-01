import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load dataset
df = pd.read_csv('cla/datasets/train.csv')

# Extract relevant features and target variable
X = df[['LotArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'TotRmsAbvGrd']]
y = df['SalePrice']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize Ridge regression model
ridge_model = Ridge()

# Define the parameter grid to tune alpha (regularization strength)
param_grid = {
    'alpha': np.logspace(-6, 6, 13)  # Test a range of alpha values
}

# Initialize GridSearchCV with 5-fold cross-validation
grid_search = GridSearchCV(estimator=ridge_model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=1)

# Fit the model to the training data
grid_search.fit(X_train, y_train)

# Get the best parameters and best score
best_params = grid_search.best_params_
best_score = grid_search.best_score_

print(f'Best hyperparameters: {best_params}')
print(f'Best cross-validation score: {best_score}')

# Use the best model found by GridSearchCV
best_model = grid_search.best_estimator_
y_pred_tuned = best_model.predict(X_test)

# Evaluate the tuned model
mse_tuned = mean_squared_error(y_test, y_pred_tuned)
mae_tuned = mean_absolute_error(y_test, y_pred_tuned)
r2_tuned = r2_score(y_test, y_pred_tuned)

# Print the performance metrics for the tuned model
print(f'Tuned Mean Squared Error (MSE): {mse_tuned}')
print(f'Tuned Mean Absolute Error (MAE): {mae_tuned}')
print(f'Tuned R-squared (R²): {r2_tuned}')

# Visualize the actual vs predicted sale prices for the tuned model
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_tuned, alpha=0.5)
plt.xlabel('Actual Sale Price')
plt.ylabel('Predicted Sale Price')
plt.title('Actual vs Predicted Sale Prices (Tuned Model)')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)  # Line for perfect prediction
plt.show()

# Residual analysis for the tuned model
residuals_tuned = y_test - y_pred_tuned

# Plot residuals for the tuned model
plt.figure(figsize=(10, 6))
plt.scatter(y_pred_tuned, residuals_tuned, alpha=0.5)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Predicted Sale Price')
plt.ylabel('Residuals')
plt.title('Residuals vs Predicted Sale Price (Tuned Model)')
plt.show()

# Plot the distribution of residuals for the tuned model
plt.figure(figsize=(10, 6))
plt.hist(residuals_tuned, bins=30, alpha=0.7, color='blue', edgecolor='black')
plt.title('Residual Distribution (Tuned Model)')
plt.xlabel('Residual')
plt.ylabel('Frequency')
plt.show()


#线性回归模型
# Best hyperparameters: {'alpha': np.float64(1.0)}
# Best cross-validation score: 0.5154148505012615
# Tuned Mean Squared Error (MSE): 3336506161.2413607
# Tuned Mean Absolute Error (MAE): 38567.55408183455
# Tuned R-squared (R²): 0.5650111779972224