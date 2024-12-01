import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import randint

# Load dataset
df = pd.read_csv('cla/datasets/train.csv')

# Extract relevant features and target variable
X = df[['LotArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'TotRmsAbvGrd']]
y = df['SalePrice']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize Random Forest model
rf_model = RandomForestRegressor(random_state=42)

# Hyperparameters to tune
param_dist = {
    'n_estimators': randint(100, 200),  # Number of trees
    'max_depth': randint(10, 20),  # Maximum depth of trees
    'min_samples_split': randint(2, 20),  # Minimum samples required to split a node
    'min_samples_leaf': randint(1, 20),  # Minimum samples required at a leaf node
    'max_features': ['auto', 'sqrt', 'log2'],  # Number of features to consider for splitting
    'bootstrap': [True, False]  # Whether bootstrap samples are used
}

# Use RandomizedSearchCV for hyperparameter tuning
random_search = RandomizedSearchCV(rf_model, param_distributions=param_dist, n_iter=100, cv=5, verbose=2, random_state=42, n_jobs=-1)

# Train the model with randomized search
random_search.fit(X_train, y_train)

# Get the best parameters from the random search
best_params = random_search.best_params_
print(f"Best Parameters: {best_params}")

# Evaluate the model with the best parameters
y_pred_rf = random_search.predict(X_test)

# Evaluate the model with multiple metrics
mse_rf = mean_squared_error(y_test, y_pred_rf)
mae_rf = mean_absolute_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

# Print the performance metrics
print(f'Random Forest (Tuned) Mean Squared Error (MSE): {mse_rf}')
print(f'Random Forest (Tuned) Mean Absolute Error (MAE): {mae_rf}')
print(f'Random Forest (Tuned) R-squared (R²): {r2_rf}')

# Visualize the actual vs predicted sale prices
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_rf, alpha=0.5)
plt.xlabel('Actual Sale Price')
plt.ylabel('Predicted Sale Price')
plt.title('Actual vs Predicted Sale Prices (Tuned Random Forest)')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)  # Line for perfect prediction
plt.show()

# Residual analysis (difference between actual and predicted values)
residuals_rf = y_test - y_pred_rf

# Plot residuals
plt.figure(figsize=(10, 6))
plt.scatter(y_pred_rf, residuals_rf, alpha=0.5)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Predicted Sale Price')
plt.ylabel('Residuals')
plt.title('Residuals vs Predicted Sale Price (Tuned Random Forest)')
plt.show()

# Plot the distribution of residuals
plt.figure(figsize=(10, 6))
plt.hist(residuals_rf, bins=30, alpha=0.7, color='blue', edgecolor='black')
plt.title('Residual Distribution (Tuned Random Forest)')
plt.xlabel('Residual')
plt.ylabel('Frequency')
plt.show()

#KNN模型RandomizedSearchCV调优
# Random Forest (Tuned) Mean Squared Error (MSE): 2700715350.563379
# Random Forest (Tuned) Mean Absolute Error (MAE): 32846.27042484431
# Random Forest (Tuned) R-squared (R²): 0.6479008483325261