import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV  # 确保导入 RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import randint
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('cla/datasets/train.csv')

# Extract relevant features and target variable
X = df[['LotArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'TotRmsAbvGrd']]
y = df['SalePrice']

# Split the dataset into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the parameter distributions for RandomizedSearchCV
param_dist = {
    'n_estimators': randint(50, 200),         # Number of trees in the forest
    'max_depth': [None, 10, 20, 30],           # Maximum depth of the tree
    'min_samples_split': randint(2, 11),       # Minimum number of samples required to split a node
    'min_samples_leaf': randint(1, 5),         # Minimum number of samples required at each leaf node
    'bootstrap': [True, False]                 # Whether bootstrap samples are used when building trees
}

# Initialize the RandomForestRegressor
rf_model = RandomForestRegressor(random_state=42)

# Set up RandomizedSearchCV with 5-fold cross-validation
random_search = RandomizedSearchCV(estimator=rf_model, param_distributions=param_dist, 
                                   n_iter=100, cv=5, n_jobs=-1, verbose=2, scoring='neg_mean_squared_error', random_state=42)

# Fit RandomizedSearchCV to the training data
random_search.fit(X_train, y_train)

# Get the best hyperparameters
best_params_random = random_search.best_params_
print(f"Best Parameters from RandomizedSearchCV: {best_params_random}")

# Use the best model found by RandomizedSearchCV
best_rf_model_random = random_search.best_estimator_

# Make predictions and evaluate the model
y_pred_random = best_rf_model_random.predict(X_test)

# Evaluate the model performance
mse_random = mean_squared_error(y_test, y_pred_random)
mae_random = mean_absolute_error(y_test, y_pred_random)
r2_random = r2_score(y_test, y_pred_random)

print(f"RandomizedSearchCV MSE: {mse_random}")
print(f"RandomizedSearchCV MAE: {mae_random}")
print(f"RandomizedSearchCV R²: {r2_random}")

# Visualize the actual vs predicted sale prices
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_random, alpha=0.5)
plt.xlabel('Actual Sale Price')
plt.ylabel('Predicted Sale Price')
plt.title('Actual vs Predicted Sale Prices (RandomizedSearchCV - Random Forest)')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)  # Line for perfect prediction
plt.show()

# Residual analysis (difference between actual and predicted values)
residuals_random = y_test - y_pred_random

# Plot residuals
plt.figure(figsize=(10, 6))
plt.scatter(y_pred_random, residuals_random, alpha=0.5)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Predicted Sale Price')
plt.ylabel('Residuals')
plt.title('Residuals vs Predicted Sale Price (RandomizedSearchCV - Random Forest)')
plt.show()

# Plot the distribution of residuals
plt.figure(figsize=(10, 6))
plt.hist(residuals_random, bins=30, alpha=0.7, color='blue', edgecolor='black')
plt.title('Residual Distribution (RandomizedSearchCV - Random Forest)')
plt.xlabel('Residual')
plt.ylabel('Frequency')
plt.show()

#RandomizedSearchCV调优得随机森林回归
# Best Parameters from RandomizedSearchCV: {'bootstrap': True, 'max_depth': 10, 'min_samples_leaf': 1, 'min_samples_split': 8, 'n_estimators': 176}
# RandomizedSearchCV MSE: 2661071477.884012
# RandomizedSearchCV MAE: 33237.37411232581
# RandomizedSearchCV R²: 0.6530693211729929
