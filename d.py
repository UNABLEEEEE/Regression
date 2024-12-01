import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import lightgbm as lgb
from scipy.stats import uniform, randint

# Load dataset
df = pd.read_csv('cla/datasets/train.csv')

# Extract relevant features and target variable
X = df[['LotArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'TotRmsAbvGrd']]
y = df['SalePrice']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize LightGBM regressor model
lgb_model = lgb.LGBMRegressor(random_state=42)

# Define the parameter grid for RandomizedSearchCV
param_dist = {
    'n_estimators': randint(50, 200),  # Number of trees in the forest
    'learning_rate': uniform(0.01, 0.2),  # Learning rate
    'num_leaves': randint(20, 100),  # Number of leaves in one tree
    'max_depth': randint(3, 15),  # Maximum depth of the tree
    'min_child_samples': randint(10, 50),  # Minimum number of samples in a leaf node
    'subsample': uniform(0.5, 0.5),  # Fraction of data to be used for each tree
    'colsample_bytree': uniform(0.5, 0.5)  # Fraction of features to be used for each tree
}

# Setup RandomizedSearchCV
random_search = RandomizedSearchCV(estimator=lgb_model, param_distributions=param_dist, 
                                   n_iter=50, cv=5, verbose=2, random_state=42, n_jobs=-1)

# Perform the random search
random_search.fit(X_train, y_train)

# Print the best parameters found by RandomizedSearchCV
print("Best parameters found: ", random_search.best_params_)

# Evaluate the best model
best_model = random_search.best_estimator_

# Make predictions
y_pred_lgb_optimized = best_model.predict(X_test)

# Evaluate the optimized model with multiple metrics
mse_lgb_optimized = mean_squared_error(y_test, y_pred_lgb_optimized)
mae_lgb_optimized = mean_absolute_error(y_test, y_pred_lgb_optimized)
r2_lgb_optimized = r2_score(y_test, y_pred_lgb_optimized)

# Print the performance metrics
print(f'Optimized LightGBM Mean Squared Error (MSE): {mse_lgb_optimized}')
print(f'Optimized LightGBM Mean Absolute Error (MAE): {mae_lgb_optimized}')
print(f'Optimized LightGBM R-squared (R²): {r2_lgb_optimized}')

# Visualize the actual vs predicted sale prices
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_lgb_optimized, alpha=0.5)
plt.xlabel('Actual Sale Price')
plt.ylabel('Predicted Sale Price')
plt.title('Actual vs Predicted Sale Prices (Optimized LightGBM)')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)  # Line for perfect prediction
plt.show()

# Residual analysis (difference between actual and predicted values)
residuals_lgb_optimized = y_test - y_pred_lgb_optimized

# Plot residuals
plt.figure(figsize=(10, 6))
plt.scatter(y_pred_lgb_optimized, residuals_lgb_optimized, alpha=0.5)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Predicted Sale Price')
plt.ylabel('Residuals')
plt.title('Residuals vs Predicted Sale Price (Optimized LightGBM)')
plt.show()

# Plot the distribution of residuals
plt.figure(figsize=(10, 6))
plt.hist(residuals_lgb_optimized, bins=30, alpha=0.7, color='blue', edgecolor='black')
plt.title('Residual Distribution (Optimized LightGBM)')
plt.xlabel('Residual')
plt.ylabel('Frequency')
plt.show()


# LightGBM 回归RandomizedSearchCV调优
# Best parameters found:  {'colsample_bytree': np.float64(0.7776004057997312), 'learning_rate': np.float64(0.1159301156712013), 'max_depth': 4, 'min_child_samples': 10, 'n_estimators': 97, 'num_leaves': 31, 'subsample': np.float64(0.7791467268035488)}
# Optimized LightGBM Mean Squared Error (MSE): 2590627686.533158
# Optimized LightGBM Mean Absolute Error (MAE): 32594.26795732278
# Optimized LightGBM R-squared (R²): 0.6622532580028044