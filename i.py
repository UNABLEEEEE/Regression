import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor

# Load dataset
df = pd.read_csv('cla/datasets/train.csv')


# Extract relevant features and target variable
X = df[['LotArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'TotRmsAbvGrd']]
y = df['SalePrice']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define base models for stacking
base_learners = [
    ('rf', RandomForestRegressor(n_estimators=100, random_state=42)),
    ('dt', DecisionTreeRegressor(random_state=42)),
    ('svr', SVR(kernel='rbf'))
]

# Define meta-learner (final model)
meta_learner = LinearRegression()

# Initialize and train the StackingRegressor
stacking_model = StackingRegressor(estimators=base_learners, final_estimator=meta_learner)
stacking_model.fit(X_train, y_train)

# Make predictions
y_pred_stacking = stacking_model.predict(X_test)

# Evaluate the model with multiple metrics
mse_stacking = mean_squared_error(y_test, y_pred_stacking)
mae_stacking = mean_absolute_error(y_test, y_pred_stacking)
r2_stacking = r2_score(y_test, y_pred_stacking)

# Print the performance metrics
print(f'Stacking Mean Squared Error (MSE): {mse_stacking}')
print(f'Stacking Mean Absolute Error (MAE): {mae_stacking}')
print(f'Stacking R-squared (R²): {r2_stacking}')

# Visualize the actual vs predicted sale prices
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_stacking, alpha=0.5)
plt.xlabel('Actual Sale Price')
plt.ylabel('Predicted Sale Price')
plt.title('Actual vs Predicted Sale Prices (Stacking)')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)  # Line for perfect prediction
plt.show()

# Residual analysis (difference between actual and predicted values)
residuals_stacking = y_test - y_pred_stacking

# Plot residuals
plt.figure(figsize=(10, 6))
plt.scatter(y_pred_stacking, residuals_stacking, alpha=0.5)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Predicted Sale Price')
plt.ylabel('Residuals')
plt.title('Residuals vs Predicted Sale Price (Stacking)')
plt.show()

# Plot the distribution of residuals
plt.figure(figsize=(10, 6))
plt.hist(residuals_stacking, bins=30, alpha=0.7, color='blue', edgecolor='black')
plt.title('Residual Distribution (Stacking)')
plt.xlabel('Residual')
plt.ylabel('Frequency')
plt.show()

#Stacking
# Stacking Mean Squared Error (MSE): 2760923752.1604156
# Stacking Mean Absolute Error (MAE): 34486.09852943962
# Stacking R-squared (R²): 0.6400513253825607