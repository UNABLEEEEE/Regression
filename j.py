import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import VotingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

# Load dataset
df = pd.read_csv('cla/datasets/train.csv')

# Extract relevant features and target variable
X = df[['LotArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'TotRmsAbvGrd']]
y = df['SalePrice']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define base models for voting
base_learners = [
    ('rf', RandomForestRegressor(n_estimators=100, random_state=42)),
    ('dt', DecisionTreeRegressor(random_state=42)),
    ('svr', SVR(kernel='rbf'))
]

# Initialize and train the VotingRegressor
voting_model = VotingRegressor(estimators=base_learners)
voting_model.fit(X_train, y_train)

# Make predictions
y_pred_voting = voting_model.predict(X_test)

# Evaluate the model with multiple metrics
mse_voting = mean_squared_error(y_test, y_pred_voting)
mae_voting = mean_absolute_error(y_test, y_pred_voting)
r2_voting = r2_score(y_test, y_pred_voting)

# Print the performance metrics
print(f'Voting Mean Squared Error (MSE): {mse_voting}')
print(f'Voting Mean Absolute Error (MAE): {mae_voting}')
print(f'Voting R-squared (R²): {r2_voting}')

# Visualize the actual vs predicted sale prices
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_voting, alpha=0.5)
plt.xlabel('Actual Sale Price')
plt.ylabel('Predicted Sale Price')
plt.title('Actual vs Predicted Sale Prices (Voting)')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)  # Line for perfect prediction
plt.show()

# Residual analysis (difference between actual and predicted values)
residuals_voting = y_test - y_pred_voting

# Plot residuals
plt.figure(figsize=(10, 6))
plt.scatter(y_pred_voting, residuals_voting, alpha=0.5)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Predicted Sale Price')
plt.ylabel('Residuals')
plt.title('Residuals vs Predicted Sale Price (Voting)')
plt.show()

# Plot the distribution of residuals
plt.figure(figsize=(10, 6))
plt.hist(residuals_voting, bins=30, alpha=0.7, color='blue', edgecolor='black')
plt.title('Residual Distribution (Voting)')
plt.xlabel('Residual')
plt.ylabel('Frequency')
plt.show()

#Voting
# Voting Mean Squared Error (MSE): 2953561115.488818
# Voting Mean Absolute Error (MAE): 36624.290892848665
# Voting R-squared (R²): 0.6149367007727364