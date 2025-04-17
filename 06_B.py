import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Fetch the California housing dataset
housing = fetch_california_housing()

# Convert the data into a DataFrame for easier manipulation
housing_df = pd.DataFrame(housing.data, columns=housing.feature_names)

# Add the target variable 'PRICE' to the DataFrame
housing_df['PRICE'] = housing.target

# Define the features (X) and target (y)
X = housing_df.drop('PRICE', axis=1)  # Drop 'PRICE' from features
y = housing_df['PRICE']  # Target is 'PRICE'

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the linear regression model
model = LinearRegression()

# Train the model using the training data
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate the Mean Squared Error and R-squared values
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print the results
print("Mean Squared Error:", mse)
print("R-squared:", r2)
print("Intercept:", model.intercept_)
print("Coefficients:", model.coef_)
