import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

# Load the dataset
data = pd.read_csv('housing.csv')

# Separate features (X) and target variable (y)
X = data[['area', 'bedrooms', 'age']]
y = data['price']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Save the trained model
joblib.dump(model, 'linear_regression_model.pkl')

# Print model coefficients
print('Intercept:', model.intercept_)
print('Coefficients:', model.coef_)
