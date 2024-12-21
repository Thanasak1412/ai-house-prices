import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load dataset
data = pd.read_csv("house_prices.csv") # Assume the dataset has 'Size', 'Bedrooms', 'Price'

# Preprocess data
X = data[['Size', 'Bedrooms']] # Features
y = data['Price'] # Target variable

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# Calculate Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Predict price for a new house
new_house = [[2000, 3]] # 2000 sq ft, 3 bedrooms
predicted_price = model.predict(new_house)
print(f"Predicted Price: {predicted_price[0]}")