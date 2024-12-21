# House Price Prediction

This project demonstrates a simple machine learning model to predict house prices based on features such as size and number of bedrooms. The model is trained using a dataset of house prices, and the code uses `scikit-learn` to implement a linear regression model.

## Requirements

To run this project, you'll need to install the following Python libraries:

- `numpy`
- `pandas`
- `scikit-learn`

You can install these dependencies using `pip`:

```bash
pip install numpy pandas scikit-learn
```

## Dataset
The dataset used for training the model is a CSV file (house_prices.csv) with the following columns:

- `Size: The size of the house in square feet.`
- `Bedrooms: The number of bedrooms in the house.`
- `Price: The price of the house (target variable).`

### Example Dataset:
```bash
Size,Bedrooms,Price
1500,3,400000
1800,4,500000
2400,3,600000
3000,5,700000
3500,4,800000
```

## Code Explanation
### 1. Import Libraries
```bash
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
```
### 2. Load and Preprocess Data
The dataset is loaded using `pandas` and split into features (`X`) and target (`y`). The data is then split into training and testing tests.
```bash
data = pd.read_csv('house_prices.csv')
X = data[['Size', 'Bedrooms']]  # Features
y = data['Price']               # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```
### 3. Train the Model
A `LinearRegression` model is trained on the training data.
```bash
model = LinearRegression()
model.fit(X_train, y_train)
```

### 4. Evaluate the Model
The modal is evaulated using Mean Squared Error (MSE) on the test set.
```bash
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
```

#### 5. Make Predictions
The trained model can be used to predict house prices for new data.
```bash
new_house = [[2000, 3]]  # Example: 2000 sq ft, 3 bedrooms
predicted_price = model.predict(new_house)
print(f"Predicted Price: {predicted_price[0]}")
```

## Running the Code
1. Install the required libraries:
```bash
pip install numpy pandas scikit-learn
```
2. Download or create the `house_prices.csv` file and place it in the same directory as the Python script.
3. Run the Python script:
```bash
python house_price_predictor.py
```

## License
This project is open-source and available under the [MIT License](https://opensource.org/license/mit).
```bash

### Notes:
- This `README.md` provides an overview of the project, installation instructions, and how to run the code.
- Make sure to include the dataset (`house_prices.csv`) in the same directory as the Python script for the code to work correctly.
```
