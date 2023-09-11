import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
cars = pd.read_csv('car_prices.csv')
cars = cars.dropna()
cars = pd.get_dummies(cars, columns=['make', 'model', 'transmission', 'fuelType'])
X_train, X_test, y_train, y_test = train_test_split(cars.drop('price', axis=1), cars['price'], test_size=0.25, random_state=0)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
model = LinearRegression()
model.fit(X_train, y_train)
print('The R-squared score of the model is:', model.score(X_test, y_test))
make = input('Enter the make of the car: ')
model = input('Enter the model of the car: ')
transmission = input('Enter the transmission type of the car: ')
fuelType = input('Enter the fuel type of the car: ')
mileage = float(input('Enter the mileage of the car: '))
year = int(input('Enter the year of the car: '))
user_input = pd.DataFrame({
    'make': [make],
    'model': [model],
    'transmission': [transmission],
    'fuelType': [fuelType],
    'mileage': [mileage],
    'year': [year]
})
user_input = pd.get_dummies(user_input, columns=['make', 'model', 'transmission', 'fuelType'])
user_input = scaler.transform(user_input)
price = model.predict(user_input)
