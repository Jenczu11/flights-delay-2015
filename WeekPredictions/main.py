from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

# Pandas options
pd.set_option('display.max_columns', None)
pd.set_option('chained_assignment', None)  # default='warn'

# Load data
print("Started loading data...")
# https://www.kaggle.com/usdot/flight-delays
flights_path = "./flights.csv"
flights_data = pd.read_csv(flights_path, low_memory=False)
print("Finished loading data.")

# Remove rows with at least one null value
print("Removing rows with at least one null value...")
flights_data = flights_data.dropna(
    subset=['AIRLINE',
            'ORIGIN_AIRPORT',
            'DESTINATION_AIRPORT',
            'MONTH',
            'DAY',
            'DISTANCE',
            'DEPARTURE_DELAY',
            'SCHEDULED_TIME',
            'ARRIVAL_DELAY'])

# Build data set with columns which will be used for prediction
print("Building data set with columns which will be used for prediction:")
print("AIRLINE, ORIGIN_AIRPORT, DESTINATION_AIRPORT, DAY, DISTANCE, DEPARTURE_DELAY, SCHEDULED_TIME.")
flights_data_X = flights_data[[
    'AIRLINE',
    'ORIGIN_AIRPORT',
    'DESTINATION_AIRPORT',
    'MONTH',
    'DAY',
    'DISTANCE',
    'DEPARTURE_DELAY',
    'SCHEDULED_TIME'
]]

# Parse objects to strings
print("Parsing columns to appropriate types...")
flights_data_X['AIRLINE'] = flights_data_X['AIRLINE'].astype("|S")
flights_data_X['ORIGIN_AIRPORT'] = flights_data_X['ORIGIN_AIRPORT'].astype("|S")
flights_data_X['DESTINATION_AIRPORT'] = flights_data_X['DESTINATION_AIRPORT'].astype("|S")

# Label string columns as integers
print("Labeling columns: AIRLINE, ORIGIN_AIRPORT, DESTINATION_AIRPORT and DAY...")
le = LabelEncoder()
flights_data_X['AIRLINE'] = le.fit_transform(flights_data_X['AIRLINE'])
flights_data_X['ORIGIN_AIRPORT'] = le.fit_transform(flights_data_X['ORIGIN_AIRPORT'])
flights_data_X['DESTINATION_AIRPORT'] = le.fit_transform(flights_data_X['DESTINATION_AIRPORT'])
#flights_data_X['DAY'] = le.fit_transform(flights_data_X['DAY'])

flights_data_Y = flights_data[[
    'ARRIVAL_DELAY',
    'MONTH',
    'DAY'
]]
flights_data_Y = flights_data_Y.dropna(
    subset=["ARRIVAL_DELAY",
            'MONTH',
            'DAY'
])

print("Splitting data set for train and test (80%/20%)...")

X_train = flights_data_X[(flights_data_X['MONTH'] == 1) & (flights_data_X['DAY'] < 23)]
X_test = flights_data_X[(flights_data_X['MONTH'] == 1) & (flights_data_X['DAY'] > 23)]

y_train = flights_data_Y[(flights_data_Y['MONTH'] == 1) & (flights_data_Y['DAY'] < 23)]
y_test = flights_data_Y[(flights_data_Y['MONTH'] == 1) & (flights_data_Y['DAY'] > 23)]


print("Normalizing data...")
sc1 = StandardScaler()
X_train_sc = sc1.fit_transform(X_train)
X_test_sc = sc1.fit_transform(X_test)

# Create regression objects
LinearRegression = LinearRegression()
Lasso = Lasso(alpha=0.25, max_iter=10000)
DecisionTree = DecisionTreeRegressor(
    min_samples_split=300,
    min_samples_leaf=75,
    max_features="auto",
    random_state=1
)

print("Started calculating regressions...")
for model, name in zip([LinearRegression, Lasso, DecisionTree],
                       ['Linear Regression', 'Lasso Regression', 'Decision Tree Regression']):
    fitted_model = model.fit(X_train_sc, y_train)
    Y_predict = model.predict(X_test_sc)
    print('===============================')
    print('Learning with ' + name)
    print('Mean Absolute Error:', mean_absolute_error(y_test, Y_predict))
    print('Mean Squared Error:', mean_squared_error(y_test, Y_predict))
    print('Root Mean Squared Error:', np.sqrt(mean_squared_error(y_test, Y_predict)))
    print('R2 : ', r2_score(y_test, Y_predict))
    print('Plotting results...')

    plt.scatter(y_test, Y_predict)
    plt.title("Test flights delay prediction - " + name)
    plt.xlabel("Truth")
    plt.ylabel("Prediction")
    plt.xlim(-250.0, 1750.0)
    plt.ylim(-250.0, 1750.0)
    plt.show()

print("\nDone.")
