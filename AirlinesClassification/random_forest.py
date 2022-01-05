import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier

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
            'DAY',
            'MONTH',
            'DISTANCE',
            'DEPARTURE_DELAY',
            'SCHEDULED_TIME',
            'ARRIVAL_DELAY'])

# Build data set with columns which will be used for prediction
print("Building data set with columns which will be used for classification:")
print("ORIGIN_AIRPORT, DESTINATION_AIRPORT, DAY, MONTH, DISTANCE, DEPARTURE_DELAY, SCHEDULED_TIME, ARRIVAL_DELAY")
flights_data_X = flights_data[[
    'ORIGIN_AIRPORT',
    'DESTINATION_AIRPORT',
    'DAY',
    'MONTH',
    'DISTANCE',
    'DEPARTURE_DELAY',
    'SCHEDULED_TIME',
    'ARRIVAL_DELAY'
]]

flights_data_Y = flights_data[[
    'AIRLINE'
]]
flights_data_Y = flights_data_Y.dropna(
    subset=["AIRLINE"])

# Parse objects to strings
print("Parsing columns to appropriate types...")
flights_data_X['ORIGIN_AIRPORT'] = flights_data_X['ORIGIN_AIRPORT'].astype("|S")
flights_data_X['DESTINATION_AIRPORT'] = flights_data_X['DESTINATION_AIRPORT'].astype("|S")
flights_data_Y['AIRLINE'] = flights_data_Y['AIRLINE'].astype("|S")

# Label string columns as integers
print("Labeling columns: AIRLINE, ORIGIN_AIRPORT, DESTINATION_AIRPORT and DAY...")
le = LabelEncoder()
flights_data_X['ORIGIN_AIRPORT'] = le.fit_transform(flights_data_X['ORIGIN_AIRPORT'])
flights_data_X['DESTINATION_AIRPORT'] = le.fit_transform(flights_data_X['DESTINATION_AIRPORT'])
flights_data_X['DAY'] = le.fit_transform(flights_data_X['DAY'])
flights_data_X['MONTH'] = le.fit_transform(flights_data_X['MONTH'])
flights_data_Y['AIRLINE'] = le.fit_transform(flights_data_Y['AIRLINE'])

print("Splitting data set for train and test (90%/10%)...")
X_train, X_test, Y_train, Y_test = train_test_split(
    flights_data_X, flights_data_Y, test_size=0.9, random_state=0)

print("Normalizing data...")
sc1 = StandardScaler()
X_train_sc = sc1.fit_transform(X_train)
X_test_sc = sc1.fit_transform(X_test)

print("Classification model fitting started...")
model = RandomForestClassifier()
model.fit(X_train_sc, Y_train['AIRLINE'])
print("Fitting finished. Started classifying test flights...")
Y_pred = model.predict(X_test_sc)

categories_dist = flights_data_Y['AIRLINE'].value_counts().sort_index()
accuracy = accuracy_score(Y_test, Y_pred)
recall = recall_score(Y_test, Y_pred, average=None)
precision = precision_score(Y_test, Y_pred, average=None)
cmt = confusion_matrix(Y_test, Y_pred)
print("Classification finished. Accuracy:", accuracy * 100.0, '%')
print("Plotting results...")

# Plot all results. Prepare labels first.
# labels = ["AA", "AS", "B6", "DL", "EV", "F9", "HA", "MQ", "NK", "OO", "UA", "US", "VX", "WN"]
# y_pos = np.arange(len(labels))

# plt.barh(y_pos, categories_dist, align='center', alpha=0.5, color=['green'])
# plt.yticks(y_pos, labels)
# plt.xlabel('Number of flights')
# plt.title('Airlines distribution')
# plt.show()

# plt.bar(y_pos, precision, align='center', alpha=0.5, color=['blue'])
# plt.xticks(y_pos, labels)
# plt.ylabel('Precision')
# plt.title('Precision for each airline')
# plt.show()

# plt.bar(y_pos, recall, align='center', alpha=0.5, color=['red'])
# plt.xticks(y_pos, labels)
# plt.ylabel('Recall')
# plt.title('Recall for each airline')
# plt.show()

# plt.subplots(figsize=(12, 9))
# plot = sns.heatmap(cmt, annot=True, cmap='Blues', fmt='g')
# plot.set_title('Predicted airlines vs true airlines - confusion matrix')
# plot.set_xlabel('\nPredicted airlines')
# plot.set_ylabel('True airlines')
# plot.xaxis.set_ticklabels(labels)
# plot.yaxis.set_ticklabels(labels)
# plt.show()

print("Finished.")
