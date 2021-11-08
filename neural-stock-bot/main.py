import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from pdb import set_trace
import random
import ta
import talib

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from sklearn.preprocessing import StandardScaler
from yahoo_fin import stock_info as si

# Set randomizer seeds for consistent results
seed = 314
np.random.seed(seed)
#tf.random.set_seed(seed)
random.seed(seed)


ticker = 'SPY'

# Create a saves folder if it doesn't already exist
if not os.path.isdir("saves"):
    os.mkdir("saves")

# Load and clean the dataset
dataset = si.get_data(ticker)
dataset.dropna()
# Get rid of the adjclose and ticker columns
dataset = dataset[['open', 'high', 'low', 'close', 'volume']]

# # Add technical indicators to dataset
# dataset = ta.add_all_ta_features(dataset, open="open", high="high", low="low", close="close", volume="volume")

# Prep the dataset and add some new technical indicators
dataset['H-L'] = dataset['high'] - dataset['low']
dataset['O-C'] = dataset['close'] - dataset['open']
dataset['10day MA'] = dataset['close'].shift(1).rolling(window=10).mean()
dataset['30day MA'] = dataset['close'].shift(1).rolling(window=30).mean()
dataset['Std_dev'] = dataset['close'].rolling(5).std()
dataset['RSI'] = talib.RSI(dataset['close'].values, timeperiod=9)
dataset['Williams %R'] = talib.WILLR(dataset['high'].values, dataset['low'].values, dataset['close'].values, 7)
dataset['Price_Rise'] = np.where(dataset['close'].shift(-1) > dataset['close'], 1, 0)

# Replace NaN values with 0
dataset = dataset.fillna(0)


dataset = dataset.dropna()

input = dataset.iloc[:, 0:-1]
output = dataset.iloc[:, -1]

# Split the dataset into training and testing
split = int(len(dataset)*0.8)
X_train, X_test, y_train, y_test = input[:split], input[split:], output[:split], output[split:]

# Normalize the data
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Build the neural network
classifier = Sequential()
classifier.add(Dense(units=128, kernel_initializer='uniform', activation='relu', input_dim=input.shape[1]))
classifier.add(Dense(units=128, kernel_initializer='uniform', activation='relu'))
classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))
classifier.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

# Train the model
classifier.fit(X_train, y_train, batch_size=10, epochs=5)

y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

dataset['y_pred'] = np.NaN
dataset.iloc[(len(dataset) - len(y_pred)):, -1:] = y_pred
trade_dataset = dataset.dropna()
set_trace()
trade_dataset['Tomorrows Returns'] = 0.
trade_dataset['Tomorrows Returns'] = np.log(trade_dataset['close']/trade_dataset['close'].shift(1))
trade_dataset['Tomorrows Returns'] = trade_dataset['Tomorrows Returns'].shift(-1)

trade_dataset['Strategy Returns'] = 0.
trade_dataset['Strategy Returns'] = np.where(trade_dataset['y_pred'] is True, trade_dataset['Tomorrows Returns'], - trade_dataset['Tomorrows Returns'])

trade_dataset['Cumulative Market Returns'] = np.cumsum(trade_dataset['Tomorrows Returns'])
trade_dataset['Cumulative Strategy Returns'] = np.cumsum(trade_dataset['Strategy Returns'])

plt.figure(figsize=(10, 5))
plt.plot(trade_dataset['Cumulative Market Returns'], color='r', label='Market Returns')
plt.plot(trade_dataset['Cumulative Strategy Returns'], color='g', label='Strategy Returns')
plt.legend()
plt.show()




