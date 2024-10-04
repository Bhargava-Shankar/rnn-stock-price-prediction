# Stock Price Prediction

## AIM

To develop a Recurrent Neural Network model for stock price prediction.

## Problem Statement and Dataset
Develop a Recurrent Neural Network (RNN) model to predict the stock prices of Google. The goal is to train the model using historical stock price data and then evaluate its performance on a separate test dataset.

Dataset: The dataset consists of two CSV files:

Trainset.csv: This file contains historical stock price data of Google, which will be used for training the RNN model. It includes features such as the opening price of the stock. Testset.csv: This file contains additional historical stock price data of Google, which will be used for testing the trained RNN model. Similarly, it includes features such as the opening price of the stock. The objective is to build a model that can effectively learn from the patterns in the training data to make accurate predictions on the test data.

## Design Steps

1. **Define Objective**: Aim to predict Google stock prices using a Simple RNN model based on historical data.
2. **Data Collection**: Gather training (trainset.csv) and testing (testset.csv) datasets.
3. **Data Preprocessing**: Normalize data with MinMaxScaler and create input-output pairs using a rolling window of 60 days.
4. **Model Development**: Build and compile a Sequential RNN model with a SimpleRNN layer and a Dense output layer.
5. **Training and Prediction**: Train the model on the training set and evaluate it by predicting stock prices on the test set, followed by visualizing results.

## Program
#### Name Bhargava S
#### Register Number: 212221040029
```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras import layers
from keras.models import Sequential
dataset_train = pd.read_csv('/content/trainset.csv')
print(dataset_train.columns)
print(dataset_train.head())
train_set = dataset_train.iloc[:, 1:2].values
print(type(train_set))
print(train_set.shape)
sc = MinMaxScaler(feature_range=(0, 1))
training_set_scaled = sc.fit_transform(train_set)
print(training_set_scaled.shape)

# Creating the X_train and y_train datasets
X_train_array = []
y_train_array = []
for i in range(60, 1259):
    X_train_array.append(training_set_scaled[i-60:i, 0])
    y_train_array.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train_array), np.array(y_train_array)
X_train1 = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
print(X_train.shape)
length = 60
n_features = 1
model = Sequential()
model.add(layers.SimpleRNN(50, input_shape=(length, n_features)))
model.add(layers.Dense(1))
model.compile(optimizer='adam', loss='mse')
print("Name: Bhargava S      Register Number: 212221040029")
model.summary()
model.fit(X_train1, y_train, epochs=100, batch_size=32)
dataset_test = pd.read_csv('/content/testset.csv')
test_set = dataset_test.iloc[:, 1:2].values
print(test_set.shape)

dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis=0)
inputs = dataset_total.values
inputs = inputs.reshape(-1, 1)
inputs_scaled = sc.transform(inputs)

# Creating X_test dataset
X_test = []
for i in range(60, 1384):
    X_test.append(inputs_scaled[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
print(X_test.shape)
predicted_stock_price_scaled = model.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price_scaled)

# Plotting the results
print("Name: Bhargava S         Register Number: 212221040029")
plt.plot(np.arange(0, 1384), inputs, color='red', label='Test(Real) Google stock price')
plt.plot(np.arange(60, 1384), predicted_stock_price, color='blue', label='Predicted Google stock price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()

```

## Output

### True Stock Price, Predicted Stock Price vs time

![image](https://github.com/user-attachments/assets/306b6ed0-a7c7-4803-bc12-5041bc6e90de)


### Mean Square Error

![image](https://github.com/user-attachments/assets/886db1c8-5655-4dfb-b862-e012c8565c5f)


## Result
The code performs Google stock price prediction using a SimpleRNN model by training on past 60-day windows of stock data, making predictions on test data, and visualizing the predicted vs. real stock prices over time.
