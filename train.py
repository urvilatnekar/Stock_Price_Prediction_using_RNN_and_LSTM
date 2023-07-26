from datetime import datetime

import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import keras
import yfinance as yf
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.preprocessing import MinMaxScaler
from pandas_datareader.data import DataReader
from pandas_datareader import data as pdr


from keras.layers import LSTM, SimpleRNN
from keras.models import Sequential
from keras.layers import Dense
from .utils import split_sequence

n_steps = 1 #Represents the number of time steps used as input for the model during training and prediction.
features = 1 #Represents the number of features used in the input data. Here, it is set to 1, suggesting that the input data has only one feature.

def sequence_generation(dataset: pd.DataFrame, sc: MinMaxScaler, model:Sequential, steps_future: int, test_set):
#The function begins by extracting the historical "High" price data from the dataset that corresponds to the same length as the test_set.
# It then applies the same scaling transformation that was used during the model training, converting the "High" price data to a scaled format.
    high_dataset = dataset.iloc[len(dataset) - len(test_set) - n_steps:]["High"]
    high_dataset = sc.transform(high_dataset.values.reshape(-1, 1))
    inputs = high_dataset[:n_steps]
#It initializes an inputs array with the last n_steps elements from the scaled "High" dataset. This will be the initial input sequence for the prediction.
    for _ in range(steps_future):
        curr_pred = model.predict(inputs[-n_steps:].reshape(-1, n_steps, features), verbose=0)
        inputs = np.append(inputs, curr_pred, axis=0)
#the function returns the predicted values for the future time steps. The sc.inverse_transform method is applied to revert the scaled predictions back to their original scale 
    return sc.inverse_transform(inputs[n_steps:])

def train_rnn_model(X_train, y_train, n_steps, features, sc,test_set,dataset, epochs=10, batch_size=32, verbose=1,steps_in_future = 25,save_model_path=None):
    model = Sequential()# building an RNN model using the Keras Sequential API.
    model.add(SimpleRNN(units=125, input_shape=(n_steps, features)))#consists of a SimpleRNN layer with 125 units (neurons) and an input shape of (n_steps, features)
    model.add(Dense(units=1))#output of the SimpleRNN layer is passed to a Dense layer with one unit (output node).
    model.compile(optimizer="RMSprop", loss="mse")
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=verbose)# trains the RNN model using the fit method
    #training progress and results are printed based on the value of the verbose parameter.
    
    # Scaling
    inputs = sc.transform(test_set.reshape(-1, 1))

    # Split into samples
    X_test, y_test = split_sequence(inputs, n_steps)
    # reshape
    X_test = X_test.reshape(-1, n_steps, features)

    # Prediction
    predicted_stock_price = model.predict(X_test)
    # Inverse transform the values
    predicted_stock_price = sc.inverse_transform(predicted_stock_price)
    
    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(y_test, predicted_stock_price))
    print("The root mean squared error is {:.2f}.".format(rmse))
    
    # Call sequence_generation function
    
    results = sequence_generation(dataset, sc, model, steps_in_future,test_set)#function calls the sequence_generation function to generate a sequence of future predictions using the trained RNN model for the specified number of steps_in_future
    print("Generated sequence of future predictions:")
    print(results)
    
    if save_model_path:
        model.save(save_model_path)
        print("Model saved successfully.")
    
    return model



def train_lstm_model(X_train, y_train, n_steps, features, sc, test_set,dataset, epochs=10, batch_size=32, verbose=1,steps_in_future = 25, save_model_path=None):
    model = Sequential() #builds an LSTM model using the Keras Sequential API with 125 units in the LSTM layer and an input shape of (n_steps, features)
    model.add(LSTM(units=125, input_shape=(n_steps, features)))
    model.add(Dense(units=1)) #The output of the LSTM layer is passed to a Dense layer with one unit for regression. 
    model.compile(optimizer="RMSprop", loss="mse") #The model is compiled with the RMSprop optimizer and Mean Squared Error (MSE) loss function.
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=verbose)
    
    # Scaling
    inputs = sc.transform(test_set.reshape(-1, 1))

    # Split into samples
    X_test, y_test = split_sequence(inputs, n_steps)
    # reshape
    X_test = X_test.reshape(-1, n_steps, features)

    # Prediction
    predicted_stock_price = model.predict(X_test)
    # Inverse transform the values
    predicted_stock_price = sc.inverse_transform(predicted_stock_price)
    
    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(y_test, predicted_stock_price))
    print("The root mean squared error is {:.2f}.".format(rmse))
    
    # Call sequence_generation function
    
    results = sequence_generation(dataset, sc, model, steps_in_future,test_set)# the sequence_generation function to generate a sequence of future predictions using the trained LSTM model for the specified number of steps_in_future.
    print("Generated sequence of future predictions:")
    print(results)
    
    if save_model_path:
        model.save(save_model_path)
        print("Model saved successfully.")
    
    return model



#The first function is suitable for univariate time series forecasting, while the second function is designed for multivariate time series forecasting where multiple features are used for prediction.
def train_multivariate_lstm(X_train, y_train, X_test, y_test, mv_features,mv_sc, save_model_path=None):
    model_mv = Sequential()
    model_mv.add(LSTM(units=125, input_shape=(1, mv_features)))
    model_mv.add(Dense(units=1))

    # Compiling the model
    model_mv.compile(optimizer="RMSprop", loss="mse")

    history = model_mv.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)

    predictions = model_mv.predict(X_test)

    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    print("The root mean squared error is {:.2f}.".format(rmse))
    
    if save_model_path:
        model_mv.save(save_model_path)
        print("Model saved successfully.")

    return model_mv