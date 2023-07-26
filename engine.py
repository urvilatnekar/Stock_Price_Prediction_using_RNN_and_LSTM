# ignore warnings
import warnings
warnings.filterwarnings('ignore')
from ml_pipeline.utils import *
from ml_pipeline.train import *
import yfinance as yf
import numpy as np
from projectpro import model_snapshot, checkpoint

yf.pdr_override()
### Loading data

dataset = pdr.get_data_yahoo('AAPL', start='2012-01-01', end=datetime.now())
checkpoint('34db30')
print("Data Loaded")
tstart = 2016
tend = 2020

training_set, test_set = train_test_split(dataset, tstart, tend)

"""### Scaling dataset values"""

sc = MinMaxScaler(feature_range=(0, 1))
training_set = training_set.reshape(-1, 1)
training_set_scaled = sc.fit_transform(training_set)

"""### Creating overlapping window batches"""

n_steps = 1
features = 1

X_train, y_train = split_sequence(training_set_scaled, n_steps)

# Reshaping X_train for model
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], features)


# Training the RNN model and saving it
model_rnn = train_rnn_model(X_train, y_train, n_steps, features,sc,test_set,dataset, epochs=10, batch_size=32, verbose=1,steps_in_future = 25, save_model_path="output/model_rnn.h5")
model_snapshot("34db30")

# Training the LSTM model and saving it
model_lstm = train_lstm_model(X_train, y_train, n_steps, features,sc,test_set,dataset, epochs=10, batch_size=32, verbose=1, steps_in_future = 25, save_model_path="output/model_lstm.h5")


# Multivariate Input
mv_features = 6

X_train, y_train, X_test, y_test, mv_sc = process_and_split_multivariate_data(dataset,tstart, tend, mv_features)


model_mv = train_multivariate_lstm(X_train, y_train, X_test, y_test, mv_features, mv_sc, save_model_path="output/model_mv_lstm.h5")
model_snapshot("34db30")


