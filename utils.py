import numpy as np
import pandas_ta as ta #extends Pandas with Technical Analysis (TA) indicators for financial data analysis.
import sklearn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

#The function returns two NumPy arrays: train and test, which contain the training and testing data, respectively, based on the specified date range and columns.
def train_test_split(dataset, tstart, tend, columns = ['High']):
    train = dataset.loc[f"{tstart}":f"{tend}", columns].values
    test = dataset.loc[f"{tend+1}":, columns].values
    return train, test
    

#function takes a sequence of data and a specified number of time steps (n_steps) and creates input-output pairs for sequence prediction
def split_sequence(sequence, n_steps):#  list or array representing the input sequence of data.,number of time steps to use as input (X) for predicting the output (y).
    X, y = list(), list()
    for i in range(len(sequence)):
        end_ix = i + n_steps
        if end_ix > len(sequence) - 1:
            break
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix] #The function then iterates over the sequence to create the input-output pairs
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

# calculates and prints the Root Mean Squared Error (RMSE) between the actual test data and the predicted data.
def return_rmse(test, predicted):
    rmse = np.sqrt(mean_squared_error(test, predicted))
    print("The root mean squared error is {:.2f}.".format(rmse))


def process_and_split_multivariate_data(dataset, tstart, tend, mv_features):
    multi_variate_df = dataset.copy()#ensures that the original dataset remains unchanged

    # Technical Indicators
    multi_variate_df['RSI'] = ta.rsi(multi_variate_df.Close, length=15) #The function calculates technical indicators, such as Relative Strength Index (RSI) and Exponential Moving Averages (EMAs), using the pandas_ta library.
    multi_variate_df['EMAF'] = ta.ema(multi_variate_df.Close, length=20)
    multi_variate_df['EMAM'] = ta.ema(multi_variate_df.Close, length=100)
    multi_variate_df['EMAS'] = ta.ema(multi_variate_df.Close, length=150)

    # Target Variable
    multi_variate_df['Target'] = multi_variate_df['Adj Close'] - dataset.Open #variable named 'Target' is created, representing the difference between the 'Adj Close' and 'Open' prices.
    multi_variate_df['Target'] = multi_variate_df['Target'].shift(-1) # target variable is then shifted one step forward to represent the future value to be predicted. 
    multi_variate_df.dropna(inplace=True) #Any rows with missing values are removed using dropna().

    # Drop unnecessary columns
    multi_variate_df.drop(['Volume', 'Close'], axis=1, inplace=True)

    # Plotting The code plots the 'High' price and the calculated RSI on one figure and the 'High' price, EMAF, EMAM, and EMAS on another figure for visualization purposes.
    multi_variate_df.loc[f"{tstart}":f"{tend}", ['High', 'RSI']].plot(figsize=(16, 4), legend=True)

    multi_variate_df.loc[f"{tstart}":f"{tend}", ['High', 'EMAF', 'EMAM', 'EMAS']].plot(figsize=(16, 4), legend=True)

    feat_columns = ['Open', 'High', 'RSI', 'EMAF', 'EMAM', 'EMAS'] #Lists feat_columns and label_col are defined to select the features and the target variable used for modeling.
    label_col = ['Target']

    # Splitting train and test data
    #The training set (mv_training_set) contains data up to the tend date, while the testing set (mv_test_set) contains data beyond the tend date.
    mv_training_set, mv_test_set = train_test_split(multi_variate_df, tstart, tend, feat_columns + label_col)

    X_train = mv_training_set[:, :-1]
    y_train = mv_training_set[:, -1]

    X_test = mv_test_set[:, :-1]
    y_test = mv_test_set[:, -1]

    # Scaling Data
    mv_sc = MinMaxScaler(feature_range=(0, 1))
    X_train = mv_sc.fit_transform(X_train).reshape(-1, 1, mv_features) #mv_features variable represents the number of features used for modeling
    X_test = mv_sc.transform(X_test).reshape(-1, 1, mv_features)

    return X_train, y_train, X_test, y_test, mv_sc