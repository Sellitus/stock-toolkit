from collections import deque
from keras_buoy.models import ResumableModel
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from yahoo_fin import stock_info as si
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional

import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import ta


def create_model(sequence_length, hidden_neurons=256, input_neurons=None, output_neurons=None, cell=LSTM,
                 n_hidden_layers=0, dropout=0.3, loss="mean_absolute_error", optimizer="rmsprop", bidirectional=False):

    if input_neurons is None:
        input_neurons = hidden_neurons
    if output_neurons is None:
        output_neurons = hidden_neurons

    model = Sequential()

    # Add first layer
    if bidirectional:
        model.add(Bidirectional(cell(input_neurons, return_sequences=True), input_shape=(None,
                                                                                         sequence_length)))
    else:
        model.add(cell(input_neurons, return_sequences=True, input_shape=(None, sequence_length)))
    model.add(Dropout(dropout))

    for i in range(n_hidden_layers):

        if bidirectional:
            model.add(Bidirectional(cell(hidden_neurons, return_sequences=True)))
        else:
            model.add(cell(hidden_neurons, return_sequences=True))
        # Add dropout after each layer
        model.add(Dropout(dropout))

    # Add last layer
    if bidirectional:
        model.add(Bidirectional(cell(output_neurons, return_sequences=False)))
    else:
        model.add(cell(output_neurons, return_sequences=False))
    model.add(Dropout(dropout))

    model.add(Dense(1, activation="linear"))
    model.compile(loss=loss, metrics=["mean_absolute_error"], optimizer=optimizer)
    return model


def load_data(ticker, n_steps=50, scale=True, shuffle=True, lookup_step=1,
                test_size=0.2, feature_columns=('adjclose', 'volume', 'open', 'high', 'low')):
    df = None
    # see if ticker is already a loaded stock from yahoo finance
    if isinstance(ticker, str):
        # load it from yahoo_fin library
        df = si.get_data(ticker)
    elif isinstance(ticker, pd.DataFrame):
        # already loaded, use it directly
        df = ticker

    # Clean NaN values
    df = ta.utils.dropna(df)

    # Add technical lib to dataset
    df = ta.add_all_ta_features(df, open="open", high="high", low="low", close="adjclose", volume="volume")

    # Replace NaN values with 0
    df = df.fillna(0)

    # this will contain all the elements we want to return from this function
    result = {}
    # we will also return the original dataframe itself
    result['df'] = df.copy()
    # make sure that the passed feature_columns exist in the dataframe

    for col in feature_columns:
        assert col in df.columns, f"'{col}' does not exist in the dataframe."
    if scale:
        column_scaler = {}
        # scale the data (prices) from 0 to 1
        for column in feature_columns:
            scaler = preprocessing.MinMaxScaler()
            df[column] = scaler.fit_transform(np.expand_dims(df[column].values, axis=1))
            column_scaler[column] = scaler

        # add the MinMaxScaler instances to the result returned
        result["column_scaler"] = column_scaler
    # add the target column (label) by shifting by `lookup_step`
    df['future'] = df['adjclose'].shift(-lookup_step)
    # last `lookup_step` columns contains NaN in future column
    # get them before droping NaNs
    last_sequence = np.array(df[feature_columns].tail(lookup_step))
    # drop NaNs
    df.dropna(inplace=True)
    sequence_data = []
    sequences = deque(maxlen=n_steps)
    for entry, target in zip(df[feature_columns].values, df['future'].values):
        sequences.append(entry)
        if len(sequences) == n_steps:
            sequence_data.append([np.array(sequences), target])
    # get the last sequence by appending the last `n_step` sequence with `lookup_step` sequence
    # for instance, if n_steps=50 and lookup_step=10, last_sequence should be of 59 (that is 50+10-1) length
    # this last_sequence will be used to predict in future dates that are not available in the dataset
    last_sequence = list(sequences) + list(last_sequence)
    # shift the last sequence by -1
    last_sequence = np.array(pd.DataFrame(last_sequence).shift(-1).dropna())
    # add to result
    result['last_sequence'] = last_sequence
    # construct the X's and y's
    X, y = [], []
    for seq, target in sequence_data:
        X.append(seq)
        y.append(target)
    # convert to numpy arrays
    X = np.array(X)
    y = np.array(y)
    # reshape X to fit the neural network
    if len(X.shape) == 3:
        X = X.reshape((X.shape[0], X.shape[2], X.shape[1]))
    if len(X.shape) == 2:
        X = X.reshape(X.shape[0], X.shape[1])
    if len(X.shape) == 1:
        X = X.reshape(X.shape[0])


    # split the dataset
    result["X_train"], result["X_test"], result["y_train"], result["y_test"] = train_test_split(X, y,
                                                                                                test_size=test_size,
                                                                                                shuffle=shuffle)
    # return the result
    return result


def parse_arguments():
    parser = argparse.ArgumentParser(description='Train that thing')

    parser.add_argument('--tickers', nargs="+", dest="TICKERS", required=True,
                        help="Stock tickers to train with. Ex: --tickers AMD GOOGL INTC")
    # parser.add_argument('--model-load', action="store_true", dest="arg_model_load", default=None,
    #                     help="Pass to load a model rather than create a new one")
    # parser.add_argument('--model-filename', action="store", dest="arg_model_filename", default=None,
    #                     help="Pass a model filename to either load from or save to (in results/ folder). Ex: "
    #                          "--model-filename model.h5")
    # parser.add_argument('--log-filename', action="store", dest="arg_log_filename", default=None,
    #                     help="Pass a log filename to either load from or save to (in logs/ folder). Ex: --log-filename "
    #                          "main")

    parser.add_argument('--model-name', action="store", dest="MODEL_NAME", default=None,
                        help="Pass a model name to either load from or save to (in logs/ and results/ folders). Ex: "
                             "--log-filename main")
    parser.add_argument('--ticker-epochs', action="store", type=int, dest="TICKER_EPOCHS", default=100,
                        help="Number of epochs per ticker passed")

    parser.add_argument('--delete-logs', action="store_true", dest="DELETE_LOGS", default=False,
                        help="Pass to load a model rather than create a new one")
    parser.add_argument('--resume-model', action="store_true", dest="RESUME_MODEL", default=False,
                        help="Pass to load a model rather than create a new one")
    parser.add_argument('--update-data', action="store_true", dest="UPDATE_DATA", default=False,
                        help="Updates the ticker dataset, even if it already exists.")
    parser.add_argument('--only-test', action="store_true", dest="ONLY_TEST_MODEL", default=False,
                        help="Skips training and plots the graph only")

    parser.add_argument('--n-steps', action="store", dest="N_STEPS", default=50, type=int,
                        help="Window size / sequence length. Ex: --n-steps 50")
    parser.add_argument('--lookup-step', action="store", dest="LOOKUP_STEP", default=1, type=int,
                        help="Number of steps in the future to predict. Ex: --lookup-step 1")
    parser.add_argument('--test-size', action="store", dest="TEST_SIZE", default=0.2, type=float,
                        help="Test ratio size, percentage between 0 and 1. Ex: --test-size 0.2")
    parser.add_argument('--n-hidden-layers', action="store", dest="N_HIDDEN_LAYERS", default=1, type=int,
                        help="Number of layers. Ex: --n-layers 1")
    parser.add_argument('--units', action="store", dest="UNITS", default=256, type=int,
                        help="Number of units. Ex: --units 256")
    parser.add_argument('--dropout', action="store", dest="DROPOUT", default=0.4, type=float,
                        help="Dropout rate, percentage between 0 and 1. Ex: --dropout 0.4")
    parser.add_argument('--not-bidirectional', action="store_true", dest="NOT_BIDIRECTIONAL", default=False,
                        help="Enables bidirectional RNNs. Ex: --bidirectional")
    parser.add_argument('--loss', action="store", dest="LOSS", default="huber_loss",
                        help="Loss type. Ex: --loss huber_loss")
    parser.add_argument('--optimizer', action="store", dest="OPTIMIZER", default="adam",
                        help="Optimizer to use. Ex: --optimizer adam")
    parser.add_argument('--batch-size', action="store", dest="BATCH_SIZE", default=64, type=int,
                        help="Batch size. Ex: --batch-size 64")
    parser.add_argument('--epochs', action="store", dest="EPOCHS", default=400, type=int,
                        help="Number of epochs. Ex: --epochs 400")
    args = parser.parse_args()


    variables = {}


    # Default: None
    variables['TICKERS'] = args.TICKERS
    # Default: 100
    variables['TICKER_EPOCHS'] = args.TICKER_EPOCHS
    # Default: False
    variables['DELETE_LOGS'] = args.DELETE_LOGS
    # Default: None
    variables['MODEL_NAME'] = args.MODEL_NAME
    variables['RESUME_MODEL'] = args.RESUME_MODEL
    variables['UPDATE_DATA'] = args.UPDATE_DATA
    variables['ONLY_TEST_MODEL'] = args.ONLY_TEST_MODEL

    # Window size or the sequence length
    # Default: 50
    variables['N_STEPS'] = args.N_STEPS
    # Lookup step, 1 is the next day
    # Default: 1
    variables['LOOKUP_STEP'] = args.LOOKUP_STEP
    # test ratio size, 0.2 is 20%
    # Default: 0.2
    variables['TEST_SIZE'] = args.TEST_SIZE
    # features to use
    # Default: ["adjclose", "volume", "open", "high", "low"]
    FEATURE_COLUMNS = ['open', 'high', 'low', 'close', 'adjclose', 'volume',
                       'volume_adi', 'volume_obv', 'volume_cmf', 'volume_fi', 'volume_mfi',
                       'volume_em', 'volume_sma_em', 'volume_vpt', 'volume_nvi', 'volume_vwap',
                       'volatility_atr', 'volatility_bbm', 'volatility_bbh', 'volatility_bbl',
                       'volatility_bbw', 'volatility_bbp', 'volatility_bbhi',
                       'volatility_bbli', 'volatility_kcc', 'volatility_kch', 'volatility_kcl',
                       'volatility_kcw', 'volatility_kcp', 'volatility_kchi',
                       'volatility_kcli', 'volatility_dcl', 'volatility_dch', 'volatility_dcm',
                       'volatility_dcw', 'volatility_dcp', 'volatility_ui', 'trend_macd',
                       'trend_macd_signal', 'trend_macd_diff', 'trend_sma_fast',
                       'trend_sma_slow', 'trend_ema_fast', 'trend_ema_slow', 'trend_adx',
                       'trend_adx_pos', 'trend_adx_neg', 'trend_vortex_ind_pos',
                       'trend_vortex_ind_neg', 'trend_vortex_ind_diff', 'trend_trix',
                       'trend_mass_index', 'trend_cci', 'trend_dpo', 'trend_kst',
                       'trend_kst_sig', 'trend_kst_diff', 'trend_ichimoku_conv',
                       'trend_ichimoku_base', 'trend_ichimoku_a', 'trend_ichimoku_b',
                       'trend_visual_ichimoku_a', 'trend_visual_ichimoku_b', 'trend_aroon_up',
                       'trend_aroon_down', 'trend_aroon_ind', 'trend_psar_up',
                       'trend_psar_down', 'trend_psar_up_indicator',
                       'trend_psar_down_indicator', 'trend_stc', 'momentum_rsi',
                       'momentum_stoch_rsi', 'momentum_stoch_rsi_k', 'momentum_stoch_rsi_d',
                       'momentum_tsi', 'momentum_uo', 'momentum_stoch',
                       'momentum_stoch_signal', 'momentum_wr', 'momentum_ao', 'momentum_kama',
                       'momentum_roc', 'momentum_ppo', 'momentum_ppo_signal',
                       'momentum_ppo_hist', 'others_dr', 'others_dlr', 'others_cr']
    variables['FEATURE_COLUMNS'] = FEATURE_COLUMNS


    ### MODEL PARAMETERS

    # Number of hidden layers
    # Default: 1
    variables['N_HIDDEN_LAYERS'] = args.N_HIDDEN_LAYERS
    # LSTM cell
    # Default: LSTM
    variables['CELL'] = LSTM
    # 256 LSTM neurons
    # Default: 256
    variables['UNITS'] = args.UNITS
    # 40% dropout
    # Default: 0.4
    variables['DROPOUT'] = args.DROPOUT
    # whether to use bidirectional RNNs
    # Default: False
    # Recommended: True
    if args.NOT_BIDIRECTIONAL is True:
        variables['BIDIRECTIONAL'] = False
    else:
        variables['BIDIRECTIONAL'] = True
    ### training parameters
    # mean absolute error loss
    # LOSS = "mae"
    # huber loss
    # Default: "huber_loss"
    variables['LOSS'] = args.LOSS
    # Default: "adam"
    variables['OPTIMIZER'] = args.OPTIMIZER
    # Default: 64
    variables['BATCH_SIZE'] = args.BATCH_SIZE
    # Default: 400
    variables['EPOCHS'] = args.EPOCHS
    return variables




def predict(model, data, N_STEPS, classification=False):
    # retrieve the last sequence from data
    last_sequence = data["last_sequence"][:N_STEPS]
    # retrieve the column scalers
    column_scaler = data["column_scaler"]
    # reshape the last sequence
    last_sequence = last_sequence.reshape((last_sequence.shape[1], last_sequence.shape[0]))
    # expand dimension
    last_sequence = np.expand_dims(last_sequence, axis=0)
    # get the prediction (scaled from 0 to 1)
    prediction = model.predict(last_sequence)
    # get the price (by inverting the scaling)
    predicted_price = column_scaler["adjclose"].inverse_transform(prediction)[0][0]
    return predicted_price


def get_accuracy(model, data, lookup_step):
    y_test = data["y_test"]
    X_test = data["X_test"]
    y_pred = model.predict(X_test)
    y_test = np.squeeze(data["column_scaler"]["adjclose"].inverse_transform(np.expand_dims(y_test, axis=0)))
    y_pred = np.squeeze(data["column_scaler"]["adjclose"].inverse_transform(y_pred))
    y_pred = list(map(lambda current, future: int(float(future) > float(current)), y_test[:-lookup_step],
                      y_pred[lookup_step:]))
    y_test = list(map(lambda current, future: int(float(future) > float(current)), y_test[:-lookup_step],
                      y_test[lookup_step:]))
    return accuracy_score(y_test, y_pred)


def plot_graph(model, data):
    y_test = data["y_test"]
    X_test = data["X_test"]
    y_pred = model.predict(X_test)
    y_test = np.squeeze(data["column_scaler"]["adjclose"].inverse_transform(np.expand_dims(y_test, axis=0)))
    y_pred = np.squeeze(data["column_scaler"]["adjclose"].inverse_transform(y_pred))
    # last 200 days, feel free to edit that
    plt.plot(y_test[-400:], c='b')
    plt.plot(y_pred[-400:], c='r')
    plt.xlabel("Days")
    plt.ylabel("Price")
    plt.legend(["Actual Price", "Predicted Price"])
    plt.show()

