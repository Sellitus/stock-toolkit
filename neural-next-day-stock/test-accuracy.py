""":"

# Find a suitable python interpreter (adapt for your specific needs)
for cmd in ~/anaconda3/envs/stock-toolkit/bin/python ; do
   command -v > /dev/null $cmd && exec $cmd $0 "$@"
done

echo "Python not found!" >2

exit 2

":"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard

import numpy as np
import pandas as pd
import tensorflow as tf

import argparse
import time
import os
import pickle
import random
import shutil

from utils import load_data, predict, get_accuracy, plot_graph



# Disable GPU
tf.config.experimental.set_visible_devices([], 'GPU')



# Set seed for consistent results
seed = 314
np.random.seed(seed)
tf.random.set_seed(seed)
random.seed(seed)





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
parser.add_argument('--plot', action="store_true", dest="PLOT_GRAPH", default=False,
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


# Default: None
TICKERS = args.TICKERS
# Default: 100
TICKER_EPOCHS = args.TICKER_EPOCHS
# Default: False
DELETE_LOGS = args.DELETE_LOGS
# Default: None
MODEL_NAME = args.MODEL_NAME
RESUME_MODEL = args.RESUME_MODEL
UPDATE_DATA = args.UPDATE_DATA
PLOT_GRAPH = args.PLOT_GRAPH


# Window size or the sequence length
# Default: 50
N_STEPS = args.N_STEPS
# Lookup step, 1 is the next day
# Default: 1
LOOKUP_STEP = args.LOOKUP_STEP
# test ratio size, 0.2 is 20%
# Default: 0.2
TEST_SIZE = args.TEST_SIZE
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
# date now
# Default: time.strftime("%Y-%m-%d")
date_now = time.strftime("%Y-%m-%d_%H:%M:%S")
### model parameters
# Number of hidden layers
# Default: 1
N_HIDDEN_LAYERS = args.N_HIDDEN_LAYERS
# LSTM cell
# Default: LSTM
CELL = LSTM
# 256 LSTM neurons
# Default: 256
UNITS = args.UNITS
# 40% dropout
# Default: 0.4
DROPOUT = args.DROPOUT
# whether to use bidirectional RNNs
# Default: False
# Recommended: True
if args.NOT_BIDIRECTIONAL is True:
    BIDIRECTIONAL = False
else:
    BIDIRECTIONAL = True
### training parameters
# mean absolute error loss
# LOSS = "mae"
# huber loss
# Default: "huber_loss"
LOSS = args.LOSS
# Default: "adam"
OPTIMIZER = args.OPTIMIZER
# Default: 64
BATCH_SIZE = args.BATCH_SIZE
# Default: 400
EPOCHS = args.EPOCHS


tickers = [ticker.upper() for ticker in TICKERS]
tickers_str = '-'.join(tickers)

# model name to save, making it as unique as possible based on parameters
model_name_specs = f"{tickers_str}_{date_now}-{LOSS}-{OPTIMIZER}-{CELL.__name__}-seq-{N_STEPS}-step-{LOOKUP_STEP}-layers-" \
                   f"{N_HIDDEN_LAYERS}-units-{UNITS}"
if BIDIRECTIONAL:
    model_name_specs += "-b"


if MODEL_NAME is None:
    MODEL_NAME = model_name_specs

filename_model = os.path.join("results", MODEL_NAME) + ".h5"


# Delete all previous model data if --delete-data is passed
if DELETE_LOGS is True:
    try:
        shutil.rmtree(os.path.join("logs", MODEL_NAME))
    except Exception:
        pass
else:
    if os.path.isdir(os.path.join("logs", MODEL_NAME)):
        shutil.move(os.path.join("logs", MODEL_NAME), os.path.join("logs", MODEL_NAME + date_now))


# create these folders if they does not exist
if not os.path.isdir("results"):
    os.mkdir("results")
if not os.path.isdir("logs"):
    os.mkdir("logs")
if not os.path.isdir("data"):
    os.mkdir("data")

data = []
for ticker in tickers:
    ticker_data_filename = os.path.join("data", f"{ticker}.csv")
    # Load the data from disk if it exists or --update-data is not passed, otherwise pull info from Yahoo Finance
    if os.path.exists(ticker_data_filename) and UPDATE_DATA is False:
        print("Loading Ticker History: {}".format(ticker_data_filename))
        curr_data = pickle.load(open(ticker_data_filename, "rb"))

        data.append(curr_data)
    else:
        print("Downloading Ticker History: {}".format(ticker))
        curr_data = load_data(ticker, N_STEPS, lookup_step=LOOKUP_STEP, test_size=TEST_SIZE, feature_columns=FEATURE_COLUMNS)

        data.append(curr_data)
        # Save the dataframe to prevent fetching every run
        pickle.dump(curr_data, open(ticker_data_filename, "wb"))


if RESUME_MODEL is True and os.path.exists(filename_model):
    model = tf.keras.models.load_model(filename_model)
else:
    # construct the model
    model = create_model(N_STEPS, loss=LOSS, hidden_neurons=UNITS, cell=CELL,
                         n_hidden_layers=N_HIDDEN_LAYERS, dropout=DROPOUT, optimizer=OPTIMIZER,
                         bidirectional=BIDIRECTIONAL)


# if PLOT_GRAPH is False:
#     # some tensorflow callbacks
#     checkpointer = ModelCheckpoint(filename_model, save_weights_only=False, save_best_only=True, verbose=1)
#     if MODEL_NAME is not None:
#         tensorboard = TensorBoard(log_dir=os.path.join("logs", MODEL_NAME))
#     else:
#         tensorboard = TensorBoard(log_dir=os.path.join("logs", model_name_specs))
#
#
#
#     if len(tickers) == 1:
#         model.fit(curr_data["X_train"], curr_data["y_train"],
#                   batch_size=BATCH_SIZE,
#                   epochs=EPOCHS,
#                   validation_data=(curr_data["X_test"], curr_data["y_test"]),
#                   callbacks=[checkpointer, tensorboard],
#                   verbose=1)
#     else:
#         epoch_count = 0
#         while epoch_count < EPOCHS:
#             for i in range(len(tickers)):
#                 print('Training with ticker: {}'.format(tickers[i]))
#                 curr_data = data[i]
#                 model.fit(curr_data["X_train"], curr_data["y_train"],
#                           batch_size=BATCH_SIZE,
#                           epochs=TICKER_EPOCHS,
#                           validation_data=(curr_data["X_test"], curr_data["y_test"]),
#                           callbacks=[checkpointer, tensorboard],
#                           verbose=1)
#                 epoch_count += TICKER_EPOCHS
#                 if epoch_count >= EPOCHS:
#                     break
#
#
#     # Save model to disk
#     if os.path.exists(filename_model):
#         shutil.move(filename_model, os.path.join("results", MODEL_NAME + date_now + '.h5'))
#
#     model.save(filename_model)


# Now test the model
data = load_data(tickers[0], N_STEPS, lookup_step=LOOKUP_STEP, test_size=TEST_SIZE,
                 feature_columns=FEATURE_COLUMNS, shuffle=False)


model.load_weights(filename_model)

# FIX THIS AND REENABLE
# # evaluate the model
# mse, mae = model.evaluate(data["X_test"], data["y_test"], verbose=0)
# # calculate the mean absolute error (inverse scaling)
# mean_absolute_error = data["column_scaler"]["adjclose"].inverse_transform([[mae]])[0][0]
# print("Mean Absolute Error:", mean_absolute_error)


# predict the future price
future_price = predict(model, data, N_STEPS)
print(f"Future price after {LOOKUP_STEP} days is {future_price:.2f}$")


# Print accuracy
print(str(LOOKUP_STEP) + ":", "Accuracy Score:", get_accuracy(model, data, LOOKUP_STEP))

# Plot results
plot_graph(model, data)
