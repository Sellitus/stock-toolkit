""":"

# Find a suitable python interpreter (adapt for your specific needs)
for cmd in ~/anaconda3/envs/stock-toolkit/bin/python ; do
   command -v > /dev/null $cmd && exec $cmd $0 "$@"
done

echo "Python not found!" >2

exit 2

":"""

from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from keras_buoy.models import ResumableModel

import numpy as np
import tensorflow as tf

import time
import os
import pickle
import random
import shutil

from utils import parse_arguments, create_model, load_data, predict, get_accuracy, plot_graph




# Set randomizer seeds for consistent results
seed = 314
np.random.seed(seed)
tf.random.set_seed(seed)
random.seed(seed)

# Grab the current date
date_now = time.strftime("%Y-%m-%d_%H:%M:%S")


#!!!!!!!!!!!!! READ ARGS, OR IF CONFIG FILE IS PASSED, USE OPTIONS IN CONFIG FILE INSTEAD
a = parse_arguments()


# Save args to variables for easier reading / usage
TICKERS = a['TICKERS']; DELETE_LOGS = a['DELETE_LOGS']; MODEL_NAME = a['MODEL_NAME']; RESUME_MODEL = a['RESUME_MODEL']
UPDATE_DATA = a['UPDATE_DATA']; N_STEPS = a['N_STEPS']; LOOKUP_STEP = a['LOOKUP_STEP']; TEST_SIZE = a['TEST_SIZE']
FEATURE_COLUMNS = a['FEATURE_COLUMNS']; N_HIDDEN_LAYERS = a['N_HIDDEN_LAYERS']; CELL = a['CELL']; UNITS = a['UNITS']
DROPOUT = a['DROPOUT']; BIDIRECTIONAL = a['BIDIRECTIONAL']; LOSS = a['LOSS']; OPTIMIZER = a['OPTIMIZER']
BATCH_SIZE = a['BATCH_SIZE']; EPOCHS = a['EPOCHS']; ONLY_TEST_MODEL = a['ONLY_TEST_MODEL']; TICKER_EPOCHS = a['TICKER_EPOCHS']



tickers = [ticker.upper() for ticker in TICKERS]
tickers_str = '-'.join(tickers)

# Name of the model
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


model = ResumableModel(model, save_every_epochs=4, custom_objects=None, to_path=filename_model)


if ONLY_TEST_MODEL is False:
    # some tensorflow callbacks
    checkpointer = ModelCheckpoint(filename_model, save_weights_only=False, save_best_only=True, verbose=1)
    if MODEL_NAME is not None:
        tensorboard = TensorBoard(log_dir=os.path.join("logs", MODEL_NAME))
    else:
        tensorboard = TensorBoard(log_dir=os.path.join("logs", model_name_specs))



    if len(tickers) == 1:
        model.fit(curr_data["X_train"], curr_data["y_train"],
                  batch_size=BATCH_SIZE,
                  epochs=EPOCHS,
                  validation_data=(curr_data["X_test"], curr_data["y_test"]),
                  callbacks=[checkpointer, tensorboard],
                  verbose=1)
    else:
        epoch_count = 0
        while epoch_count < EPOCHS:
            for i in range(len(tickers)):
                print('Training with ticker: {}'.format(tickers[i]))
                curr_data = data[i]
                model.fit(curr_data["X_train"], curr_data["y_train"],
                          batch_size=BATCH_SIZE,
                          epochs=TICKER_EPOCHS,
                          validation_data=(curr_data["X_test"], curr_data["y_test"]),
                          callbacks=[checkpointer, tensorboard],
                          verbose=1)
                epoch_count += TICKER_EPOCHS
                if epoch_count >= EPOCHS:
                    break


    # Save model to disk
    if os.path.exists(filename_model):
        shutil.move(filename_model, os.path.join("results", MODEL_NAME + date_now + '.h5'))

    model.save(filename_model)


# Now test the model
data = load_data(tickers[0], N_STEPS, lookup_step=LOOKUP_STEP, test_size=TEST_SIZE,
                 feature_columns=FEATURE_COLUMNS, shuffle=False)


model.load_weights(filename_model)

# evaluate the model
mse, mae = model.evaluate(data["X_test"], data["y_test"], verbose=0)
# calculate the mean absolute error (inverse scaling)
mean_absolute_error = data["column_scaler"]["adjclose"].inverse_transform([[mae]])[0][0]
print("Mean Absolute Error:", mean_absolute_error)


# predict the future price
future_price = predict(model, data, N_STEPS)
print(f"Future price after {LOOKUP_STEP} days is {future_price:.2f}$")


# Print accuracy
print(str(LOOKUP_STEP) + ":", "Accuracy Score:", get_accuracy(model, data, LOOKUP_STEP))

# Plot results
plot_graph(model, data)
