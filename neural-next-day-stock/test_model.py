""":"

# Find a suitable python interpreter (adapt for your specific needs)
for cmd in ~/anaconda3/envs/stock-toolkit/bin/python ; do
   command -v > /dev/null $cmd && exec $cmd $0 "$@"
done

echo "Python not found!" >2

exit 2

":"""

# Force CPU by disabling the GPU
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


from utils import load_data, predict, get_accuracy, plot_graph

import tensorflow as tf

import argparse
import os


parser = argparse.ArgumentParser(description='Test that thing')

parser.add_argument('--ticker', action="store", dest="TICKER", default=None,
                    help="Stock ticker to train with")
parser.add_argument('--lookup-step', action="store", dest="LOOKUP_STEP", default=1, type=int,
                    help="Number of steps in the future to predict. Ex: --lookup-step 1")
parser.add_argument('--test-size', action="store", dest="TEST_SIZE", default=0.2, type=float,
                    help="Test ratio size, percentage between 0 and 1. Ex: --test-size 0.2")
parser.add_argument('--model-name', action="store", dest="MODEL_NAME", default=None,
                    help="Pass a model name to either load from or save to (in logs/ and results/ folders). Ex: "
                         "--log-filename main")

args = parser.parse_args()


if args.MODEL_NAME is None:
    raise Exception("No model_name passed, use --model-name <name>")

TICKER = args.TICKER
if TICKER is not None:
    ticker = TICKER.upper()
else:
    ticker = "AMD"

LOOKUP_STEP = args.LOOKUP_STEP
TEST_SIZE = args.TEST_SIZE
MODEL_NAME = args.MODEL_NAME


filename_model = os.path.join("results", MODEL_NAME) + ".h5"

FEATURE_COLUMNS = ["adjclose", "volume", "open", "high", "low",
       'volume_adi', 'volume_obv', 'volume_cmf', 'volume_fi', 'momentum_mfi',
       'volume_em', 'volume_sma_em', 'volume_vpt', 'volume_nvi', 'volume_vwap',
       'volatility_atr', 'volatility_bbm', 'volatility_bbh', 'volatility_bbl',
       'volatility_bbw', 'volatility_bbp', 'volatility_bbhi',
       'volatility_bbli', 'volatility_kcc', 'volatility_kch', 'volatility_kcl',
       'volatility_kcw', 'volatility_kcp', 'volatility_kchi',
       'volatility_kcli', 'volatility_dcl', 'volatility_dch', 'trend_macd',
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
       'trend_psar_down_indicator', 'momentum_rsi', 'momentum_tsi',
       'momentum_uo', 'momentum_stoch', 'momentum_stoch_signal', 'momentum_wr',
       'momentum_ao', 'momentum_kama', 'momentum_roc', 'others_dr',
       'others_dlr', 'others_cr']


model = tf.keras.models.load_model(filename_model)
N_STEPS = model.variables[0].shape[0]

# Now test the model
data = load_data(ticker, N_STEPS, lookup_step=LOOKUP_STEP, test_size=TEST_SIZE,
                feature_columns=FEATURE_COLUMNS, shuffle=False)


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
