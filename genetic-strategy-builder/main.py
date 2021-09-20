import argparse
import numpy as np
import os
import pandas as pd
import pickle
import random
import ta
import time

from sklearn import preprocessing
from yahoo_fin import stock_info as si


def load_data(ticker, feature_columns=('adjclose', 'volume', 'open', 'high', 'low'), scale=True):
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

    # Add technical indicators to dataset
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

    return result





# parse arguments
parser = argparse.ArgumentParser(description='Find That Setup')
parser.add_argument('--tickers', nargs="+", dest="TICKERS", required=True,
                    help="Stock tickers to find trading setups for. Ex: --tickers AMD GOOGL INTC")
args = parser.parse_args()

TICKERS = args.TICKERS
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


# Set randomizer seeds for consistent results
seed = 314
np.random.seed(seed)
random.seed(seed)

# Grab the current date
date_time_start = time.strftime("%Y-%m-%d_%H:%M:%S")

tickers = [ticker.upper() for ticker in TICKERS]









if not os.path.isdir("data"):
    os.mkdir("data")


data = []
for ticker in tickers:
    ticker_data_filename = os.path.join("data", f"{ticker}.csv")
    # Load the data from disk if it exists, otherwise pull info from Yahoo Finance
    if os.path.exists(ticker_data_filename):
        print("Loading Ticker History: {}".format(ticker_data_filename))
        curr_data = pickle.load(open(ticker_data_filename, "rb"))

        data.append(curr_data)
    else:
        print("Downloading Ticker History: {}".format(ticker))
        curr_data = load_data(ticker, feature_columns=FEATURE_COLUMNS)

        data.append(curr_data)
        # Save the dataframe to prevent fetching every run
        pickle.dump(curr_data, open(ticker_data_filename, "wb"))








import pdb
pdb.set_trace()