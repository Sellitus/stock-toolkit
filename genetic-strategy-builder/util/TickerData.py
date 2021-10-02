
import multiprocessing as mp
import numpy as np
import os
import pandas as pd
import pickle
import ta

from sklearn import preprocessing
from util.TrendIndicators import TrendIndicators as ti
from util.MomentumIndicators import MomentumIndicators as mi
from yahoo_fin import stock_info as si
import yfinance as yf



# Data that retrieves and contains data for a specified ticker
class TickerData:
    """
    Downloads or loads all daily data for specified stock tickers
    """

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

    basic_feature_columns = ('adjclose', 'volume', 'open', 'high', 'low', 'close')

    intervals = {'1m': '7d', '2m': '60d', '5m': '60d', '15m': '60d', '30m': '60d', '60m': '730d', '90m': '60d',
                 '1h': '730d', '1d': 'max', '5d': 'max', '1wk': 'max', '1mo': 'max', '3mo': 'max'}

    def __init__(self, tickers=None, interval='1d'):
        self.data = None
        self.scalers = {}
        self.indicator_settings = {}

        if tickers is not None:
            self.data = self.load_ticker_data_yf(tickers, interval)

    def load_ticker_data_yf(self, tickers, interval='1d'):
        """
        Download the stock data from Yahoo Finance if the stock data exists. if it is already downloaded, load and use
        that data
        """

        if interval not in self.intervals.keys():
            print("Interval '{}' not recognized. Must be from this list: {}".format(interval, self.intervals.keys()))
            exit()

        data = {}
        for ticker in tickers:
            ticker_data_filename = os.path.join(os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))
                                                                 ), 'data'), f"{ticker}-{interval}.csv")
            #ticker_data_filename = os.path.join("../../data", f"{ticker}.csv")
            # Load the data from disk if it exists, otherwise pull info from Yahoo Finance
            if os.path.exists(ticker_data_filename):
                print("Loading Ticker History: {}".format(ticker_data_filename))
                curr_data_df = pickle.load(open(ticker_data_filename, "rb"))
            else:
                print("Downloading Ticker History: {}".format(ticker))
                #curr_data = load_data(ticker, feature_columns=FEATURE_COLUMNS)

                curr_data_df = None
                # see if ticker is already a loaded stock from yahoo finance
                if isinstance(ticker, str):
                    # load it from yahoo_fin library
                    # OLD LIBRARY FOR DOWNLOADING: curr_data_df = si.get_data(ticker)
                    curr_data_df = yf.download(ticker, interval=interval, period='max')
                elif isinstance(ticker, pd.DataFrame):
                    # already loaded, use it directly
                    curr_data_df = ticker

                # Rename the columns to the standard format
                renamed_columns = {}
                for i in range(len(curr_data_df.columns)):
                    renamed_columns[curr_data_df.columns[i]] = str(curr_data_df.columns[i]).lower().replace(' ', '')
                curr_data_df = curr_data_df.rename(columns=renamed_columns)

                # Save the dataframe to prevent fetching every run
                pickle.dump(curr_data_df, open(ticker_data_filename, "wb"))

            data[ticker] = curr_data_df
            if 'ticker' in curr_data_df.columns:
                data[ticker] = curr_data_df.drop(columns='ticker')

        self.data = data
        return data

    def add_technical_indicators_to_dataset(self, data=None, feature_columns=('adjclose', 'volume', 'open', 'high',
                                                                              'low', 'close'), scale=False):
        """
        Adds all of the technical library from the Python ta library with the default settings.
        """
        if data is None:
            data = self.data

        for ticker in data:
            df = self.data[ticker]

            # Clean NaN values
            df = ta.utils.dropna(df)
            # df = df.replace([0], 0.000000001)

            # Add technical library to dataset
            with np.errstate(divide='ignore'):
                df = ta.add_all_ta_features(df, open="open", high="high", low="low", close="adjclose", volume="volume",
                                            fillna=True)

            all_indicators = ti.trend_dna + mi.momentum_dna
            for indicator in all_indicators:
                # Save each indicator's settings
                self.indicator_settings[str(indicator)] = indicator.get_settings()

            # Replace NaN values with 0
            df = df.fillna(0)

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
                self.scalers[ticker] = column_scaler
            self.data[ticker] = df

        return self.data

    def add_individual_indicators_to_dataset(self, data=None, randomize=False,
                                             feature_columns=('adjclose', 'volume', 'open', 'high', 'low', 'close'),
                                             scale=False):
        if data is None:
            data = self.data
        else:
            self.data = data
        if randomize is True:
            randomize = 0.1

        for ticker in data:
            df = data[ticker]

            # Clean NaN values
            #df = ta.utils.dropna(df)
            # df = df.replace([0], 0.000000001)

            all_indicators = ti.trend_dna + mi.momentum_dna
            for indicator in all_indicators:
                with np.errstate(divide='ignore'):
                    # Append indicator results to dataframe
                    df = indicator.add_indicator(df, randomize=randomize)
                    # Save each indicator's settings
                    self.indicator_settings[str(indicator)] = indicator.get_settings()

            # Replace NaN values with 0
            df = df.fillna(0)

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
                self.scalers[ticker] = column_scaler
            self.data[ticker] = df

        return self.data

    def clear_ticker_data(self, data=None, feature_columns=('adjclose', 'volume', 'open', 'high', 'low', 'close')):
        if data is None:
            data = self.data

        for ticker in data:
            df = self.data[ticker]

            # Drop all columns
            df.drop(columns=df.columns.difference(feature_columns))

        return data
