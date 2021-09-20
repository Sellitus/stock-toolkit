import pandas as pd
import pickle
import ta

from sklearn import preprocessing
from yahoo_fin import stock_info as si



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

    basic_feature_columns = ('adjclose', 'volume', 'open', 'high', 'low')

    def __init__(self, tickers=None):
        self.data = None

        if tickers is not None:
            self.data = self.load_stock_data_yf(tickers)

    def load_stock_data_yf(self, tickers, feature_columns=None, scale=True):
        """
        Download the stock data from Yahoo Finance if the stock data exists. if it is already downloaded, load and use
        that data
        """

        if feature_columns is None:
            feature_columns = self.FEATURE_COLUMNS

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
                #curr_data = load_data(ticker, feature_columns=FEATURE_COLUMNS)

                curr_data_df = None
                # see if ticker is already a loaded stock from yahoo finance
                if isinstance(ticker, str):
                    # load it from yahoo_fin library
                    curr_data_df = si.get_data(ticker)
                elif isinstance(ticker, pd.DataFrame):
                    # already loaded, use it directly
                    curr_data_df = ticker

                curr_data_df = self.add_technical_indicators_to_dataset(curr_data_df, feature_columns=feature_columns,
                                                                        scale=scale)

                # Save the dataframe to prevent fetching every run
                pickle.dump(curr_data_df, open(ticker_data_filename, "wb"))

                data.append(curr_data_df)

        self.data = data
        return data

    def add_technical_indicators_to_dataset(self, df, feature_columns=('adjclose', 'volume', 'open', 'high', 'low'),
                                            scale=True):
        """
        Adds all of the technical indicators from the Python ta library with the default settings.
        """
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
