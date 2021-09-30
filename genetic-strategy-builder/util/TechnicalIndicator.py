from abc import ABC, abstractmethod


class TechnicalIndicator(ABC):
    """
    Represents a single technical indicator. May be one of many different types and settings.
    """

    # 'NEUTRAL' means no action should be taken, 'ACTION' means either a buy or sell order should be placed
    buy_sell = ['BUY', 'SELL', 'NEUTRAL', 'ACTION', 'INACTION']

    def __init__(self, df=None):
        self.df = {}
        if df is not None:
            self.df = df
        else:
            self.df['open'] = None
            self.df['high'] = None
            self.df['low'] = None
            self.df['close'] = None
            self.df['adjclose'] = None
            self.df['volume'] = None
        self.name = None
        self.strategy = None
        self.strategy_settings = {}
        # Data and indicator dataframes
        self.indicator_df = None
        self.results = {}

        self.randomize_default = 0.1

        self.set_settings()

    def set_name(self, name):
        self.name = name

    def clear_settings(self):
        # Fixes performance issues by removing dataframes from memory for this object
        self.strategy_settings['open'] = None
        self.strategy_settings['high'] = None
        self.strategy_settings['low'] = None
        self.strategy_settings['close'] = None
        self.strategy_settings['adjclose'] = None
        self.strategy_settings['volume'] = None
        self.strategy = None
        self.df = {}

    def get_name(self):
        return self.name

    # def set_settings(self):
    #     # Must be implemented
    #     pass

    def get_settings(self, feature_columns=None):
        if feature_columns is None:
            feature_columns = {'adjclose', 'volume', 'open', 'high', 'low', 'close'}
        result = {k: self.strategy_settings[k] for k in self.strategy_settings.keys() - feature_columns}
        return result


    # @abstractmethod
    # def signal(self, price, args=None):
    #     """
    #     Determine if the signal is a buy signal, sell signal or neutral (if applicable). This can be implemented
    #     differently if you want to change how the signal decides when to buy and sell
    #     """







