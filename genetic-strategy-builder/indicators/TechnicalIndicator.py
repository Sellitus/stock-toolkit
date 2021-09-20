from abc import ABC, abstractmethod


class TechnicalIndicator(ABC):
    """
    Represents a single technical indicator. May be one of many different types and settings.
    """

    buy_sell = ['BUY', 'SELL', 'NEUTRAL']

    def __init__(self, name=None):
        self.name = name


    @abstractmethod
    def buy_sell_signal(self, price, args=None):
        """
        Determine if the signal is a buy signal, sell signal or neutral (if applicable). This can be implemented
        differently if you want to change how the signal decides when to buy and sell
        """







