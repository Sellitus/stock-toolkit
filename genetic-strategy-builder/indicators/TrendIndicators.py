

from TechnicalIndicator import TechnicalIndicator


class SignalMA(TechnicalIndicator):
    """
    Moving Average buy and sell signals
    """

    def signal(self, price, ma_value, buy_type='STANDARD', neutral_percentage=1):
        """
        Provides a signal for buy, sell or neutral (if supported)
        price: current price of asset
        buy_type: 'STANDARD' for a standard buy signal, 'NEUTRAL_RANGE' for a neutral range around the current price
        neutral_percentage: Percentage around upper and lower range to provide buy or sell signal
        """

        signal = None
        # Normal SMA buy / sell signal, where buy is when price is above the SMA dn sell is when price is below
        if buy_type == 'STANDARD':
            if price > ma_value:
                signal = 'BUY'
            elif price <= ma_value:
                signal = 'SELL'
        # Buy signal with a neutral range percentage around the price
        elif buy_type == 'NEUTRAL_RANGE':
            # Calculate the neutral range around the price
            price_offset = price * (neutral_percentage * 0.005)
            lower_range = price - price_offset
            upper_range = price + price_offset

            if upper_range >= price >= lower_range:
                signal = 'NEUTRAL'
            elif price > upper_range:
                signal = 'BUY'
            elif price < lower_range:
                signal = 'SELL'

        return signal


class SignalMACD(TechnicalIndicator):
    """
    MACD buy and sell signals
    """

    def signal(self, slow, fast):
        """
        Provides a signal for buy, sell or neutral (if supported)
        """

        if slow is None:
            raise ValueError("values dictionary does not contain key 'SLOW', which is required for this function")
        if fast is None:
            raise ValueError("values dictionary does not contain key 'FAST', which is required for this function")

        signal = None
        if fast > slow:
            signal = 'BUY'
        elif fast <= slow:
            signal = 'SELL'

        return signal


class SignalADX(TechnicalIndicator):
    """
    ADX buy and sell signals
    """

    def signal(self, adx):
        if adx > 25:
            return 'ACTION'
        return 'NEUTRAL'


class SignalVortex(TechnicalIndicator):
    """
    Vortex buy and sell signal
    """

    def signal(self, uptrend, downtrend):
        if uptrend > downtrend:
            return 'BUY'
        return 'SELL'


class SignalTRIX(TechnicalIndicator):
    """
    TRIX buy and sell signal
    """

    def signal(self, trix):
        if trix > 0:
            return 'BUY'
        return 'SELL'


class SignalMassIndex(TechnicalIndicator):
    """
    Placeholder since the Mass Index does not provide a definitive buy or sell signal
    """

    pass


class SignalCCI(TechnicalIndicator):
    """
    Channel Commodity Index buy and sell signals
    TODO: Since a single cross is a buy or sell signal, design it to provide buy and sell signals in different ways or refer to recent history
    """

    def signal(self, cci, bottom=-100, top=100, action_range=25):
        # Calculate the neutral range around the price
        range_offset = 100 * (action_range * 0.005)
        top_low_range = top - range_offset
        top_high_range = top + range_offset
        bottom_low_range = bottom - range_offset
        bottom_high_range = bottom + range_offset

        if top_high_range >= cci >= top_low_range:
            signal = 'SELL'
        elif bottom_high_range >= cci >= bottom_low_range:
            signal = 'BUY'
        else:
            signal = 'NEUTRAL'

        return signal


class SignalDetrendedPriceOscillator(TechnicalIndicator):
    """
    Detrended Price Oscillator buy and sell signals
    """

    def signal(self, dpo):
        if dpo > 0:
            return 'BUY'
        return 'SELL'


class SignalKnowSureThingOscillator(TechnicalIndicator):
    """
    Know Sure Thing Oscillator buy and sell signals
    """

    def signal(self, fast, slow):
        if fast > slow:
            return 'BUY'
        return 'SELL'


class SignalIchimoku(TechnicalIndicator):
    """
    Ichimoku buy and sell signals
    """

    def signal(self, price, leading_span):
        if price > leading_span:
            return 'BUY'
        return 'SELL'


class SignalParabolicSAR(TechnicalIndicator):
    """
    Parabolic SAR buy and sell signals
    """

    def signal(self, price, parabolic_sar):
        if price > parabolic_sar:
            return 'BUY'
        return 'SELL'


class SignalSchaffTrendCycle(TechnicalIndicator):
    """
    Schaff Trend Cycle buy and sell signals
    """

    def signal(self, cci, bottom=25, top=75, action_range=10):
        # Calculate the neutral range around the price
        range_offset = 100 * (action_range * 0.005)
        top_low_range = top - range_offset
        top_high_range = top + range_offset
        bottom_low_range = bottom - range_offset
        bottom_high_range = bottom + range_offset

        if top_high_range >= cci >= top_low_range:
            signal = 'SELL'
        elif bottom_high_range >= cci >= bottom_low_range:
            signal = 'BUY'
        else:
            signal = 'NEUTRAL'

        return signal




