

from TechnicalIndicator import TechnicalIndicator


class SimpleMovingAverage(TechnicalIndicator):


    def buy_sell_signal(self, price, values=None):
        if values is None:
            values = {}

        sma = values['SMA'] if 'SMA' in values else None

        if sma is None:
            raise ValueError("values dictionary does not contain 'SMA' key which is required for this function")

        buy_type = values.get('BUY_TYPE')

        signal = None
        # Normal SMA buy / sell signal, where buy is when price is above the SMA dn sell is when price is below
        if buy_type is None:
            if price > sma:
                signal = 'BUY'
            elif price <= sma:
                signal = 'SELL'
        # Buy signal with a neutral range percentage around the price
        elif buy_type == 'NEUTRAL_RANGE':
            # Percent range around the price where a neutral signal will be thrown
            neutral_range = values.get('NEUTRAL_RANGE')
            if neutral_range is None:
                raise ValueError("values dictionary does not contain 'NEUTRAL_RANGE' key which is required for this "
                                 "BUY_TYPE")

            price_offset = price * (neutral_range * 0.005)
            lower_range = price - price_offset
            upper_range = price + price_offset

            if upper_range >= price >= lower_range:
                signal = 'NEUTRAL'
            elif price > upper_range:
                signal = 'BUY'
            elif price < lower_range:
                signal = 'SELL'


        return signal




