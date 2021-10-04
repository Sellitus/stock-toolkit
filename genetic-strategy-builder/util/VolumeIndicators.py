
import math
import random

from .TechnicalIndicator import TechnicalIndicator

from ta.volume import AccDistIndexIndicator, ChaikinMoneyFlowIndicator, EaseOfMovementIndicator, ForceIndexIndicator, \
    MFIIndicator, NegativeVolumeIndexIndicator, OnBalanceVolumeIndicator, VolumePriceTrendIndicator, \
    VolumeWeightedAveragePrice



class VolumeIndicators:
    class SignalAccDistIndex(TechnicalIndicator):
        def __init__(self):
            super().__init__()
            super().set_name('Accumulation Distribution Index')

        def __str__(self):
            return 'ADI'

        def __repr__(self):
            return 'ADI'

        def set_settings(self, look_back=5, fillna=False, randomize=False):
            if randomize is True:
                randomize = self.randomize_default
            # Cut randomize in half for determining range
            randomize = False if randomize is False else randomize * 0.5

            self.strategy_settings['look_back'] = look_back if randomize is False else random.randint(
                math.floor(look_back - (look_back * randomize)) - 1, math.ceil(look_back + (look_back * randomize)))

            self.strategy_settings['fillna'] = fillna

        def add_indicator(self, df):

            self.strategy = AccDistIndexIndicator(high=df['high'], low=df['low'], close=df['close'],
                                                  volume=df['volume'], fillna=self.strategy_settings['fillna'])

            df[f"volume_adi"] = self.strategy.acc_dist_index()
            self.clear_settings()
            return df

        def signal(self, row, buy_type='STANDARD'):
            """
            Provides a signal for buy, sell or neutral (if supported)
            row: current row for buy or sell signal
            buy_type: 'STANDARD' for a standard buy signal
            """
            curr_row = row[0]
            prev_rows = row[1]

            curr_adi_value = curr_row['volume_adi']
            look_back_idx = int(self.strategy_settings['look_back'])
            look_back_idx = look_back_idx if look_back_idx < len(prev_rows) else len(prev_rows) - 1
            prev_adi_value = prev_rows[look_back_idx]['volume_adi']

            signal = None
            if buy_type == 'STANDARD':
                if curr_adi_value <= prev_adi_value:
                    signal = 'SELL'
                else:
                    signal = 'BUY'

            return signal

    class SignalChaikinMoneyFlow(TechnicalIndicator):
        def __init__(self):
            super().__init__()
            super().set_name('Chaikin Money Flow')

        def __str__(self):
            return 'CMF'

        def __repr__(self):
            return 'CMF'

        def set_settings(self, window=20, fillna=False, randomize=False):
            if randomize is True:
                randomize = self.randomize_default
            # Cut randomize in half for determining range
            randomize = False if randomize is False else randomize * 0.5

            self.strategy_settings['window'] = window if randomize is False else random.randint(
                math.floor(window - (window * randomize)) - 1, math.ceil(window + (window * randomize)))

            self.strategy_settings['fillna'] = fillna

        def add_indicator(self, df):

            self.strategy = ChaikinMoneyFlowIndicator(high=df['high'], low=df['low'], close=df['close'],
                                                      volume=df['volume'], window=self.strategy_settings['window'],
                                                      fillna=self.strategy_settings['fillna'])

            df[f"volume_cmf"] = self.strategy.chaikin_money_flow()
            self.clear_settings()
            return df

        def signal(self, row, buy_type='STANDARD'):
            """
            Provides a signal for buy, sell or neutral (if supported)
            row: current row for buy or sell signal
            buy_type: 'STANDARD' for a standard buy signal
            """
            curr_row = row[0]

            cmf_value = curr_row['volume_cmf']

            signal = None
            if buy_type == 'STANDARD':
                if cmf_value <= 0:
                    signal = 'SELL'
                else:
                    signal = 'BUY'

            return signal

    class SignalEaseOfMovement(TechnicalIndicator):
        def __init__(self):
            super().__init__()
            super().set_name('Ease of Movement')

        def __str__(self):
            return 'EM'

        def __repr__(self):
            return 'EM'

        def set_settings(self, window=14, fillna=False, randomize=False):
            if randomize is True:
                randomize = self.randomize_default
            # Cut randomize in half for determining range
            randomize = False if randomize is False else randomize * 0.5

            self.strategy_settings['window'] = window if randomize is False else random.randint(
                math.floor(window - (window * randomize)) - 1, math.ceil(window + (window * randomize)))

            self.strategy_settings['fillna'] = fillna

        def add_indicator(self, df):

            self.strategy = EaseOfMovementIndicator(high=df['high'], low=df['low'], volume=df['volume'],
                                                    window=self.strategy_settings['window'],
                                                    fillna=self.strategy_settings['fillna'])

            df[f"volume_em"] = self.strategy.ease_of_movement()
            self.clear_settings()
            return df

        def signal(self, row, buy_type='STANDARD'):
            """
            Provides a signal for buy, sell or neutral (if supported)
            row: current row for buy or sell signal
            buy_type: 'STANDARD' for a standard buy signal
            """
            curr_row = row[0]

            em_value = curr_row['volume_em']

            signal = None
            if buy_type == 'STANDARD':
                if em_value <= 0:
                    signal = 'SELL'
                else:
                    signal = 'BUY'

            return signal

    class SignalForceIndex(TechnicalIndicator):
        def __init__(self):
            super().__init__()
            super().set_name('Force Index')

        def __str__(self):
            return 'FI'

        def __repr__(self):
            return 'FI'

        def set_settings(self, window=13, fillna=False, randomize=False):
            if randomize is True:
                randomize = self.randomize_default
            # Cut randomize in half for determining range
            randomize = False if randomize is False else randomize * 0.5

            self.strategy_settings['window'] = window if randomize is False else random.randint(
                math.floor(window - (window * randomize)) - 1, math.ceil(window + (window * randomize)))

            self.strategy_settings['fillna'] = fillna

        def add_indicator(self, df):

            self.strategy = ForceIndexIndicator(close=df['close'], volume=df['volume'],
                                                window=self.strategy_settings['window'],
                                                fillna=self.strategy_settings['fillna'])

            df[f"volume_fi"] = self.strategy.force_index()
            self.clear_settings()
            return df

        def signal(self, row, buy_type='STANDARD'):
            """
            Provides a signal for buy, sell or neutral (if supported)
            row: current row for buy or sell signal
            buy_type: 'STANDARD' for a standard buy signal
            """
            curr_row = row[0]

            fi_value = curr_row['volume_fi']

            signal = None
            if buy_type == 'STANDARD':
                if fi_value <= 0:
                    signal = 'SELL'
                else:
                    signal = 'BUY'

            return signal

    class SignalMFI(TechnicalIndicator):
        def __init__(self):
            super().__init__()
            super().set_name('Money Flow Index')

        def __str__(self):
            return 'MFI'

        def __repr__(self):
            return 'MFI'

        def set_settings(self, window=13, fillna=False, randomize=False):
            if randomize is True:
                randomize = self.randomize_default
            # Cut randomize in half for determining range
            randomize = False if randomize is False else randomize * 0.5

            self.strategy_settings['window'] = window if randomize is False else random.randint(
                math.floor(window - (window * randomize)) - 1, math.ceil(window + (window * randomize)))

            self.strategy_settings['fillna'] = fillna

        def add_indicator(self, df):

            self.strategy = MFIIndicator(close=df['close'], volume=df['volume'], high=df['high'], low=df['low'],
                                         window=self.strategy_settings['window'],
                                         fillna=self.strategy_settings['fillna'])

            df[f"volume_mfi"] = self.strategy.money_flow_index()
            self.clear_settings()
            return df

        def signal(self, row, buy_type='STANDARD'):
            """
            Provides a signal for buy, sell or neutral (if supported)
            row: current row for buy or sell signal
            buy_type: 'STANDARD' for a standard buy signal
            """
            curr_row = row[0]

            mfi_value = curr_row['volume_mfi']

            signal = None
            if buy_type == 'STANDARD':
                if mfi_value >= 20:
                    signal = 'SELL'
                elif mfi_value <= 80:
                    signal = 'BUY'
                else:
                    signal = 'NEUTRAL'

            return signal

    class SignalNVI(TechnicalIndicator):
        def __init__(self):
            super().__init__()
            super().set_name('Negative Volume Index')

        def __str__(self):
            return 'NVI'

        def __repr__(self):
            return 'NVI'

        def set_settings(self, look_back=5, fillna=False, randomize=False):
            if randomize is True:
                randomize = self.randomize_default
            # Cut randomize in half for determining range
            randomize = False if randomize is False else randomize * 0.5

            self.strategy_settings['look_back'] = look_back if randomize is False else random.randint(
                math.floor(look_back - (look_back * randomize)) - 1, math.ceil(look_back + (look_back * randomize)))

            self.strategy_settings['fillna'] = fillna

        def add_indicator(self, df):

            self.strategy = NegativeVolumeIndexIndicator(close=df['close'], volume=df['volume'],
                                                         fillna=self.strategy_settings['fillna'])

            df[f"volume_nvi"] = self.strategy.negative_volume_index()
            self.clear_settings()
            return df

        def signal(self, row, buy_type='STANDARD'):
            """
            Provides a signal for buy, sell or neutral (if supported)
            row: current row for buy or sell signal
            buy_type: 'STANDARD' for a standard buy signal
            """
            curr_row = row[0]
            prev_rows = row[1]

            nvi_value = curr_row['volume_nvi']
            look_back_idx = int(self.strategy_settings['look_back'])
            look_back_idx = look_back_idx if look_back_idx < len(prev_rows) else len(prev_rows) - 1
            prev_nvi_value = prev_rows[look_back_idx]['volume_nvi']

            signal = None
            if buy_type == 'STANDARD':
                if nvi_value <= prev_nvi_value:
                    signal = 'SELL'
                else:
                    signal = 'BUY'

            return signal

    class SignalOBV(TechnicalIndicator):
        def __init__(self):
            super().__init__()
            super().set_name('On Balance Volume')

        def __str__(self):
            return 'OBV'

        def __repr__(self):
            return 'OBV'

        def set_settings(self, look_back=5, fillna=False, randomize=False):
            if randomize is True:
                randomize = self.randomize_default
            # Cut randomize in half for determining range
            randomize = False if randomize is False else randomize * 0.5

            self.strategy_settings['look_back'] = look_back if randomize is False else random.randint(
                math.floor(look_back - (look_back * randomize)) - 1, math.ceil(look_back + (look_back * randomize)))

            self.strategy_settings['fillna'] = fillna

        def add_indicator(self, df):

            self.strategy = OnBalanceVolumeIndicator(close=df['close'], volume=df['volume'],
                                                     fillna=self.strategy_settings['fillna'])

            df[f"volume_obv"] = self.strategy.on_balance_volume()
            self.clear_settings()
            return df

        def signal(self, row, buy_type='STANDARD'):
            """
            Provides a signal for buy, sell or neutral (if supported)
            row: current row for buy or sell signal
            buy_type: 'STANDARD' for a standard buy signal
            """
            curr_row = row[0]
            prev_rows = row[1]

            obv_value = curr_row['volume_obv']
            look_back_idx = int(self.strategy_settings['look_back'])
            look_back_idx = look_back_idx if look_back_idx < len(prev_rows) else len(prev_rows) - 1
            prev_obv_value = prev_rows[look_back_idx]['volume_obv']

            signal = None
            if buy_type == 'STANDARD':
                if obv_value <= prev_obv_value:
                    signal = 'SELL'
                else:
                    signal = 'BUY'

            return signal

    class SignalVPT(TechnicalIndicator):
        def __init__(self):
            super().__init__()
            super().set_name('Volume Price Trend')

        def __str__(self):
            return 'VPT'

        def __repr__(self):
            return 'VPT'

        def set_settings(self, look_back=5, fillna=False, randomize=False):
            if randomize is True:
                randomize = self.randomize_default
            # Cut randomize in half for determining range
            randomize = False if randomize is False else randomize * 0.5

            self.strategy_settings['look_back'] = look_back if randomize is False else random.randint(
                math.floor(look_back - (look_back * randomize)) - 1, math.ceil(look_back + (look_back * randomize)))

            self.strategy_settings['fillna'] = fillna

        def add_indicator(self, df):

            self.strategy = VolumePriceTrendIndicator(close=df['close'], volume=df['volume'],
                                                      fillna=self.strategy_settings['fillna'])

            df[f"volume_vpt"] = self.strategy.volume_price_trend()
            self.clear_settings()
            return df

        def signal(self, row, buy_type='STANDARD'):
            """
            Provides a signal for buy, sell or neutral (if supported)
            row: current row for buy or sell signal
            buy_type: 'STANDARD' for a standard buy signal
            """
            curr_row = row[0]
            prev_rows = row[1]

            vpt_value = curr_row['volume_vpt']
            look_back_idx = int(self.strategy_settings['look_back'])
            look_back_idx = look_back_idx if look_back_idx < len(prev_rows) else len(prev_rows) - 1
            prev_vpt_value = prev_rows[look_back_idx]['volume_vpt']

            signal = None
            if buy_type == 'STANDARD':
                if vpt_value <= prev_vpt_value:
                    signal = 'SELL'
                else:
                    signal = 'BUY'

            return signal

    class SignalVWAP(TechnicalIndicator):
        def __init__(self):
            super().__init__()
            super().set_name('Volume Weighted Average Price')

        def __str__(self):
            return 'VWAP'

        def __repr__(self):
            return 'VWAP'

        def set_settings(self, window=5, fillna=False, randomize=False):
            if randomize is True:
                randomize = self.randomize_default
            # Cut randomize in half for determining range
            randomize = False if randomize is False else randomize * 0.5

            self.strategy_settings['window'] = window if randomize is False else random.randint(
                math.floor(window - (window * randomize)) - 1, math.ceil(window + (window * randomize)))

            self.strategy_settings['fillna'] = fillna

        def add_indicator(self, df):

            self.strategy = VolumeWeightedAveragePrice(close=df['close'], volume=df['volume'], high=df['high'],
                                                       low=df['low'], window=self.strategy_settings['window'],
                                                       fillna=self.strategy_settings['fillna'])

            df[f"volume_vwap"] = self.strategy.volume_weighted_average_price()
            self.clear_settings()
            return df

        def signal(self, row, buy_type='STANDARD'):
            """
            Provides a signal for buy, sell or neutral (if supported)
            row: current row for buy or sell signal
            buy_type: 'STANDARD' for a standard buy signal
            """
            curr_row = row[0]

            vwap_value = curr_row['volume_vwap']
            price = curr_row['close']

            signal = None
            if buy_type == 'STANDARD':
                if price >= vwap_value:
                    signal = 'SELL'
                else:
                    signal = 'BUY'

            return signal

    volume_dna = [SignalAccDistIndex(), SignalChaikinMoneyFlow(), SignalEaseOfMovement(), SignalForceIndex(),
                  SignalMFI(), SignalNVI(), SignalOBV(), SignalVPT(), SignalVWAP()]
