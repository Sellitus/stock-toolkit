import math
import random

from .TechnicalIndicator import TechnicalIndicator

from ta.momentum import AwesomeOscillatorIndicator, KAMAIndicator, PercentagePriceOscillator, PercentageVolumeOscillator, ROCIndicator, RSIIndicator, StochasticOscillator, StochRSIIndicator, TSIIndicator, UltimateOscillator, WilliamsRIndicator




class MomentumIndicators:
    class SignalAO(TechnicalIndicator):
        def __init__(self):
            super().__init__()
            super().set_name('Awesome Oscillator')

        def __str__(self):
            return 'AO'

        def __repr__(self):
            return 'AO'

        def set_settings(self, window1=5, window2=34, fillna=False, randomize=False):
            if randomize is True:
                randomize = self.randomize_default
            # Cut randomize in half for determining range
            randomize = False if randomize is False else randomize * 0.5

            self.strategy_settings['window1'] = window1 if randomize is False else random.randint(
                math.floor(window1 - (window1 * randomize)) - 1, math.ceil(window1 + (window1 * randomize)))
            self.strategy_settings['window2'] = window2 if randomize is False else random.randint(
                math.floor(window2 - (window2 * randomize)) - 1, math.ceil(window2 + (window2 * randomize)))
            self.strategy_settings['fillna'] = fillna

        def add_indicator(self, df):

            self.strategy = AwesomeOscillatorIndicator(high=df['high'],
                                                       low=df['low'],
                                                       window1=self.strategy_settings['window1'],
                                                       window2=self.strategy_settings['window2'],
                                                       fillna=self.strategy_settings['fillna'])

            df[f"momentum_ao"] = self.strategy.awesome_oscillator()
            self.clear_settings()
            return df

        def signal(self, row, buy_type='STANDARD'):
            """
            Provides a signal for buy, sell or neutral (if supported)
            row: current row for buy or sell signal
            buy_type: 'STANDARD' for a standard buy signal
            """

            ao_value = row['momentum_ao']

            signal = None
            if buy_type == 'STANDARD':
                if ao_value <= 0:
                    signal = 'SELL'
                else:
                    signal = 'BUY'

            return signal

    class SignalKAMA(TechnicalIndicator):
        def __init__(self):
            super().__init__()
            super().set_name('KAMA Indicator')

        def __str__(self):
            return 'KAMA'

        def __repr__(self):
            return 'KAMA'

        def set_settings(self, window=10, pow1=2, pow2=30, fillna=False, randomize=False):
            if randomize is True:
                randomize = self.randomize_default
            # Cut randomize in half for determining range
            randomize = False if randomize is False else randomize * 0.5
            
            self.strategy_settings['window'] = window if randomize is False else random.randint(
                math.floor(window - (window * randomize)) - 1, math.ceil(window + (window * randomize)))
            self.strategy_settings['pow1'] = pow1 if randomize is False else random.randint(
                math.floor(pow1 - (pow1 * randomize)) - 1, math.ceil(pow1 + (pow1 * randomize)))
            self.strategy_settings['pow2'] = pow2 if randomize is False else random.randint(
                math.floor(pow2 - (pow2 * randomize)) - 1, math.ceil(pow2 + (pow2 * randomize)))
            self.strategy_settings['fillna'] = fillna

        def add_indicator(self, df):

            self.strategy = KAMAIndicator(close=df['close'],
                                          window=self.strategy_settings['window'],
                                          pow1=self.strategy_settings['pow1'],
                                          pow2=self.strategy_settings['pow2'],
                                          fillna=self.strategy_settings['fillna'])

            df[f"momentum_kama"] = self.strategy.kama()
            self.clear_settings()
            return df

        def signal(self, row, buy_type='STANDARD'):
            """
            Provides a signal for buy, sell or neutral (if supported)
            row: current row for buy or sell signal
            buy_type: 'STANDARD' for a standard buy signal
            """

            price = row['close']
            kama_value = row['momentum_kama']

            signal = None
            if buy_type == 'STANDARD':
                if price <= kama_value:
                    signal = 'SELL'
                else:
                    signal = 'BUY'

            return signal


    class SignalPPO(TechnicalIndicator):
        def __init__(self):
            super().__init__()
            super().set_name('Percentage Price Oscillator')

        def __str__(self):
            return 'PPO'

        def __repr__(self):
            return 'PPO'

        def set_settings(self, window_slow=26, window_fast=12, window_sign=9, fillna=False, randomize=False):
            if randomize is True:
                randomize = self.randomize_default
            # Cut randomize in half for determining range
            randomize = False if randomize is False else randomize * 0.5

            self.strategy_settings['window_slow'] = window_slow if randomize is False else random.randint(
                math.floor(window_slow - (window_slow * randomize)) - 1,
                math.ceil(window_slow + (window_slow * randomize)))
            self.strategy_settings['window_fast'] = window_fast if randomize is False else random.randint(
                math.floor(window_fast - (window_fast * randomize)) - 1,
                math.ceil(window_fast + (window_fast * randomize)))
            self.strategy_settings['window_sign'] = window_sign if randomize is False else random.randint(
                math.floor(window_sign - (window_sign * randomize)) - 1,
                math.ceil(window_sign + (window_sign * randomize)))
            self.strategy_settings['fillna'] = fillna

        def add_indicator(self, df):

            self.strategy = PercentagePriceOscillator(close=df['close'],
                                                      window_slow=self.strategy_settings['window_slow'],
                                                      window_fast=self.strategy_settings['window_fast'],
                                                      window_sign=self.strategy_settings['window_sign'],
                                                      fillna=self.strategy_settings['fillna'])

            df[f"momentum_ppo"] = self.strategy.ppo()
            df[f"momentum_ppo_signal"] = self.strategy.ppo_signal()
            df[f"momentum_ppo_hist"] = self.strategy.ppo_hist()
            self.clear_settings()
            return df

        def signal(self, row, buy_type='STANDARD'):
            """
            Provides a signal for buy, sell or neutral (if supported)
            row: current row for buy or sell signal
            buy_type: 'STANDARD' for a standard buy signal
            """

            ppo = row['momentum_ppo']
            ppo_signal = row['momentum_ppo_signal']

            signal = None
            if buy_type == 'STANDARD':
                if ppo <= ppo_signal:
                    signal = 'SELL'
                else:
                    signal = 'BUY'

            return signal

    class SignalPVO(TechnicalIndicator):
        # NOTE: Not implemented, PVO values not present when adding all technical indicators for some reason
        def __init__(self):
            super().__init__()
            super().set_name('Percentage Volume Oscillator')

        def __str__(self):
            return 'PVO'

        def __repr__(self):
            return 'PVO'

        def set_settings(self, window_slow=26, window_fast=12, window_sign=9, fillna=False, randomize=False):
            if randomize is True:
                randomize = self.randomize_default
            # Cut randomize in half for determining range
            randomize = False if randomize is False else randomize * 0.5
            
            self.strategy_settings['window_slow'] = window_slow if randomize is False else random.randint(
                math.floor(window_slow - (window_slow * randomize)) - 1,
                math.ceil(window_slow + (window_slow * randomize)))
            self.strategy_settings['window_fast'] = window_fast if randomize is False else random.randint(
                math.floor(window_fast - (window_fast * randomize)) - 1,
                math.ceil(window_fast + (window_fast * randomize)))
            self.strategy_settings['window_sign'] = window_sign if randomize is False else random.randint(
                math.floor(window_sign - (window_sign * randomize)) - 1,
                math.ceil(window_sign + (window_sign * randomize)))
            self.strategy_settings['fillna'] = fillna

        def add_indicator(self, df):

            self.strategy = PercentageVolumeOscillator(volume=self.strategy_settings['volume'],
                                                       window_slow=self.strategy_settings['window_slow'],
                                                       window_fast=self.strategy_settings['window_fast'],
                                                       window_sign=self.strategy_settings['window_sign'],
                                                       fillna=self.strategy_settings['fillna'])

            df[f"momentum_pvo"] = self.strategy.pvo()
            df[f"momentum_pvo_signal"] = self.strategy.pvo_signal()
            df[f"momentum_pvo_hist"] = self.strategy.pvo_hist()
            self.clear_settings()
            return df

        def signal(self, row, buy_type='STANDARD'):
            pass


    class SignalROC(TechnicalIndicator):
        def __init__(self):
            super().__init__()
            super().set_name('ROC Indicator')

        def __str__(self):
            return 'ROC'

        def __repr__(self):
            return 'ROC'

        def set_settings(self, window=12, fillna=False, randomize=False):
            if randomize is True:
                randomize = self.randomize_default
            # Cut randomize in half for determining range
            randomize = False if randomize is False else randomize * 0.5

            self.strategy_settings['window'] = window if randomize is False else random.randint(
                math.floor(window - (window * randomize)) - 1, math.ceil(window + (window * randomize)))
            self.strategy_settings['fillna'] = fillna

        def add_indicator(self, df):

            self.strategy = ROCIndicator(close=df['close'], window=self.strategy_settings['window'],
                                         fillna=self.strategy_settings['fillna'])

            df[f"momentum_roc"] = self.strategy.roc()
            self.clear_settings()
            return df

        def signal(self, row, buy_type='STANDARD'):
            """
            Provides a signal for buy, sell or neutral (if supported)
            row: current row for buy or sell signal
            buy_type: 'STANDARD' for a standard buy signal
            """

            roc = row['momentum_roc']

            signal = None
            if buy_type == 'STANDARD':
                if roc > 0:
                    signal = 'BUY'
                else:
                    signal = 'SELL'

            return signal

    class SignalRSI(TechnicalIndicator):
        def __init__(self):
            super().__init__()
            super().set_name('RSI Indicator')

        def __str__(self):
            return 'RSI'

        def __repr__(self):
            return 'RSI'

        def set_settings(self, window=14, fillna=False, randomize=False):
            if randomize is True:
                randomize = self.randomize_default
            # Cut randomize in half for determining range
            randomize = False if randomize is False else randomize * 0.5
            
            self.strategy_settings['window'] = window if randomize is False else random.randint(
                math.floor(window - (window * randomize)) - 1, math.ceil(window + (window * randomize)))
            self.strategy_settings['fillna'] = fillna

        def add_indicator(self, df):

            self.strategy = RSIIndicator(close=df['close'], 
                                         window=self.strategy_settings['window'],
                                         fillna=self.strategy_settings['fillna'])

            df[f"momentum_rsi"] = self.strategy.rsi()
            self.clear_settings()
            return df

        def signal(self, row, buy_type='STANDARD'):
            """
            Provides a signal for buy, sell or neutral (if supported)
            row: current row for buy or sell signal
            buy_type: 'STANDARD' for a standard buy signal
            """

            rsi = row['momentum_rsi']

            signal = None
            if buy_type == 'STANDARD':
                if rsi > 70:
                    signal = 'BUY'
                elif rsi < 30:
                    signal = 'SELL'
                else:
                    signal = 'NEUTRAL'

            return signal

    class SignalStochastic(TechnicalIndicator):
        def __init__(self):
            super().__init__()
            super().set_name('Stochastic Oscillator')

        def __str__(self):
            return 'Stochastic Osc'

        def __repr__(self):
            return 'Stochastic Osc'

        def set_settings(self, window=14, smooth_window=3, fillna=False, randomize=False):
            if randomize is True:
                randomize = self.randomize_default
            # Cut randomize in half for determining range
            randomize = False if randomize is False else randomize * 0.5

            self.strategy_settings['window'] = window if randomize is False else random.randint(
                math.floor(window - (window * randomize)) - 1, math.ceil(window + (window * randomize)))
            self.strategy_settings['smooth_window'] = smooth_window if randomize is False else random.randint(
                math.floor(smooth_window - (smooth_window * randomize)) - 1,
                math.ceil(smooth_window + (smooth_window * randomize)))
            self.strategy_settings['fillna'] = fillna

        def add_indicator(self, df):

            self.strategy = StochasticOscillator(close=df['close'],
                                                 high=df['high'],
                                                 low=df['low'],
                                                 window=self.strategy_settings['window'],
                                                 smooth_window=self.strategy_settings['smooth_window'],
                                                 fillna=self.strategy_settings['fillna'])

            df[f"momentum_stoch"] = self.strategy.stoch()
            df[f"momentum_stoch_signal"] = self.strategy.stoch_signal()
            self.clear_settings()
            return df

        def signal(self, row, buy_type='STANDARD'):
            """
            Provides a signal for buy, sell or neutral (if supported)
            row: current row for buy or sell signal
            buy_type: 'STANDARD' for a standard buy signal
            """

            stochastic_slow = row['momentum_stoch_signal']
            stochastic_fast = row['momentum_stoch']

            signal = None
            if buy_type == 'STANDARD':
                if stochastic_fast > stochastic_slow:
                    signal = 'BUY'
                else:
                    signal = 'SELL'

            return signal

    class SignalStochasticRSI(TechnicalIndicator):
        def __init__(self):
            super().__init__()
            super().set_name('Stochastic RSI Indicator')


        def __str__(self):
            return 'Stochastic RSI'

        def __repr__(self):
            return 'Stochastic RSI'

        def set_settings(self, window=14, smooth1=3, smooth2=3, fillna=False, randomize=False):
            if randomize is True:
                randomize = self.randomize_default
            # Cut randomize in half for determining range
            randomize = False if randomize is False else randomize * 0.5
            
            self.strategy_settings['window'] = window if randomize is False else random.randint(
                math.floor(window - (window * randomize)) - 1, math.ceil(window + (window * randomize)))
            self.strategy_settings['smooth1'] = smooth1 if randomize is False else random.randint(
                math.floor(smooth1 - (smooth1 * randomize)) - 1, math.ceil(smooth1 + (smooth1 * randomize)))
            self.strategy_settings['smooth2'] = smooth2 if randomize is False else random.randint(
                math.floor(smooth2 - (smooth2 * randomize)) - 1, math.ceil(smooth2 + (smooth2 * randomize)))
            self.strategy_settings['fillna'] = fillna

        def add_indicator(self, df):

            self.strategy = StochRSIIndicator(close=df['close'],
                                              window=self.strategy_settings['window'],
                                              smooth1=self.strategy_settings['smooth1'],
                                              smooth2=self.strategy_settings['smooth2'],
                                              fillna=self.strategy_settings['fillna'])

            df[f"momentum_stoch_rsi"] = self.strategy.stochrsi()
            df[f"momentum_stoch_rsi_d"] = self.strategy.stochrsi_d()
            df[f"momentum_stoch_rsi_k"] = self.strategy.stochrsi_k()
            self.clear_settings()
            return df

        def signal(self, row, buy_type='STANDARD'):
            """
            Provides a signal for buy, sell or neutral (if supported)
            row: current row for buy or sell signal
            buy_type: 'STANDARD' for a standard buy signal
            """

            stochastic_rsi = row['momentum_stoch_rsi']

            signal = None
            if buy_type == 'STANDARD':
                if stochastic_rsi > 50:
                    signal = 'BUY'
                else:
                    signal = 'SELL'

            return signal

    class SignalTSI(TechnicalIndicator):
        def __init__(self):
            super().__init__()
            super().set_name('TSI Indicator')

        def __str__(self):
            return 'TSI'

        def __repr__(self):
            return 'TSI'

        def set_settings(self, window_slow=25, window_fast=13, fillna=False, randomize=False):
            if randomize is True:
                randomize = self.randomize_default
            # Cut randomize in half for determining range
            randomize = False if randomize is False else randomize * 0.5

            self.strategy_settings['window_slow'] = window_slow if randomize is False else random.randint(
                math.floor(window_slow - (window_slow * randomize)) - 1,
                math.ceil(window_slow + (window_slow * randomize)))
            self.strategy_settings['window_fast'] = window_fast if randomize is False else random.randint(
                math.floor(window_fast - (window_fast * randomize)) - 1,
                math.ceil(window_fast + (window_fast * randomize)))
            self.strategy_settings['fillna'] = fillna

        def add_indicator(self, df):

            self.strategy = TSIIndicator(close=df['close'],
                                         window_slow=self.strategy_settings['window_slow'],
                                         window_fast=self.strategy_settings['window_fast'],
                                         fillna=self.strategy_settings['fillna'])

            df[f"momentum_tsi"] = self.strategy.tsi()
            self.clear_settings()
            return df

        def signal(self, row, buy_type='STANDARD'):
            """
            Provides a signal for buy, sell or neutral (if supported)
            row: current row for buy or sell signal
            buy_type: 'STANDARD' for a standard buy signal
            """

            tsi = row['momentum_tsi']

            signal = None
            if buy_type == 'STANDARD':
                if tsi > 0:
                    signal = 'BUY'
                else:
                    signal = 'SELL'

            return signal

    class SignalUO(TechnicalIndicator):
        def __init__(self):
            super().__init__()
            super().set_name('Ultimate Oscillator')

            self.last_action = None

        def __str__(self):
            return 'UO'

        def __repr__(self):
            return 'UO'

        def set_settings(self, window1=7, window2=14, window3=28, weight1=4.0, weight2=2.0, weight3=1.0, fillna=False,
                         randomize=False):
            if randomize is True:
                randomize = self.randomize_default
            # Cut randomize in half for determining range
            randomize = False if randomize is False else randomize * 0.5

            self.strategy_settings['window1'] = window1 if randomize is False else random.randint(
                math.floor(window1 - (window1 * randomize)) - 1, math.ceil(window1 + (window1 * randomize)))
            self.strategy_settings['window2'] = window2 if randomize is False else random.randint(
                math.floor(window2 - (window2 * randomize)) - 1, math.ceil(window2 + (window2 * randomize)))
            self.strategy_settings['window3'] = window3 if randomize is False else random.randint(
                math.floor(window3 - (window3 * randomize)) - 1, math.ceil(window3 + (window3 * randomize)))
            self.strategy_settings['weight1'] = weight1 if randomize is False else random.randint(
                math.floor(weight1 - (weight1 * randomize)) - 1, math.ceil(weight1 + (weight1 * randomize)))
            self.strategy_settings['weight2'] = weight2 if randomize is False else random.randint(
                math.floor(weight2 - (weight2 * randomize)) - 1, math.ceil(weight2 + (weight2 * randomize)))
            self.strategy_settings['weight3'] = weight3 if randomize is False else random.randint(
                math.floor(weight3 - (weight3 * randomize)) - 1, math.ceil(weight3 + (weight3 * randomize)))
            self.strategy_settings['fillna'] = fillna

        def add_indicator(self, df):

            self.strategy = UltimateOscillator(close=df['close'],
                                               high=df['high'],
                                               low=df['low'],
                                               window1=self.strategy_settings['window1'],
                                               window2=self.strategy_settings['window2'],
                                               window3=self.strategy_settings['window3'],
                                               weight1=self.strategy_settings['weight1'],
                                               weight2=self.strategy_settings['weight2'],
                                               weight3=self.strategy_settings['weight3'],
                                               fillna=self.strategy_settings['fillna'])

            df[f"momentum_uo"] = self.strategy.ultimate_oscillator()
            self.clear_settings()
            return df

        def signal(self, row, buy_type='STANDARD'):
            """
            Provides a signal for buy, sell or neutral (if supported)
            row: current row for buy or sell signal
            buy_type: 'STANDARD' for a standard buy signal
            """

            uo = row['momentum_uo']

            signal = None
            if buy_type == 'STANDARD':
                if uo < 30:
                    signal = 'BUY'
                elif uo > 70:
                    signal = 'SELL'
                else:
                    signal = 'NEUTRAL'
                    # Carry over buy or sell from last action, since buy and sell signals are rare
                    signal = self.last_action if self.last_action is not None else 'SELL'

            self.last_action = signal
            return signal

    class SignalWR(TechnicalIndicator):
        def __init__(self):
            super().__init__()
            super().set_name('Williams Indicator')

            self.last_action = None

        def __str__(self):
            return 'WR'

        def __repr__(self):
            return 'WR'

        def set_settings(self, lbp=14, fillna=False, randomize=False):
            if randomize is True:
                randomize = self.randomize_default
            # Cut randomize in half for determining range
            randomize = False if randomize is False else randomize * 0.5

            self.strategy_settings['lbp'] = lbp if randomize is False else random.randint(
                math.floor(lbp - (lbp * randomize)) - 1, math.ceil(lbp + (lbp * randomize)))
            self.strategy_settings['fillna'] = fillna

        def add_indicator(self, df):

            self.strategy = WilliamsRIndicator(close=df['close'],
                                               high=df['high'],
                                               low=df['low'],
                                               lbp=self.strategy_settings['lbp'],
                                               fillna=self.strategy_settings['fillna'])

            df[f"momentum_wr"] = self.strategy.williams_r()
            self.clear_settings()
            return df

        def signal(self, row, buy_type='STANDARD'):
            """
            Provides a signal for buy, sell or neutral (if supported)
            row: current row for buy or sell signal
            buy_type: 'STANDARD' for a standard buy signal
            """

            uo = row['momentum_wr']

            signal = None
            if buy_type == 'STANDARD':
                if uo < 20:
                    signal = 'BUY'
                elif uo > 80:
                    signal = 'SELL'
                else:
                    signal = 'NEUTRAL'
                    # Carry over buy or sell from last action, since buy and sell signals are rare
                    signal = self.last_action if self.last_action is not None else 'SELL'

            self.last_action = signal
            return signal


    # REMOVED: SignalPVO()
    momentum_dna = [SignalAO(), SignalKAMA(), SignalPPO(), SignalROC(), SignalRSI(), SignalStochastic(),
                    SignalStochasticRSI(), SignalTSI(), SignalUO(), SignalWR()]
