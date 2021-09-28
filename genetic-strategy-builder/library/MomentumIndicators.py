

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

        def add_indicator(self, df, window1=5, window2=34, fillna=False):
            self.strategy_settings['high'] = df['high']
            self.strategy_settings['low'] = df['low']
            self.strategy_settings['window1'] = window1
            self.strategy_settings['window2'] = window2
            self.strategy_settings['fillna'] = fillna

            self.strategy = AwesomeOscillatorIndicator(high=self.strategy_settings['high'],
                                                       low=self.strategy_settings['low'],
                                                       window1=self.strategy_settings['window1'],
                                                       window2=self.strategy_settings['window2'],
                                                       fillna=self.strategy_settings['fillna'])
            return self.strategy

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

        def add_indicator(self, df, window=10, pow1=2, pow2=30, fillna=False):
            self.strategy_settings['close'] = df['close']
            self.strategy_settings['window'] = window
            self.strategy_settings['pow1'] = pow1
            self.strategy_settings['pow2'] = pow2
            self.strategy_settings['fillna'] = fillna

            self.strategy = KAMAIndicator(close=self.strategy_settings['close'],
                                          window=self.strategy_settings['window'],
                                          pow1=self.strategy_settings['pow1'],
                                          pow2=self.strategy_settings['pow2'],
                                          fillna=self.strategy_settings['fillna'])
            return self.strategy

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

        def add_indicator(self, df, window_slow=26, window_fast=12, window_sign=9, fillna=False):
            self.strategy_settings['close'] = df['close']
            self.strategy_settings['window_slow'] = window_slow
            self.strategy_settings['window_fast'] = window_fast
            self.strategy_settings['window_sign'] = window_sign
            self.strategy_settings['fillna'] = fillna

            self.strategy = PercentagePriceOscillator(close=self.strategy_settings['close'],
                                                      window_slow=window_slow,
                                                      window_fast=window_fast,
                                                      window_sign=window_sign,
                                                      fillna=self.strategy_settings['fillna'])
            return self.strategy

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

        def add_indicator(self, df, window_slow=26, window_fast=12, window_sign=9, fillna=False):
            self.strategy_settings['volume'] = df['volume']
            self.strategy_settings['window_slow'] = window_slow
            self.strategy_settings['window_fast'] = window_fast
            self.strategy_settings['window_sign'] = window_sign
            self.strategy_settings['fillna'] = fillna

            self.strategy = PercentageVolumeOscillator(volume=self.strategy_settings['volume'],
                                                       window_slow=self.strategy_settings['window_slow'],
                                                       window_fast=self.strategy_settings['window_fast'],
                                                       window_sign=self.strategy_settings['window_sign'],
                                                       fillna=self.strategy_settings['fillna'])
            return self.strategy

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

        def add_indicator(self, df, window=12, fillna=False):
            self.strategy_settings['close'] = df['close']
            self.strategy_settings['window'] = window
            self.strategy_settings['fillna'] = fillna

            self.strategy = ROCIndicator(close=self.strategy_settings['close'], window=self.strategy_settings['window'],
                                         fillna=self.strategy_settings['fillna'])
            return self.strategy

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

        def add_indicator(self, df, window=14, fillna=False):
            self.strategy_settings['close'] = df['close']
            self.strategy_settings['window'] = window
            self.strategy_settings['fillna'] = fillna

            self.strategy = RSIIndicator(close=self.strategy_settings['close'], window=self.strategy_settings['window'],
                                         fillna=self.strategy_settings['fillna'])
            return self.strategy

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

        def add_indicator(self, df, window=14, smooth_window=3, fillna=False):
            self.strategy_settings['close'] = df['close']
            self.strategy_settings['high'] = df['high']
            self.strategy_settings['low'] = df['low']
            self.strategy_settings['window'] = window
            self.strategy_settings['smooth_window'] = smooth_window
            self.strategy_settings['fillna'] = fillna

            self.strategy = StochasticOscillator(close=self.strategy_settings['close'],
                                                 high=self.strategy_settings['high'],
                                                 low=self.strategy_settings['low'],
                                                 window=self.strategy_settings['window'],
                                                 smooth_window=self.strategy_settings['smooth_window'], fillna=self.strategy_settings['fillna'])
            return self.strategy

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

        def add_indicator(self, df, window=14, smooth1=3, smooth2=3, fillna=False):
            self.strategy_settings['close'] = df['close']
            self.strategy_settings['window'] = window
            self.strategy_settings['smooth1'] = smooth1
            self.strategy_settings['smooth2'] = smooth2
            self.strategy_settings['fillna'] = fillna

            self.strategy = StochRSIIndicator(close=self.strategy_settings['close'],
                                              window=self.strategy_settings['window'],
                                              smooth1=self.strategy_settings['smooth1'],
                                              smooth2=self.strategy_settings['smooth2'],
                                              fillna=self.strategy_settings['fillna'])
            return self.strategy

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

        def add_indicator(self, df, window_slow=25, window_fast=13, fillna=False):
            self.strategy_settings['close'] = df['close']
            self.strategy_settings['window_slow'] = window_slow
            self.strategy_settings['window_fast'] = window_fast
            self.strategy_settings['fillna'] = fillna

            self.strategy = TSIIndicator(close=self.strategy_settings['close'],
                                         window_slow=self.strategy_settings['window_slow'],
                                         window_fast=self.strategy_settings['window_fast'], fillna=self.strategy_settings['fillna'])
            return self.strategy

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

        def add_indicator(self, df, window1=7, window2=14, window3=28, weight1=4.0, weight2=2.0, weight3=1.0,
                          fillna=False):
            self.strategy_settings['close'] = df['close']
            self.strategy_settings['high'] = df['high']
            self.strategy_settings['low'] = df['low']
            self.strategy_settings['window1'] = window1
            self.strategy_settings['window2'] = window2
            self.strategy_settings['window3'] = window3
            self.strategy_settings['weight1'] = weight1
            self.strategy_settings['weight2'] = weight2
            self.strategy_settings['weight3'] = weight3
            self.strategy_settings['fillna'] = fillna

            self.strategy = UltimateOscillator(close=self.strategy_settings['close'],
                                               high=self.strategy_settings['high'], low=self.strategy_settings['low'],
                                               window1=self.strategy_settings['window1'],
                                               window2=self.strategy_settings['window2'],
                                               window3=self.strategy_settings['window3'],
                                               weight1=self.strategy_settings['weight1'],
                                               weight2=self.strategy_settings['weight2'],
                                               weight3=self.strategy_settings['weight3'],
                                               fillna=self.strategy_settings['fillna'])
            return self.strategy

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

        def add_indicator(self, df, lbp=14, fillna=False):
            self.strategy_settings['close'] = df['close']
            self.strategy_settings['high'] = df['high']
            self.strategy_settings['low'] = df['low']
            self.strategy_settings['lbp'] = lbp
            self.strategy_settings['fillna'] = fillna

            self.strategy = WilliamsRIndicator(close=self.strategy_settings['close'],
                                               high=self.strategy_settings['high'], low=self.strategy_settings['close'],
                                               lbp=lbp, fillna=self.strategy_settings['fillna'])
            return self.strategy

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
