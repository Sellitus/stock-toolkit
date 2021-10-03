import math
import random

from .TechnicalIndicator import TechnicalIndicator

from ta.trend import SMAIndicator, MACD, ADXIndicator, VortexIndicator, TRIXIndicator, MassIndex, CCIIndicator, DPOIndicator, KSTIndicator, IchimokuIndicator, PSARIndicator, STCIndicator


class TrendIndicators:
    class SignalMA(TechnicalIndicator):
        """
        Moving Average buy and sell signals
        """

        def __init__(self):
            super().__init__()
            super().set_name('ma')

        def __str__(self):
            return 'MA'

        def __repr__(self):
            return 'MA'

        def set_settings(self, window=20, fillna=False, randomize=False):
            if randomize is True:
                randomize = self.randomize_default
            # Cut randomize in half for determining range
            randomize = False if randomize is False else randomize * 0.5

            self.strategy_settings['window'] = window if randomize is False else random.randint(
                math.floor(window - (window * randomize)) - 1, math.ceil(window + (window * randomize)))
            self.strategy_settings['fillna'] = fillna

        def add_indicator(self, df):

            self.strategy = SMAIndicator(close=df['close'],
                                         window=self.strategy_settings['window'],
                                         fillna=self.strategy_settings['fillna'])

            df[f"trend_sma_fast"] = self.strategy.sma_indicator()
            self.clear_settings()
            return df

        def signal(self, row, buy_type='STANDARD', neutral_percentage=1):
            """
            Provides a signal for buy, sell or neutral (if supported)
            price: current price of asset
            buy_type: 'STANDARD' for a standard buy signal, 'NEUTRAL_RANGE' for a neutral range around the current price
            neutral_percentage: Percentage around upper and lower range to provide buy or sell signal
            """
            row = row[0]

            price = row['close']
            ma_value = row['trend_sma_fast']

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

        def __init__(self):
            super().__init__()
            super().set_name('macd')

        def __str__(self):
            return 'MACD'

        def __repr__(self):
            return 'MACD'

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

            self.strategy = MACD(close=df['close'],
                                 window_slow=self.strategy_settings['window_slow'],
                                 window_fast=self.strategy_settings['window_fast'],
                                 window_sign=self.strategy_settings['window_sign'],
                                 fillna=self.strategy_settings['fillna'])

            df[f"trend_macd"] = self.strategy.macd()
            self.clear_settings()
            return df

        def signal(self, row):
            """
            Provides a signal for buy, sell or neutral (if supported)
            """
            row = row[0]

            macd = row['trend_macd']

            if macd is None:
                raise ValueError("values dictionary does not contain key 'trend_macd', which is required for this "
                                 "function")

            signal = None
            if macd > 0:
                signal = 'BUY'
            else:
                signal = 'SELL'

            return signal


    class SignalADX(TechnicalIndicator):
        """
        ADX buy and sell signals
        """
        def __init__(self):
            super().__init__()
            super().set_name('adx')

        def __str__(self):
            return 'ADX'

        def __repr__(self):
            return 'ADX'

        def set_settings(self, window=14, fillna=False, randomize=False):
            if randomize is True:
                randomize = self.randomize_default
            # Cut randomize in half for determining range
            randomize = False if randomize is False else randomize * 0.5

            
            
            self.strategy_settings['window'] = window if randomize is False else random.randint(
                math.floor(window - (window * randomize)) - 1, math.ceil(window + (window * randomize)))
            self.strategy_settings['fillna'] = fillna

        def add_indicator(self, df):

            self.strategy = ADXIndicator(close=df['close'],
                                         high=df['high'],
                                         low=df['low'],
                                         window=self.strategy_settings['window'],
                                         fillna=self.strategy_settings['fillna'])

            df[f"trend_adx"] = self.strategy.adx()
            self.clear_settings()
            return df

        def signal(self, row):
            row = row[0]

            adx = row['trend_adx']

            if adx > 25:
                return 'ACTION'
            return 'NEUTRAL'


    class SignalVortex(TechnicalIndicator):
        """
        Vortex buy and sell signal
        """

        def __init__(self):
            super().__init__()
            super().set_name('vortex')

        def __str__(self):
            return 'Vortex'

        def __repr__(self):
            return 'Vortex'

        def set_settings(self, window=14, fillna=False, randomize=False):
            if randomize is True:
                randomize = self.randomize_default
            # Cut randomize in half for determining range
            randomize = False if randomize is False else randomize * 0.5

            
            
            
            self.strategy_settings['window'] = window if randomize is False else random.randint(
                math.floor(window - (window * randomize)) - 1, math.ceil(window + (window * randomize)))
            self.strategy_settings['fillna'] = fillna

        def add_indicator(self, df):

            self.strategy = VortexIndicator(close=df['close'],
                                            high=df['high'],
                                            low=df['low'],
                                            window=self.strategy_settings['window'],
                                            fillna=self.strategy_settings['fillna'])

            df[f"trend_vortex_ind_pos"] = self.strategy.vortex_indicator_pos()
            df[f"trend_vortex_ind_neg"] = self.strategy.vortex_indicator_neg()
            df[f"trend_vortex_ind_diff"] = self.strategy.vortex_indicator_neg()
            self.clear_settings()
            return df

        def signal(self, row):
            row = row[0]

            uptrend = row['trend_vortex_ind_pos']
            downtrend = row['trend_vortex_ind_neg']

            if uptrend > downtrend:
                return 'BUY'
            return 'SELL'


    class SignalTRIX(TechnicalIndicator):
        """
        TRIX buy and sell signal
        """

        def __init__(self):
            super().__init__()
            super().set_name('trix')

        def __str__(self):
            return 'TRIX'

        def __repr__(self):
            return 'TRIX'

        def set_settings(self, window=15, fillna=False, randomize=False):
            if randomize is True:
                randomize = self.randomize_default
            # Cut randomize in half for determining range
            randomize = False if randomize is False else randomize * 0.5

            
            self.strategy_settings['window'] = window if randomize is False else random.randint(
                math.floor(window - (window * randomize)) - 1, math.ceil(window + (window * randomize)))
            self.strategy_settings['fillna'] = fillna

        def add_indicator(self, df):

            self.strategy = TRIXIndicator(close=df['close'],
                                          window=self.strategy_settings['window'],
                                          fillna=self.strategy_settings['fillna'])

            df[f"trend_trix"] = self.strategy.trix()
            self.clear_settings()
            return df

        def signal(self, row):
            row = row[0]

            trix = row['trend_trix']

            if trix > 0:
                return 'BUY'
            return 'SELL'


    class SignalMassIndex(TechnicalIndicator):
        """
        Placeholder since the Mass Index does not provide a definitive buy or sell signal
        """

        def __init__(self):
            super().__init__()
            super().set_name('mass_index')

        def __str__(self):
            return 'Mass Index'

        def __repr__(self):
            return 'Mass Index'

        def set_settings(self, window_fast=9, window_slow=25, fillna=False, randomize=False):
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

            self.strategy = MassIndex(high=df['high'],
                                      low=df['low'],
                                      window_fast=self.strategy_settings['window_fast'],
                                      window_slow=self.strategy_settings['window_slow'],
                                      fillna=self.strategy_settings['fillna'])

            df[f"trend_mass_index"] = self.strategy.mass_index()
            self.clear_settings()
            return df

        def signal(self, row):
            row = row[0]
            return 'NEUTRAL'


    class SignalCCI(TechnicalIndicator):
        """
        Channel Commodity Index buy and sell signals
        TODO: Since a single cross is a buy or sell signal, design it to provide buy and sell signals in different ways or refer to recent history
        """

        def __init__(self):
            super().__init__()
            super().set_name('cci')

        def __str__(self):
            return 'CCI'

        def __repr__(self):
            return 'CCI'

        def set_settings(self, window=20, constant=0.015, fillna=False, randomize=False):
            if randomize is True:
                randomize = self.randomize_default
            # Cut randomize in half for determining range
            randomize = False if randomize is False else randomize * 0.5

            self.strategy_settings['window'] = window if randomize is False else random.randint(
                math.floor(window - (window * randomize)) - 1, math.ceil(window + (window * randomize)))
            self.strategy_settings['constant'] = constant if randomize is False else random.randint(
                math.floor(constant - (constant * randomize)) - 1, math.ceil(constant + (constant * randomize)))
            self.strategy_settings['fillna'] = fillna

        def add_indicator(self, df):

            self.strategy = CCIIndicator(close=df['close'],
                                         high=df['high'],
                                         low=df['low'],
                                         window=self.strategy_settings['window'],
                                         constant=self.strategy_settings['constant'],
                                         fillna=self.strategy_settings['fillna'])

            df[f"trend_cci"] = self.strategy.cci()
            self.clear_settings()
            return df

        def signal(self, row, bottom=-100, top=100, action_range=25):
            row = row[0]

            cci = row['trend_cci']

            # Calculate the neutral range around the bottom and top values
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

        def __init__(self):
            super().__init__()
            super().set_name('detrended_price_oscillator')

        def __str__(self):
            return 'DPO'

        def __repr__(self):
            return 'DPO'

        def set_settings(self, window=20, fillna=False, randomize=False):
            if randomize is True:
                randomize = self.randomize_default
            # Cut randomize in half for determining range
            randomize = False if randomize is False else randomize * 0.5

            self.strategy_settings['window'] = window if randomize is False else random.randint(
                math.floor(window - (window * randomize)) - 1, math.ceil(window + (window * randomize)))
            self.strategy_settings['fillna'] = fillna

        def add_indicator(self, df):

            self.strategy = DPOIndicator(close=df['close'], window=self.strategy_settings['window'],
                                         fillna=self.strategy_settings['fillna'])

            df[f"trend_dpo"] = self.strategy.dpo()
            self.clear_settings()
            return df

        def signal(self, row):
            row = row[0]

            dpo = row['trend_dpo']

            if dpo > 0:
                return 'BUY'
            return 'SELL'


    class SignalKnowSureThingOscillator(TechnicalIndicator):
        """
        Know Sure Thing Oscillator buy and sell signals
        """

        def __init__(self):
            super().__init__()
            super().set_name('know_sure_thing_oscillator')

        def __str__(self):
            return 'KST'

        def __repr__(self):
            return 'KST'

        def set_settings(self, roc1=10, roc2=15, roc3=20, roc4=30, window1=10, window2=10, window3=10, window4=15,
                          nsig=9, fillna=False, randomize=False):
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
            self.strategy_settings['window4'] = window4 if randomize is False else random.randint(
                math.floor(window4 - (window4 * randomize)) - 1, math.ceil(window4 + (window4 * randomize)))
            self.strategy_settings['roc1'] = roc1 if randomize is False else random.randint(
                math.floor(roc1 - (roc1 * randomize)) - 1, math.ceil(roc1 + (roc1 * randomize)))
            self.strategy_settings['roc2'] = roc2 if randomize is False else random.randint(
                math.floor(roc2 - (roc2 * randomize)) - 1, math.ceil(roc2 + (roc2 * randomize)))
            self.strategy_settings['roc3'] = roc3 if randomize is False else random.randint(
                math.floor(roc3 - (roc3 * randomize)) - 1, math.ceil(roc3 + (roc3 * randomize)))
            self.strategy_settings['roc4'] = roc4 if randomize is False else random.randint(
                math.floor(roc4 - (roc4 * randomize)) - 1, math.ceil(roc4 + (roc4 * randomize)))
            self.strategy_settings['nsig'] = nsig if randomize is False else random.randint(
                math.floor(nsig - (nsig * randomize)) - 1, math.ceil(nsig + (nsig * randomize)))
            self.strategy_settings['fillna'] = fillna

        def add_indicator(self, df):

            self.strategy = KSTIndicator(close=df['close'],
                                         roc1=self.strategy_settings['roc1'],
                                         roc2=self.strategy_settings['roc2'],
                                         roc3=self.strategy_settings['roc3'],
                                         roc4=self.strategy_settings['roc4'],
                                         window1=self.strategy_settings['window1'],
                                         window2=self.strategy_settings['window2'],
                                         window3=self.strategy_settings['window3'],
                                         window4=self.strategy_settings['window4'],
                                         nsig=self.strategy_settings['nsig'],
                                         fillna=self.strategy_settings['fillna'])

            df[f"trend_kst"] = self.strategy.kst()
            df[f"trend_kst_sig"] = self.strategy.kst_sig()
            df[f"trend_kst_diff"] = self.strategy.kst_diff()
            self.clear_settings()
            return df

        def signal(self, row):
            row = row[0]

            # NOTE: Might be incorrect signals being saved here
            fast = row['trend_kst_sig']
            slow = row['trend_kst']

            if fast > slow:
                return 'BUY'
            return 'SELL'


    class SignalIchimoku(TechnicalIndicator):
        """
        Ichimoku buy and sell signals
        """

        def __init__(self):
            super().__init__()
            super().set_name('ichimoku')

        def __str__(self):
            return 'Ichimoku'

        def __repr__(self):
            return 'Ichimoku'

        def set_settings(self, window1=9, window2=26, window3=52, visual=False, fillna=False, randomize=False):
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
            self.strategy_settings['visual'] = visual if randomize is False else random.randint(
                math.floor(visual - (visual * randomize)) - 1, math.ceil(visual + (visual * randomize)))
            self.strategy_settings['fillna'] = fillna

        def add_indicator(self, df):

            self.strategy = IchimokuIndicator(high=df['high'],
                                              low=df['low'],
                                              window1=self.strategy_settings['window1'],
                                              window2=self.strategy_settings['window2'],
                                              window3=self.strategy_settings['window3'],
                                              visual=self.strategy_settings['visual'],
                                              fillna=self.strategy_settings['fillna'])

            df[f"trend_ichimoku_a"] = self.strategy.ichimoku_a()
            df[f"trend_ichimoku_b"] = self.strategy.ichimoku_b()
            df[f"trend_ichimoku_base_line"] = self.strategy.ichimoku_base_line()
            df[f"trend_ichimoku_conversion_line"] = self.strategy.ichimoku_conversion_line()
            self.clear_settings()
            return df

        def signal(self, row):
            row = row[0]

            price = row['close']
            leading_span_a = row['trend_ichimoku_a']

            if price > leading_span_a:
                return 'BUY'
            return 'SELL'


    class SignalParabolicSAR(TechnicalIndicator):
        """
        Parabolic SAR buy and sell signals
        """

        def __init__(self):
            super().__init__()
            super().set_name('parabolic_sar')

        def __str__(self):
            return 'Parabolic SAR'

        def __repr__(self):
            return 'Parabolic SAR'

        def set_settings(self, step=0.02, max_step=0.2, fillna=False, randomize=False):
            if randomize is True:
                randomize = self.randomize_default
            # Cut randomize in half for determining range
            randomize = False if randomize is False else randomize * 0.5
            
            self.strategy_settings['step'] = step if randomize is False else random.randint(
                int(100 * (step - (step * randomize))), int(100 * (step + (step * randomize)))) / 100.0
            self.strategy_settings['max_step'] = max_step if randomize is False else random.randint(
                int(100 * (max_step - (max_step * randomize))), int(100 * (max_step + (max_step * randomize)))) / 100.0
            self.strategy_settings['fillna'] = fillna

        def add_indicator(self, df):

            self.strategy = PSARIndicator(close=df['close'],
                                          high=df['high'],
                                          low=df['low'],
                                          step=self.strategy_settings['step'],
                                          max_step=self.strategy_settings['max_step'],
                                          fillna=self.strategy_settings['fillna'])

            df[f"trend_psar"] = self.strategy.psar()
            df[f"trend_psar_up"] = self.strategy.psar_up()
            df[f"trend_psar_down"] = self.strategy.psar_down()
            df[f"trend_psar_up_indicator"] = self.strategy.psar_up_indicator()
            df[f"trend_psar_down_indicator"] = self.strategy.psar_down_indicator()
            self.clear_settings()
            return df

        def signal(self, row):
            row = row[0]

            price = row['close']
            parabolic_sar = row['trend_psar_up']

            if price > parabolic_sar:
                return 'BUY'
            return 'SELL'


    class SignalSchaffTrendCycle(TechnicalIndicator):
        """
        Schaff Trend Cycle buy and sell signals
        """

        def __init__(self):
            super().__init__()
            super().set_name('schaff_trend_cycle')

        def __str__(self):
            return 'STC'

        def __repr__(self):
            return 'STC'

        def set_settings(self, window_fast=23, window_slow=50, cycle=10, smooth1=3, smooth2=3, fillna=False,
                         randomize=False):
            if randomize is True:
                randomize = self.randomize_default
            # Cut randomize in half for determining range
            randomize = False if randomize is False else randomize * 0.5

            self.strategy_settings['cycle'] = cycle if randomize is False else random.randint(
                math.floor(cycle - (cycle * randomize)) - 1, math.ceil(cycle + (cycle * randomize)))
            self.strategy_settings['window_slow'] = window_slow if randomize is False else random.randint(
                math.floor(window_slow - (window_slow * randomize)) - 1,
                math.ceil(window_slow + (window_slow * randomize)))
            self.strategy_settings['window_fast'] = window_fast if randomize is False else random.randint(
                math.floor(window_fast - (window_fast * randomize)) - 1,
                math.ceil(window_fast + (window_fast * randomize)))
            self.strategy_settings['smooth1'] = smooth1 if randomize is False else random.randint(
                math.floor(smooth1 - (smooth1 * randomize)) - 1, math.ceil(smooth1 + (smooth1 * randomize)))
            self.strategy_settings['smooth2'] = smooth2 if randomize is False else random.randint(
                math.floor(smooth2 - (smooth2 * randomize)) - 1, math.ceil(smooth2 + (smooth2 * randomize)))
            self.strategy_settings['fillna'] = fillna

        def add_indicator(self, df):

            self.strategy = STCIndicator(close=df['close'],
                                         window_fast=self.strategy_settings['window_fast'],
                                         window_slow=self.strategy_settings['window_slow'],
                                         cycle=self.strategy_settings['cycle'],
                                         smooth1=self.strategy_settings['smooth1'],
                                         smooth2=self.strategy_settings['smooth2'],
                                         fillna=self.strategy_settings['fillna'])

            df[f"trend_stc"] = self.strategy.stc()
            self.clear_settings()
            return df

        def signal(self, row, bottom=25, top=75, action_range=10):
            row = row[0]

            stc = row['trend_stc']

            # Calculate the neutral range around the bottom and top values
            range_offset = 100 * (action_range * 0.005)
            top_low_range = top - range_offset
            top_high_range = top + range_offset
            bottom_low_range = bottom - range_offset
            bottom_high_range = bottom + range_offset

            # NOTE: Check these buy/sell rules
            if top_high_range >= stc >= top_low_range:
                signal = 'SELL'
            elif bottom_high_range >= stc >= bottom_low_range:
                signal = 'BUY'
            else:
                signal = 'NEUTRAL'

            return signal

    # REMOVED: SignalMassIndex
    trend_dna = [SignalMA(), SignalMACD(), SignalCCI(), SignalADX(), SignalVortex(), SignalTRIX(), SignalIchimoku(),
                 SignalParabolicSAR(), SignalSchaffTrendCycle(), SignalDetrendedPriceOscillator(),
                 SignalKnowSureThingOscillator()]
