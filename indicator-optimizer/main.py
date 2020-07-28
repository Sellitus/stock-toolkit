from datetime import datetime
import multiprocessing as mp
import backtrader as bt

import collections
import math
import operator


class SmaCross(bt.SignalStrategy):
    def __init__(self, low, high):

        sma1, sma2 = bt.ind.SMA(period=low), bt.ind.SMA(period=high)
        crossover = bt.ind.CrossOver(sma1, sma2)
        self.signal_add(bt.SIGNAL_LONG, crossover)
        self.sizer = MaxRiskSizer()

class MaxSizer(bt.Sizer):
    '''
    Returns the number of shares rounded down that can be purchased for the
    max risk tolerance
    '''
    params = (('risk', 0.96),)

    def _getsizing(self, comminfo, cash, data, isbuy):
        if isbuy == True:
            size = math.floor((cash * self.p.risk) / data[0])
        else:
            size = math.floor((cash * self.p.risk) / data[0]) * -1
        return size


class MaxRiskSizer(bt.Sizer):
    params = (('risk', 0.99), )

    def __init__(self):
        if self.p.risk > 1 or self.p.risk < 0:
            raise ValueError('The risk parameter is a percentage which must be entered as a float. e.g. 0.5')

    def _getsizing(self, comminfo, cash, data, isbuy):
        position = self.broker.getposition(data)
        if not position:
            size = math.floor((cash * self.p.risk) / data.close[0])
        else:
            size = position.size
        return size

# def setLowHighSMA(low, high):
#     global gl_sma_low
#     global gl_sma_high
#     gl_sma_low = low
#     gl_sma_high = high

def strategyThread(cerebro, initial_capital, low, high, best_results, best_result):
    cerebro.run()
    final_capital = cerebro.broker.getvalue()
    final_capital_str = '%.2f' % final_capital

    #print('Ending Portfolio Value: ({}, {}) {}'.format(low, high, final_capital_str))

    if final_capital > initial_capital:
        print('Found Profitable Settings: ({}, {}) {}'.format(low, high, final_capital_str))
        best_results.append((final_capital, (low, high)))
    if best_result is None or cerebro.broker.getvalue() > best_result.broker.getvalue():
        best_result = cerebro
        #print('UPDATED: Best Ending Portfolio Value: ({}, {}) {}'.format(low, high, final_capital))

def findFrequencySMA(best_results):
    low_freq = {}
    high_freq = {}
    for result in best_results:
        low = result[1][0]
        high = result[1][1]
        if low in low_freq:
            low_freq[low] += 1
        else:
            low_freq[low] = 1

        if high in high_freq:
            high_freq[high] += 1
        else:
            high_freq[high] = 1

    return low_freq, high_freq


if __name__ == '__main__':
    INITIAL_CAPITAL = 1000.0
    RANGE_LOW_MIN = 10
    RANGE_LOW_MAX = 50
    # RANGE_HIGH_MIN automatically set based on outer loop
    RANGE_HIGH_MAX = 50

    CPU_PROCESS_MULTIPLIER = 10

    data = bt.feeds.YahooFinanceData(dataname='AMD',
                                     fromdate=datetime(2017, 1, 1),
                                     todate=datetime(2020, 7, 26))

    #data = bt.feeds.YahooFinanceCSVData(dataname='/home/sellitus/Downloads/AMD.csv')

    print('Starting Portfolio Value: %.2f' % INITIAL_CAPITAL)

    manager = mp.Manager()
    best_results = manager.list()
    best_result = None
    process_pool = mp.Pool(mp.cpu_count() * CPU_PROCESS_MULTIPLIER)
    #process_pool = mp.Pool(1)

    for low in range(RANGE_LOW_MIN, RANGE_LOW_MAX):
        for high in range(low+1, RANGE_HIGH_MAX):
            cerebro = bt.Cerebro()
            cerebro.addstrategy(SmaCross, low=low, high=high)
            cerebro.broker.setcash(INITIAL_CAPITAL)
            cerebro.broker.setcommission(commission=0.001)
            cerebro.adddata(data)
            #strategyThread(cerebro, INITIAL_CAPITAL, low, high, best_results, best_result)
            result = process_pool.apply_async(strategyThread, (cerebro, INITIAL_CAPITAL, low, high, best_results, best_result,))

    process_pool.close()
    process_pool.join()

    best_results = sorted(best_results, key=lambda x: x[0])
    best_results.reverse()

    # Get the frequencies for each setting
    low_freq, high_freq = findFrequencySMA(best_results)

    low_freq = collections.Counter(low_freq).most_common(10)
    high_freq = collections.Counter(high_freq).most_common(10)

    # Sort by most frequent first
    # low_freq = dict(sorted(low_freq.items(), key=operator.itemgetter(1), reverse=True))
    # high_freq = dict(sorted(high_freq.items(), key=operator.itemgetter(1), reverse=True))

    # Find frequencies of profitable settings
    print('Low: ' + str(low_freq))
    print('High: ' + str(high_freq))

    # Plot best results
    cerebro = bt.Cerebro()
    cerebro.addstrategy(SmaCross, low=best_results[0][1][0], high=best_results[0][1][1])
    cerebro.broker.setcash(INITIAL_CAPITAL)
    cerebro.broker.setcommission(commission=0.001)
    cerebro.adddata(data)
    cerebro.run()
    cerebro.plot(style='candle')

