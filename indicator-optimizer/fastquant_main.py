from fastquant import get_stock_data, backtest
import pandas as pd


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.2f' % x)


def organizeColumns(df, strategy='smac'):
    if strategy == 'rsi':
        cols_to_oraganize = ['rsi_period', 'rsi_upper', 'rsi_lower',
                             'pnl', 'final_value', 'init_cash']
    elif strategy == 'smac':
        cols_to_oraganize = ['fast_period', 'slow_period',
                             'pnl', 'final_value', 'init_cash']
    elif strategy == 'emac':
        cols_to_oraganize = ['fast_period', 'slow_period',
                             'pnl', 'final_value', 'init_cash']
    elif strategy == 'macd':
        cols_to_oraganize = ['fast_period', 'slow_period', 'signal_period', 'sma_period', 'dir_period',
                             'pnl', 'final_value', 'init_cash']
    else:
        raise Exception('Incorrect strategy passed: {}'.format(strategy))

    cols_to_oraganize.reverse()

    for col_name in cols_to_oraganize:
        col = df[col_name]
        df.drop(labels=[col_name], axis=1, inplace=True)
        df.insert(0, col_name, col)

def optimizeRsi(df):
    result = backtest('rsi', df, verbose=False, plot=False,
                      rsi_period=range(7, 30), rsi_upper=range(60, 80), rsi_lower=range(20, 40))
    organizeColumns(result, 'rsi')
    return result

def optimizeSmaCross(df):
    result = backtest('smac', df, verbose=False, plot=False,
                      fast_period=range(10, 25), slow_period=range(25, 55))
                      # fast_period=range(15, 25), slow_period=range(45, 55))
    organizeColumns(result, 'smac')
    return result

def optimizeEmaCross(df):
    result = backtest('emac', df, verbose=False, plot=False,
                      fast_period=range(10, 31), slow_period=range(25, 71))
    organizeColumns(result, 'emac')
    return result

def optimizeMacd(df):
    result = backtest('macd', df, verbose=False, plot=False,
                      fast_period=range(5, 31), slow_period=range(20, 71), signal_period=range(5, 16),
                      sma_period=range(20, 41), dir_period=range(5, 20))
    organizeColumns(result, 'macd')
    return result


tickers = ['AMD', 'INTC', 'WORK', 'ZM', 'SNE', 'ATVI', 'EBAY', 'NFLX', 'TTWO', 'EA', 'ZNGA',
           'NVDA', 'MU', 'WHR', 'ASML', 'PFE', 'TSLA', 'DXCM', 'CRM', 'WIX', 'JD', 'NTDOY',
           'FVRR']

results = []

for ticker in tickers:
    df = get_stock_data(ticker, "2015-01-01", "2020-09-09")
    result = optimizeRsi(df)
    results.append([ticker, result.nlargest(10, 'pnl')])


for result in results:
    print('')
    print('Ticker: ' + result[0])
    print(result[1].iloc[:, : 5])

import pdb
pdb.set_trace()
