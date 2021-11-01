
import argparse
import schedule
import time

import subprocess


parser = argparse.ArgumentParser(description='Python based cron alternative...for being able to send emails.')
parser.add_argument('--id', dest="SYSTEM_ID", required=True, type=int,
                    help="ID of the machine for running tests. Chooses the suite of tests to run so there is no "
                         "overlap. This arg is used in the /etc/systemd/system/pycron.service file"
                         " - Example [ --id 2 ]")
args = parser.parse_args()

SYSTEM_ID = args.SYSTEM_ID


PYTHON_EXE = '/home/sellitus/anaconda3/envs/stock-toolkit/bin/python'
GEN_ALGO_EXE = '/home/sellitus/PythonProjects/stock-toolkit/genetic-strategy-builder/gen_algo.py'

NOTIFY_VAL = 1200


def run_on_ticker(ticker):
    cmd_str = '{} {} --notify {} -production --tickers {} & '.format(PYTHON_EXE, GEN_ALGO_EXE, NOTIFY_VAL, ticker)
    subprocess.run(cmd_str, shell=True)


def start_system_run(system_id):
    crypto_morning_A = '09:30'
    crypto_morning_B = '10:30'

    stock_candle_close_A = '12:30'
    stock_candle_close_B = '13:30'

    # NOTE: Only add 3 of each symbol for each timeframe, otherwise it will become much slower
    if system_id == 1:
        # Morning A
        schedule.every().day.at(crypto_morning_A).do(run_on_ticker, 'ADA-USD')
        schedule.every().day.at(crypto_morning_A).do(run_on_ticker, 'ETH-USD')
        schedule.every().day.at(crypto_morning_A).do(run_on_ticker, 'COTI-USD')
        # Morning B
        schedule.every().day.at(crypto_morning_B).do(run_on_ticker, 'ATOM1-USD')
        # Stock Candle Close A
        schedule.every().day.at(stock_candle_close_A).do(run_on_ticker, 'SPY')
        schedule.every().day.at(stock_candle_close_A).do(run_on_ticker, 'AMD')
        schedule.every().day.at(stock_candle_close_A).do(run_on_ticker, 'NVDA')
        # Stock Candle Close B
        schedule.every().day.at(stock_candle_close_B).do(run_on_ticker, 'TSLA')

    if system_id == 2:
        # Morning A
        schedule.every().day.at(crypto_morning_A).do(run_on_ticker, 'DOT1-USD')
        schedule.every().day.at(crypto_morning_A).do(run_on_ticker, 'LINK-USD')
        schedule.every().day.at(crypto_morning_A).do(run_on_ticker, 'DOGE-USD')
        # Morning B
        schedule.every().day.at(crypto_morning_B).do(run_on_ticker, 'SHIB-USD')
        # Stock Candle Close A
        schedule.every().day.at(stock_candle_close_A).do(run_on_ticker, 'MSFT')
        schedule.every().day.at(stock_candle_close_A).do(run_on_ticker, 'GOOGL')
        schedule.every().day.at(stock_candle_close_A).do(run_on_ticker, 'INTC')
        # Stock Candle Close B
        schedule.every().day.at(stock_candle_close_B).do(run_on_ticker, 'AAPL')


while True:
    start_system_run(SYSTEM_ID)
    schedule.run_pending()
    time.sleep(60)
