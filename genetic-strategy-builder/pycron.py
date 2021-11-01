
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


def queue_system_run(system_id):
    # A is at the more optimal times, B is less optimal and so on
    crypto_morning_A = '10:30'
    crypto_morning_B = '09:15'

    stock_candle_close_A = '13:15'
    stock_candle_close_B = '12:00'

    # NOTE: Only add 3 of each symbol for each timeframe, otherwise it will become much slower
    if system_id == 1:
        # Crypto Morning A
        schedule.every().day.at(crypto_morning_A).do(run_on_ticker, 'ADA-USD')
        schedule.every().day.at(crypto_morning_A).do(run_on_ticker, 'ETH-USD')
        schedule.every().day.at(crypto_morning_A).do(run_on_ticker, 'COTI-USD')
        # Crypto Morning B
        schedule.every().day.at(crypto_morning_B).do(run_on_ticker, 'ATOM1-USD')
        schedule.every().day.at(crypto_morning_B).do(run_on_ticker, 'MATIC-USD')
        schedule.every().day.at(crypto_morning_B).do(run_on_ticker, 'ALGO-USD')
        # Stock Close A
        schedule.every().day.at(stock_candle_close_A).do(run_on_ticker, 'SPY')
        schedule.every().day.at(stock_candle_close_A).do(run_on_ticker, 'AMD')
        schedule.every().day.at(stock_candle_close_A).do(run_on_ticker, 'NVDA')
        # Stock Close B
        schedule.every().day.at(stock_candle_close_B).do(run_on_ticker, 'TSLA')
        schedule.every().day.at(stock_candle_close_B).do(run_on_ticker, 'LULU')
        schedule.every().day.at(stock_candle_close_B).do(run_on_ticker, 'FVRR')

    if system_id == 2:
        # Crypto Morning A
        schedule.every().day.at(crypto_morning_A).do(run_on_ticker, 'DOT1-USD')
        schedule.every().day.at(crypto_morning_A).do(run_on_ticker, 'LINK-USD')
        schedule.every().day.at(crypto_morning_A).do(run_on_ticker, 'DOGE-USD')
        # Crypto Morning B
        schedule.every().day.at(crypto_morning_B).do(run_on_ticker, 'SHIB-USD')
        schedule.every().day.at(crypto_morning_B).do(run_on_ticker, 'ICP1-USD')
        schedule.every().day.at(crypto_morning_B).do(run_on_ticker, 'HBAR-USD')
        # Stock Close A
        schedule.every().day.at(stock_candle_close_A).do(run_on_ticker, 'MSFT')
        schedule.every().day.at(stock_candle_close_A).do(run_on_ticker, 'GOOGL')
        schedule.every().day.at(stock_candle_close_A).do(run_on_ticker, 'INTC')
        # Stock Close B
        schedule.every().day.at(stock_candle_close_B).do(run_on_ticker, 'AAPL')
        schedule.every().day.at(stock_candle_close_B).do(run_on_ticker, 'ACN')
        schedule.every().day.at(stock_candle_close_B).do(run_on_ticker, 'UPST')


while True:
    queue_system_run(SYSTEM_ID)
    schedule.run_pending()
    time.sleep(60)
