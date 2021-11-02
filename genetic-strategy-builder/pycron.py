
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

# Argument settings
SYSTEM_ID = args.SYSTEM_ID

# Static path settings
PYTHON_EXE = '/home/sellitus/anaconda3/envs/stock-toolkit/bin/python'
GEN_ALGO_EXE = '/home/sellitus/PythonProjects/stock-toolkit/genetic-strategy-builder/gen_algo.py'

# User settings
NOTIFY_VAL = 1200
production_flag = '-production'
schedule_override = '09:50'
# if schedule_override == '' or schedule_override is None else schedule_override


def run_on_ticker(ticker):
    cmd_str = '{} {} --notify {} {} --tickers {}'.format(PYTHON_EXE, GEN_ALGO_EXE, NOTIFY_VAL, production_flag,
                                                         ticker)
    subprocess.Popen(cmd_str.split(' '), stdin=None, stdout=None, stderr=None)


def queue_system_run(system_id):
    # A is at the more optimal time, B is less optimal and so on
    crypto_morning_A = '09:30' if schedule_override == '' or schedule_override is None else schedule_override
    crypto_morning_B = '08:30'

    stock_candle_close_A = '13:30'
    stock_candle_close_B = '12:30'

    # NOTE: Only add 3 of each symbol for each timeframe, otherwise it will become much slower
    if system_id == 1:
        # Crypto Morning A
        schedule.every().day.at(crypto_morning_A).do(run_on_ticker, 'ADA-USD')
        schedule.every().day.at(crypto_morning_A).do(run_on_ticker, 'ETH-USD')
        schedule.every().day.at(crypto_morning_A).do(run_on_ticker, 'COTI-USD')
        schedule.every().day.at(crypto_morning_A).do(run_on_ticker, 'DOT1-USD')
        # Crypto Morning B
        schedule.every().day.at(crypto_morning_B).do(run_on_ticker, 'BTC-USD')
        schedule.every().day.at(crypto_morning_B).do(run_on_ticker, 'ICP1-USD')
        schedule.every().day.at(crypto_morning_B).do(run_on_ticker, 'MANA-USD')
        schedule.every().day.at(crypto_morning_B).do(run_on_ticker, 'AAVE-USD')
        # Stock Close A
        schedule.every().day.at(stock_candle_close_A).do(run_on_ticker, 'SPY')
        schedule.every().day.at(stock_candle_close_A).do(run_on_ticker, 'AMD')
        schedule.every().day.at(stock_candle_close_A).do(run_on_ticker, 'NVDA')
        schedule.every().day.at(stock_candle_close_A).do(run_on_ticker, 'SNOW')
        # Stock Close B
        schedule.every().day.at(stock_candle_close_B).do(run_on_ticker, 'TSLA')
        schedule.every().day.at(stock_candle_close_B).do(run_on_ticker, 'LULU')
        schedule.every().day.at(stock_candle_close_B).do(run_on_ticker, 'FVRR')
        schedule.every().day.at(stock_candle_close_B).do(run_on_ticker, 'SHOP')

    if system_id == 2:
        # Crypto Morning A
        schedule.every().day.at(crypto_morning_A).do(run_on_ticker, 'LINK-USD')
        schedule.every().day.at(crypto_morning_A).do(run_on_ticker, 'ATOM1-USD')
        schedule.every().day.at(crypto_morning_A).do(run_on_ticker, 'MATIC-USD')
        schedule.every().day.at(crypto_morning_A).do(run_on_ticker, 'ALGO-USD')
        # Crypto Morning B
        schedule.every().day.at(crypto_morning_B).do(run_on_ticker, 'HBAR-USD')
        schedule.every().day.at(crypto_morning_B).do(run_on_ticker, 'SHIB-USD')
        schedule.every().day.at(crypto_morning_B).do(run_on_ticker, 'DOGE-USD')
        schedule.every().day.at(crypto_morning_B).do(run_on_ticker, 'SOL1-USD')
        # Stock Close A
        schedule.every().day.at(stock_candle_close_A).do(run_on_ticker, 'MSFT')
        schedule.every().day.at(stock_candle_close_A).do(run_on_ticker, 'GOOGL')
        schedule.every().day.at(stock_candle_close_A).do(run_on_ticker, 'INTC')
        schedule.every().day.at(stock_candle_close_A).do(run_on_ticker, 'BAND')
        # Stock Close B
        schedule.every().day.at(stock_candle_close_B).do(run_on_ticker, 'AAPL')
        schedule.every().day.at(stock_candle_close_B).do(run_on_ticker, 'ACN')
        schedule.every().day.at(stock_candle_close_B).do(run_on_ticker, 'UPST')
        schedule.every().day.at(stock_candle_close_B).do(run_on_ticker, 'EA')


while True:
    queue_system_run(SYSTEM_ID)
    schedule.run_pending()
    time.sleep(60)
