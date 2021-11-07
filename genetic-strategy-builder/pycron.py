
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
EMERGENCY_MODE = False
NOTIFY_VAL = 1200
production_flag = '-production'
# Override any of the following to reschedule, save, then restart the pycron service
override_crypto_morning_A = ''
override_crypto_morning_B = ''
override_stock_candle_close_A = ''
override_stock_candle_close_B = ''
# Set overrides if present
crypto_morning_A = '09:30' if override_crypto_morning_A == '' or override_crypto_morning_A is None \
    else override_crypto_morning_A
crypto_morning_B = '08:30' if override_crypto_morning_B == '' or override_crypto_morning_B is None \
    else override_crypto_morning_B
stock_close_A = '13:30' if override_stock_candle_close_A == '' or override_stock_candle_close_A is None \
    else override_stock_candle_close_A
stock_close_B = '12:30' if override_stock_candle_close_B == '' or override_stock_candle_close_B is None \
    else override_stock_candle_close_B


def run_on_ticker(ticker):
    cmd_str = '{} {} --notify {} {} --tickers {}'.format(PYTHON_EXE, GEN_ALGO_EXE, NOTIFY_VAL, production_flag,
                                                         ticker)
    subprocess.Popen(cmd_str.split(' '), stdin=None, stdout=None, stderr=None)


def queue_weekdays(time, ticker):
    schedule.every().monday.at(time).do(run_on_ticker, ticker)
    schedule.every().tuesday.at(time).do(run_on_ticker, ticker)
    schedule.every().wednesday.at(time).do(run_on_ticker, ticker)
    schedule.every().thursday.at(time).do(run_on_ticker, ticker)
    schedule.every().friday.at(time).do(run_on_ticker, ticker)


def queue_system_run(system_id):
    # Class A, the tickers we care about trading the most.
    most_important_crypto_morning_A = ['ADA-USD', 'ETH-USD', 'COTI-USD', 'DOT1-USD', 'LINK-USD', 'ATOM1-USD',
                                       'MATIC-USD', 'ALGO-USD']
    most_important_stock_close_A = ['SPY', 'AMD', 'NVDA', 'SNOW', 'MSFT', 'GOOGL', 'INTC', 'TSLA']
    # Class B, extra tickers we want to trade but aren't as interesting as class A
    most_important_crypto_morning_B = ['BTC-USD', 'ICP1-USD', 'MANA-USD', 'AAVE-USD', 'HBAR-USD', 'SHIB-USD',
                                       'DOGE-USD', 'SOL1-USD']
    most_important_stock_close_B = ['BAND', 'LULU', 'FVRR', 'SHOP', 'AAPL', 'ACN', 'UPST', 'EA']

    if not EMERGENCY_MODE:
        # NOTE: Only add 3 of each symbol for each timeframe, otherwise it will become much slower
        if system_id == 1:
            # Crypto Morning A
            for i in range(4):
                schedule.every().day.at(crypto_morning_A).do(run_on_ticker, most_important_crypto_morning_A[i])
            # Crypto Morning B
            for i in range(4):
                schedule.every().day.at(crypto_morning_B).do(run_on_ticker, most_important_crypto_morning_B[i])
            # Stock Close A
            for i in range(4):
                queue_weekdays(stock_close_A, most_important_stock_close_A[i])
            # Stock Close B
            for i in range(4):
                queue_weekdays(stock_close_B, most_important_stock_close_B[i])

        if system_id == 2:
            # Crypto Morning A
            for i in range(4, 8):
                schedule.every().day.at(crypto_morning_A).do(run_on_ticker, most_important_crypto_morning_A[i])
            # Crypto Morning B
            for i in range(4, 8):
                schedule.every().day.at(crypto_morning_B).do(run_on_ticker, most_important_crypto_morning_B[i])
            # Stock Close A
            for i in range(4, 8):
                queue_weekdays(stock_close_A, most_important_stock_close_A[i])
            # Stock Close B
            for i in range(4, 8):
                queue_weekdays(stock_close_B, most_important_stock_close_B[i])
    else:
        # EMERGENCY_MODE runs all the important symbols on one machine, split by time
        # Crypto Morning A
        for i in range(4):
            schedule.every().day.at(crypto_morning_A).do(run_on_ticker, most_important_crypto_morning_A[i])
        # Crypto Morning A in the crypto morning B timeslot
        for i in range(4, 8):
            schedule.every().day.at(crypto_morning_B).do(run_on_ticker, most_important_crypto_morning_A[i])
        # Stock Close A
        for i in range(4):
            queue_weekdays(stock_close_A, most_important_stock_close_A[i])
        # Stock Close A in the stock close B timeslot
        for i in range(4, 8):
            queue_weekdays(stock_close_B, most_important_stock_close_A[i])


queue_system_run(SYSTEM_ID)


while True:
    schedule.run_pending()
    time.sleep(60)
