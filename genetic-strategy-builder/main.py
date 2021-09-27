import argparse
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import os
import random
import time

from lib.Candidate import Candidate
from lib.StrategyTester import StrategyTester
from threading import Thread
from TickerData import TickerData



# parse arguments
parser = argparse.ArgumentParser(description='Find That Setup')
parser.add_argument('--tickers', nargs="+", dest="TICKERS", required=True,
                    help="Stock tickers to find trading setups for. Ex: --tickers AMD GOOGL INTC")
args = parser.parse_args()


# Save arguments from parser
TICKERS = args.TICKERS


# Set randomizer seeds for consistent results between runs
seed = 314
np.random.seed(seed)
random.seed(seed)

# Grab the current date
date_time_start = time.strftime("%Y-%m-%d_%H:%M:%S")

# Save the tickers to a list all uppercase
tickers = [ticker.upper() for ticker in TICKERS]


# Create the data/ subfolder if it does not already exist
if not os.path.isdir("data"):
    os.mkdir("data")


# Initialize TickerData, passing a list of tickers to load
ticker_data = TickerData(tickers=tickers)





tester = StrategyTester()

population_size = 10
results = []
pool = ThreadPoolExecutor(max_workers=10)
for _ in range(population_size):
    # Mock data here for strategy tester
    m_candidate = Candidate()

    t = pool.submit(tester.test_strategy, results, ticker_data, m_candidate)

pool.shutdown(wait=True)








import pdb
pdb.set_trace()